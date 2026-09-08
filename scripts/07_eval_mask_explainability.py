from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
import torch

from lesionmamba.analysis.gradcam import compute_gradcam
from lesionmamba.analysis.mask_metrics import lesion_background_ratio, pointing_game_hit, saliency_mask_iou
from lesionmamba.data.transforms import build_transforms
from lesionmamba.engine.checkpoint import load_checkpoint
from lesionmamba.models.registry import build_model_for_checkpoint
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, write_json


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-6)


def find_saliency_map(run_id: str, image_id: str, attention_dir: Path, gradcam_dir: Path) -> Path | None:
    attention_candidates = list((attention_dir / run_id).glob(f"{image_id}.*"))
    if attention_candidates:
        return attention_candidates[0]
    gradcam_candidates = list(gradcam_dir.glob(f"{run_id}_{image_id}_gradcam.png"))
    if gradcam_candidates:
        return gradcam_candidates[0]
    return None


def image_id_from_mask(mask_path: Path) -> str:
    stem = mask_path.stem
    for suffix in [
        "_segmentation",
        "-segmentation",
        "_ground_truth",
        "-ground-truth",
        "_groundtruth",
        "-groundtruth",
        "_mask",
        "-mask",
        "_lesion",
        "_gt",
        "-gt",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def collect_mask_files(mask_dir: Path) -> list[Path]:
    files = sorted(
        list(mask_dir.rglob("*.png")) + list(mask_dir.rglob("*.jpg")) + list(mask_dir.rglob("*.jpeg")),
        key=lambda path: str(path).lower(),
    )
    preferred = [
        p
        for p in files
        if any(token in p.stem.lower() for token in ["segmentation", "mask", "groundtruth", "ground_truth"])
    ]
    return preferred if preferred else files


def unique_paired_masks(mask_files: list[Path], image_index: dict[str, Path]) -> list[Path]:
    paired: dict[str, Path] = {}
    for path in mask_files:
        image_id = image_id_from_mask(path)
        if image_id in image_index:
            paired.setdefault(image_id, path)
    return [paired[image_id] for image_id in sorted(paired)]


def build_image_index(paths: list[Path], mask_files: list[Path]) -> dict[str, Path]:
    mask_set = {p.resolve() for p in mask_files if p.exists()}
    index: dict[str, Path] = {}
    for root in paths:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if p.resolve() in mask_set:
                continue
            if any(token in p.stem.lower() for token in ["segmentation", "mask", "groundtruth", "ground_truth"]):
                continue
            index.setdefault(p.stem, p)
    return index


def attention_to_array(attention: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    arr = attention.detach().float().cpu()[0, 0].numpy()
    arr = normalize_map(arr)
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize(size)
    return np.array(img)


def saliency_to_uint8_image(saliency: np.ndarray | torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    if isinstance(saliency, torch.Tensor):
        arr = saliency.detach().float().cpu().numpy()
    else:
        arr = np.asarray(saliency)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[0] <= 4:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D saliency map after squeezing, got shape {arr.shape}")
    arr = normalize_map(arr)
    return np.array(Image.fromarray((arr * 255).astype(np.uint8)).resize(size))


def compute_and_save_saliency(
    model: torch.nn.Module,
    image_path: Path,
    transform,
    device: str,
    out_path: Path,
) -> tuple[Path, str]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    ensure_dir(out_path.parent)
    if out.get("attention") is not None:
        sal = attention_to_array(out["attention"], image.size)
        source = "internal_attention"
    else:
        cam, _, _ = compute_gradcam(model, tensor)
        sal = saliency_to_uint8_image(cam, image.size)
        source = "gradcam"
    Image.fromarray(sal.astype(np.uint8)).save(out_path)
    return out_path, source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-runs", default="outputs/runs")
    parser.add_argument("--run-ids", nargs="*", default=None, help="Optional explicit run directories.")
    parser.add_argument("--max-masks", type=int, default=0, help="0 means use every available mask.")
    parser.add_argument(
        "--min-masks",
        type=int,
        default=0,
        help="Write a not-run status unless at least this many image-mask pairs are available.",
    )
    parser.add_argument("--result-stem", default="explainability_mask")
    args = parser.parse_args()
    mask_dir = Path("data/raw/ISIC_masks")
    ensure_dir("results")
    status_path = Path("results") / f"{args.result_stem}_status.json"
    table_path = Path("results") / f"{args.result_stem}_table.csv"
    mask_files = collect_mask_files(mask_dir)
    if not mask_files:
        write_json(
            {"status": "not_run", "reason": "No mask files found under data/raw/ISIC_masks"},
            status_path,
        )
        print("mask explainability skipped: no masks")
        return
    image_index = build_image_index(
        [Path("data/raw/ISIC_masks"), Path("data/raw/HAM10000/images"), Path("data/raw/ISIC_external/images")],
        mask_files,
    )
    mask_files = unique_paired_masks(mask_files, image_index)
    available_pairs = len(mask_files)
    if args.min_masks > 0 and available_pairs < args.min_masks:
        table_path.unlink(missing_ok=True)
        write_json(
            {
                "status": "not_run_insufficient_pairs",
                "required_pairs": args.min_masks,
                "available_pairs": available_pairs,
                "reason": "Available image-mask pairs are below the requested --min-masks count.",
            },
            status_path,
        )
        print(f"mask explainability skipped: required={args.min_masks} available_pairs={available_pairs}")
        return
    if args.max_masks > 0:
        mask_files = mask_files[: args.max_masks]
    rows = []
    statuses = []
    att_dir = Path("outputs/attention_maps")
    gradcam_dir = Path("figures/explainability/qualitative")
    generated_dir = ensure_dir(Path("figures/explainability") / f"{args.result_stem}_saliency")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dirs = [Path(item) for item in args.run_ids] if args.run_ids else list(Path(args.selected_runs).glob("*"))
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        model = None
        transform = None
        scores = []
        generated = 0
        missing_images = 0
        saliency_failures = 0
        first_saliency_error = None
        for mask_path in mask_files:
            image_id = image_id_from_mask(mask_path)
            saliency_path = find_saliency_map(run_dir.name, image_id, att_dir, gradcam_dir)
            source = "precomputed_saliency"
            if saliency_path is None:
                image_path = image_index.get(image_id)
                if image_path is None:
                    missing_images += 1
                    continue
                if model is None:
                    try:
                        config = load_yaml(run_dir / "config.yaml")
                        transform = build_transforms(config, "eval")
                        model = build_model_for_checkpoint(config).to(device).eval()
                        ckpt_path = run_dir / "checkpoints" / "best_val_macro_f1.pth"
                        if not ckpt_path.exists():
                            statuses.append({"run_id": run_dir.name, "status": "skipped", "reason": "missing_checkpoint"})
                            break
                        model.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"], strict=True)
                    except Exception as exc:
                        statuses.append({"run_id": run_dir.name, "status": "failed", "error": str(exc)})
                        break
                try:
                    saliency_path, source = compute_and_save_saliency(
                        model,
                        image_path,
                        transform,
                        device,
                        generated_dir / f"{run_dir.name}_{image_id}.png",
                    )
                except Exception as exc:
                    saliency_failures += 1
                    if first_saliency_error is None:
                        first_saliency_error = str(exc)
                    continue
                generated += 1
            if saliency_path is None:
                continue
            sal = np.array(Image.open(saliency_path).convert("L").resize(Image.open(mask_path).size))
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            scores.append(
                {
                    "iou": saliency_mask_iou(sal, mask),
                    "pointing_hit": pointing_game_hit(sal, mask),
                    "lesion_background_ratio": lesion_background_ratio(sal, mask),
                    "source": source,
                }
            )
        if scores:
            df = pd.DataFrame(scores)
            rows.append(
                {
                    "run_id": run_dir.name,
                    "n": len(scores),
                    "saliency_mask_iou": df["iou"].mean(),
                    "pointing_game_hit_rate": df["pointing_hit"].mean(),
                    "lesion_background_saliency_ratio": df["lesion_background_ratio"].mean(),
                    "generated_saliency_maps": generated,
                    "missing_image_count": missing_images,
                    "saliency_failure_count": saliency_failures,
                    "first_saliency_error": first_saliency_error,
                }
            )
            statuses.append(
                {
                    "run_id": run_dir.name,
                    "status": "completed",
                    "matched_masks": len(scores),
                    "generated_saliency_maps": generated,
                    "missing_image_count": missing_images,
                    "saliency_failure_count": saliency_failures,
                    "first_saliency_error": first_saliency_error,
                }
            )
        else:
            rows.append(
                {
                    "run_id": run_dir.name,
                    "n": 0,
                    "generated_saliency_maps": generated,
                    "missing_image_count": missing_images,
                    "saliency_failure_count": saliency_failures,
                    "first_saliency_error": first_saliency_error,
                    "status": "no_matching_images_or_saliency_maps",
                }
            )
            if not any(s.get("run_id") == run_dir.name for s in statuses):
                statuses.append(
                    {
                        "run_id": run_dir.name,
                        "status": "no_matching_images_or_saliency_maps",
                        "missing_image_count": missing_images,
                        "saliency_failure_count": saliency_failures,
                        "first_saliency_error": first_saliency_error,
                    }
                )
    columns = [
        "run_id",
        "n",
        "saliency_mask_iou",
        "pointing_game_hit_rate",
        "lesion_background_saliency_ratio",
        "generated_saliency_maps",
        "missing_image_count",
        "saliency_failure_count",
        "first_saliency_error",
        "status",
    ]
    pd.DataFrame(rows, columns=columns if not rows else None).to_csv(table_path, index=False)
    write_json(
        {
            "status": "completed_or_partial",
            "available_image_mask_pairs": available_pairs,
            "evaluated_mask_count": len(mask_files),
            "minimum_requested": args.min_masks,
            "runs": statuses,
        },
        status_path,
    )
    print(f"wrote mask explainability status for {len(rows)} runs")


if __name__ == "__main__":
    main()
