from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from lesionmamba.data.splits import make_splits
from lesionmamba.engine.checkpoint import load_checkpoint
from lesionmamba.engine.evaluate import evaluate_and_save
from lesionmamba.engine.train import train_one_run
from lesionmamba.engine.train import build_dataloader
from lesionmamba.models.registry import build_model
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, read_json, write_json


def run_score(run_dir: Path) -> float:
    metrics = read_json(run_dir / "metrics_test.json", default={}) or read_json(run_dir / "metrics_val.json", default={})
    value = metrics.get("macro_f1")
    return float(value) if value is not None else -1.0


def classify_run(run_dir: Path, cfg: dict) -> str:
    name = str(cfg.get("model", {}).get("name", run_dir.name))
    source = cfg.get("model", {}).get("source")
    if cfg.get("eval", {}).get("p2_highres") or cfg.get("eval", {}).get("p2_second_seed"):
        return "skip"
    if name == "lesionmamba_compact":
        return "lesionmamba"
    if name.startswith("lesionmamba"):
        return "skip"
    if source in {"timm", "vmamba"}:
        return "baseline"
    return "skip"


def select_top2(runs_dir: Path) -> list[Path]:
    best: dict[str, tuple[float, Path]] = {}
    for run_dir in runs_dir.glob("*"):
        if not run_dir.is_dir() or not (run_dir / "config.yaml").exists():
            continue
        cfg = load_yaml(run_dir / "config.yaml")
        kind = classify_run(run_dir, cfg)
        if kind == "skip":
            continue
        score = run_score(run_dir)
        if kind not in best or score > best[kind][0]:
            best[kind] = (score, run_dir)
    return [best[k][1] for k in ["baseline", "lesionmamba"] if k in best]


def second_seed_config(source_run: Path, second_cfg: dict) -> dict:
    cfg = copy.deepcopy(load_yaml(source_run / "config.yaml"))
    seed = int(second_cfg.get("seed", 2024))
    cfg["project"]["seed"] = seed
    cfg["train"]["epochs"] = int(second_cfg.get("epochs", 30))
    cfg["train"]["early_stopping_patience"] = min(int(cfg["train"].get("early_stopping_patience", 8)), 6)
    cfg["train"].pop("init_checkpoint", None)
    cfg["train"]["second_seed_from_run"] = source_run.name
    cfg["model"]["name"] = f"{cfg['model'].get('name', source_run.name)}_seed{seed}"
    cfg.setdefault("eval", {})["p2_second_seed"] = True
    return cfg


def evaluate_run(run_dir: Path) -> dict[str, dict]:
    cfg = load_yaml(run_dir / "config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    ckpt_path = run_dir / "checkpoints" / "best_val_macro_f1.pth"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pth"
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    results = {}
    for split in ["val", "test"]:
        loader = build_dataloader(cfg, split, shuffle=False)
        results[split] = evaluate_and_save(
            model,
            loader,
            device,
            run_id=run_dir.name,
            split=split,
            run_dir=run_dir,
            amp=bool(cfg["train"].get("amp", True)),
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/second_seed_top2.yaml")
    parser.add_argument("--runs-dir", default="outputs/runs")
    args = parser.parse_args()
    second_cfg = load_yaml(args.config)
    ensure_dir("results")
    selected = select_top2(Path(args.runs_dir))
    columns = ["source_run", "second_seed_run", "status", "test_macro_f1", "test_mel_auc", "error"]
    if not selected:
        pd.DataFrame(columns=columns).to_csv("results/second_seed_top2_table.csv", index=False)
        write_json(
            {"status": "not_run", "reason": "No source runs available for top-2 second-seed selection.", "config": second_cfg},
            "results/second_seed_top2_status.json",
        )
        print("second-seed skipped: no source runs")
        return
    rows = []
    split_done = False
    for source_run in selected:
        cfg = second_seed_config(source_run, second_cfg)
        try:
            if not split_done:
                make_splits(cfg)
                split_done = True
            new_run_dir = train_one_run(cfg)
            metrics = evaluate_run(new_run_dir)
            rows.append(
                {
                    "source_run": source_run.name,
                    "second_seed_run": new_run_dir.name,
                    "status": "completed",
                    "test_macro_f1": metrics.get("test", {}).get("macro_f1"),
                    "test_mel_auc": metrics.get("test", {}).get("mel_auc"),
                }
            )
        except Exception as exc:
            rows.append({"source_run": source_run.name, "status": "failed", "error": str(exc)})
    pd.DataFrame(rows, columns=columns if not rows else None).to_csv("results/second_seed_top2_table.csv", index=False)
    write_json(
        {
            "status": "completed_or_partial",
            "seed": second_cfg.get("seed", 2024),
            "selected_runs": rows,
        },
        "results/second_seed_top2_status.json",
    )
    print(f"second-seed rows={len(rows)}")


if __name__ == "__main__":
    main()
