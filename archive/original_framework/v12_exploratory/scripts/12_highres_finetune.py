from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from lesionmamba.engine.checkpoint import load_checkpoint
from lesionmamba.engine.evaluate import evaluate_and_save
from lesionmamba.engine.train import train_one_run
from lesionmamba.engine.train import build_dataloader
from lesionmamba.models.registry import build_model
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, read_json, write_json


def is_main_lesionmamba(run_dir: Path, cfg: dict) -> bool:
    name = str(cfg.get("model", {}).get("name", run_dir.name))
    if cfg.get("eval", {}).get("p2_highres") or cfg.get("eval", {}).get("p2_second_seed"):
        return False
    return name == "lesionmamba_compact"


def is_core_baseline(run_dir: Path, cfg: dict) -> bool:
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", run_dir.name))
    if cfg.get("eval", {}).get("p2_highres") or cfg.get("eval", {}).get("p2_second_seed"):
        return False
    if name.startswith("lesionmamba"):
        return False
    return model_cfg.get("source") in {"timm", "vmamba"} and "loss_" not in name


def run_score(run_dir: Path) -> float:
    metrics = read_json(run_dir / "metrics_test.json", default={}) or read_json(run_dir / "metrics_val.json", default={})
    value = metrics.get("macro_f1")
    return float(value) if value is not None else -1.0


def select_source_runs(runs_dir: Path) -> tuple[Path | None, Path | None]:
    baseline: tuple[float, Path] | None = None
    lesion: tuple[float, Path] | None = None
    for run_dir in runs_dir.glob("*"):
        if not run_dir.is_dir() or not (run_dir / "config.yaml").exists():
            continue
        cfg = load_yaml(run_dir / "config.yaml")
        score = run_score(run_dir)
        ckpt = run_dir / "checkpoints" / "best_val_macro_f1.pth"
        if not ckpt.exists():
            continue
        if is_main_lesionmamba(run_dir, cfg):
            if lesion is None or score > lesion[0]:
                lesion = (score, run_dir)
        elif is_core_baseline(run_dir, cfg):
            if baseline is None or score > baseline[0]:
                baseline = (score, run_dir)
    return (baseline[1] if baseline else None, lesion[1] if lesion else None)


def highres_config_from_run(run_dir: Path, highres_cfg: dict) -> dict:
    cfg = copy.deepcopy(load_yaml(run_dir / "config.yaml"))
    input_size = int(highres_cfg.get("input_size", 384))
    epochs = int(highres_cfg.get("epochs", 20))
    cfg["data"]["input_size"] = input_size
    cfg["data"]["resize_short_side"] = max(int(round(input_size * 256 / 224)), input_size)
    cfg["train"]["epochs"] = epochs
    cfg["train"]["early_stopping_patience"] = min(int(cfg["train"].get("early_stopping_patience", 8)), 5)
    cfg["train"]["batch_size"] = min(int(cfg["train"].get("batch_size", 64)), int(highres_cfg.get("batch_size", 32)))
    cfg["train"]["init_checkpoint"] = str(run_dir / "checkpoints" / "best_val_macro_f1.pth")
    cfg["train"]["fine_tune_from_run"] = run_dir.name
    cfg["model"]["name"] = f"{cfg['model'].get('name', run_dir.name)}_highres384"
    cfg.setdefault("eval", {})["p2_highres"] = True
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
    parser.add_argument("--config", default="configs/eval/highres_384_finetune.yaml")
    parser.add_argument("--runs-dir", default="outputs/runs")
    args = parser.parse_args()
    highres_cfg = load_yaml(args.config)
    runs_dir = Path(args.runs_dir)
    ensure_dir("results")
    baseline_run, lesion_run = select_source_runs(runs_dir)
    selected = [p for p in [baseline_run, lesion_run] if p is not None]
    columns = ["source_run", "highres_run", "status", "test_macro_f1", "test_mel_auc", "error"]
    if not selected:
        pd.DataFrame(columns=columns).to_csv("results/highres_384_table.csv", index=False)
        write_json(
            {
                "status": "not_run",
                "reason": "No completed source runs with best_val_macro_f1.pth were found.",
                "config": highres_cfg,
            },
            "results/highres_384_status.json",
        )
        print("highres skipped: no source runs")
        return
    rows = []
    for source_run in selected:
        cfg = highres_config_from_run(source_run, highres_cfg)
        try:
            new_run_dir = train_one_run(cfg)
            metrics = evaluate_run(new_run_dir)
            rows.append(
                {
                    "source_run": source_run.name,
                    "highres_run": new_run_dir.name,
                    "status": "completed",
                    "test_macro_f1": metrics.get("test", {}).get("macro_f1"),
                    "test_mel_auc": metrics.get("test", {}).get("mel_auc"),
                }
            )
        except Exception as exc:
            rows.append({"source_run": source_run.name, "status": "failed", "error": str(exc)})
    pd.DataFrame(rows, columns=columns if not rows else None).to_csv("results/highres_384_table.csv", index=False)
    write_json(
        {
            "status": "completed_or_partial",
            "input_size": highres_cfg.get("input_size", 384),
            "selected_runs": rows,
        },
        "results/highres_384_status.json",
    )
    print(f"highres fine-tuning rows={len(rows)}")


if __name__ == "__main__":
    main()
