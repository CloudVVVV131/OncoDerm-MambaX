from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from lesionmamba.analysis.efficiency import count_params, estimate_flops, latency_throughput, model_size_mb
from lesionmamba.engine.checkpoint import load_checkpoint
from lesionmamba.models.registry import build_model
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.env import environment_summary
from lesionmamba.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-runs", default="outputs/runs")
    parser.add_argument("--config", default="configs/eval/efficiency.yaml")
    args = parser.parse_args()
    eff_cfg = load_yaml(args.config) if Path(args.config).exists() else {}
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for run_dir in Path(args.all_runs).glob("*"):
        if not run_dir.is_dir():
            continue
        config = load_yaml(run_dir / "config.yaml")
        model = build_model(config).to(device)
        ckpt_path = run_dir / "checkpoints" / "best_val_macro_f1.pth"
        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
        eval_cfg = config.get("eval", {})
        input_size = int(eval_cfg.get("input_size", config["data"].get("input_size", eff_cfg.get("input_size", 224))))
        precision = str(eval_cfg.get("precision", eff_cfg.get("precision", "amp"))).lower()
        batch_size_latency = int(eval_cfg.get("batch_size_latency", eff_cfg.get("batch_size_latency", 1)))
        batch_size_throughput = int(eval_cfg.get("batch_size_throughput", eff_cfg.get("batch_size_throughput", 64)))
        warmup_iters = int(eval_cfg.get("warmup_iters", eff_cfg.get("warmup_iters", 30)))
        measure_iters = int(eval_cfg.get("measure_iters", eff_cfg.get("measure_iters", 100)))
        eff = latency_throughput(
            model,
            input_size=input_size,
            batch_size_latency=batch_size_latency,
            batch_size_throughput=batch_size_throughput,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            amp=precision == "amp",
        )
        rows.append(
            {
                "run_id": run_dir.name,
                "input_size": input_size,
                "precision": precision,
                "batch_size_latency": batch_size_latency,
                "batch_size_throughput": batch_size_throughput,
                "warmup_iters": warmup_iters,
                "measure_iters": measure_iters,
                "params": count_params(model),
                "flops": estimate_flops(model, input_size),
                "model_size_mb": model_size_mb(ckpt_path),
                **eff,
            }
        )
    ensure_dir("results")
    columns = [
        "run_id",
        "input_size",
        "precision",
        "batch_size_latency",
        "batch_size_throughput",
        "warmup_iters",
        "measure_iters",
        "params",
        "flops",
        "model_size_mb",
        "latency_ms_b1",
        "throughput_img_s",
        "peak_mem_mb",
    ]
    pd.DataFrame(rows, columns=columns if not rows else None).to_csv("results/efficiency_table.csv", index=False)
    write_json(environment_summary(), "results/efficiency_meta.json")
    print(f"wrote {len(rows)} efficiency rows")


if __name__ == "__main__":
    main()
