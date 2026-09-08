from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from lesionmamba.analysis.efficiency import count_params, estimate_flops, latency_throughput, model_size_mb
from lesionmamba.engine.checkpoint import load_checkpoint
from lesionmamba.models.registry import build_model_for_checkpoint
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.env import environment_summary
from lesionmamba.utils.io import ensure_dir, write_json


def iter_run_dirs(base_dir: Path, selected_runs: list[str] | None, run_glob: str | None) -> list[Path]:
    if selected_runs:
        return [Path(run) for run in selected_runs]
    if run_glob:
        return sorted(base_dir.glob(run_glob))
    return sorted(path for path in base_dir.glob("*") if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-runs", default="outputs/runs")
    parser.add_argument("--config", default="configs/eval/efficiency.yaml")
    parser.add_argument("--selected-runs", nargs="*", default=None)
    parser.add_argument("--run-glob", default=None)
    args = parser.parse_args()
    eff_cfg = load_yaml(args.config) if Path(args.config).exists() else {}
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for run_dir in iter_run_dirs(Path(args.all_runs), args.selected_runs, args.run_glob):
        if not run_dir.is_dir() or not (run_dir / "config.yaml").exists():
            continue
        config = load_yaml(run_dir / "config.yaml")
        ckpt_path = run_dir / "checkpoints" / "best_val_macro_f1.pth"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Validation-selected checkpoint required: {ckpt_path}")
        model = build_model_for_checkpoint(config).to(device)
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
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
    new_df = pd.DataFrame(rows, columns=columns if not rows else None)
    out_path = Path("results/efficiency_table.csv")
    merge_existing = bool(args.selected_runs or args.run_glob)
    if merge_existing and out_path.exists() and not new_df.empty:
        old_df = pd.read_csv(out_path)
        if "run_id" in old_df.columns:
            old_df = old_df[~old_df["run_id"].isin(new_df["run_id"])]
        out_df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        out_df = new_df
    out_df.to_csv(out_path, index=False)
    write_json(environment_summary(), "results/efficiency_meta.json")
    print(f"wrote {len(out_df)} efficiency rows; new_or_updated={len(new_df)}")


if __name__ == "__main__":
    main()
