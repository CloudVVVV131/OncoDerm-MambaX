from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, read_json, write_json


SETTING_MAP = {
    "oncoderm_mambax_b0_backbone_only": "A0/B0 backbone-only",
    "oncoderm_mambax_b1_melanoma_head": "A1/B1 + melanoma head",
    "oncoderm_mambax_b2_metadata_cmgf": "A4/B2 + CMGF metadata",
    "oncoderm_mambax_b3_cssa": "A2/B3 + CSSA",
    "oncoderm_mambax_b4_lpah_mask_aux": "A3/B4 + LPAH mask aux",
    "oncoderm_mambax_b5_cssa_lpah": "A5/B5 + CSSA + LPAH",
    "oncoderm_mambax_b6_full": "A6/B6 full OncoDerm-MambaX",
}


def optional_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    ensure_dir("results")
    all_df = optional_table("results/all_runs_metrics_table.csv")
    if all_df.empty:
        print("results/all_runs_metrics_table.csv missing or empty; run scripts/09_make_tables_and_figures.py first")
        return
    external = optional_table("results/external_test_table.csv")
    calibration = optional_table("results/calibration_table.csv")
    explain = optional_table("results/explainability_mask_table.csv")
    efficiency = optional_table("results/efficiency_table.csv")
    rows = []
    for run_dir in sorted(Path("outputs/runs").glob("*")):
        if not run_dir.is_dir() or not (run_dir / "config.yaml").exists():
            continue
        cfg = load_yaml(run_dir / "config.yaml")
        model_name = cfg.get("model", {}).get("name", "")
        if not model_name.startswith("oncoderm_mambax_b"):
            continue
        meta = read_json(run_dir / "environment.json", default={}).get("model_metadata", {})
        base = all_df[all_df["run_id"] == run_dir.name]
        row = {
            "run_id": run_dir.name,
            "Setting": SETTING_MAP.get(model_name, model_name),
            "Model": model_name,
            "CSSA": bool(cfg.get("model", {}).get("cssa", {}).get("enabled", False)),
            "LPAH_Mask_Aux": bool(cfg.get("model", {}).get("lesion_prior", {}).get("enabled", False)),
            "CMGF_Metadata": bool(cfg.get("model", {}).get("metadata_fusion", {}).get("enabled", False)),
            "Melanoma_Head": bool(cfg.get("model", {}).get("heads", {}).get("melanoma_endpoint", False)),
            "Multi_Task": bool(cfg.get("loss", {}).get("multitask", False)),
            "Backbone": cfg.get("model", {}).get("timm_name"),
            "Pretrained": cfg.get("model", {}).get("pretrained"),
            "SSM_Backend": meta.get("effective_ssm_backend"),
            "Inference_Requires_Mask": meta.get("inference_requires_mask", False),
        }
        if not base.empty:
            for src, dst in [
                ("Acc", "Acc"),
                ("BAcc", "BAcc"),
                ("Macro-F1", "Macro-F1"),
                ("Macro-AUC", "Macro-AUC"),
                ("Mel-AUC", "Internal Mel-AUC"),
                ("Mel-Sens", "Mel-Sens"),
                ("Params", "Params"),
                ("FLOPs", "FLOPs"),
                ("latency_ms_b1", "Latency B=1"),
                ("throughput_img_s", "Throughput"),
            ]:
                if src in base.columns:
                    row[dst] = base.iloc[0].get(src)
        if not external.empty and "run_id" in external.columns:
            m = external[external["run_id"] == run_dir.name]
            if not m.empty:
                for src in ["External Mel-AUC", "Drop", "pr_auc", "sensitivity", "specificity", "brier", "ece"]:
                    if src in m.columns:
                        row[f"External {src}" if src in {"brier", "ece"} else src] = m.iloc[0].get(src)
        if not calibration.empty and "run_id" in calibration.columns:
            m = calibration[calibration["run_id"] == run_dir.name]
            if not m.empty:
                for src in ["temperature", "ece_before", "ece_after", "brier_before", "brier_after", "nll_before", "nll_after"]:
                    if src in m.columns:
                        row[src] = m.iloc[0].get(src)
        if not explain.empty and "run_id" in explain.columns:
            m = explain[explain["run_id"] == run_dir.name]
            if not m.empty:
                for src in ["n", "saliency_mask_iou", "pointing_game_hit_rate", "lesion_background_saliency_ratio"]:
                    if src in m.columns:
                        row[src] = m.iloc[0].get(src)
        if not efficiency.empty and "run_id" in efficiency.columns:
            m = efficiency[efficiency["run_id"] == run_dir.name]
            if not m.empty:
                for src in ["params", "flops", "model_size_mb", "latency_ms_b1", "throughput_img_s", "peak_mem_mb"]:
                    if src in m.columns and src not in row:
                        row[src] = m.iloc[0].get(src)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Setting")
    out.to_csv("results/submission_v1_ablation_table.csv", index=False)
    write_json(
        {
            "status": "completed_or_empty",
            "rows": len(out),
            "note": "B0-B6 submission V1 ablation table merging internal, external, calibration, explainability and efficiency outputs.",
        },
        "results/submission_v1_summary_status.json",
    )
    print(f"wrote submission_v1_ablation_table rows={len(out)}")


if __name__ == "__main__":
    main()

