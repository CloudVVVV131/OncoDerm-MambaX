from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from lesionmamba.analysis.plots import save_binary_roc_pr, save_confusion_matrix, save_efficiency_tradeoff
from lesionmamba.data.datasets import LABEL_MAP
from lesionmamba.metrics.bootstrap import bootstrap_ci
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.env import environment_summary
from lesionmamba.utils.io import ensure_dir, read_json, write_json


CORE_MODEL_NAMES = {
    "resnet50",
    "convnext_tiny",
    "efficientnet_b0",
    "swin_tiny",
    "vmamba_tiny",
    "lesionmamba_compact",
}


BASE_TABLE_COLUMNS = [
    "run_id",
    "Model",
    "experiment_role",
    "ablation_id",
    "Acc",
    "BAcc",
    "Macro-F1",
    "Macro-F1_ci_lower",
    "Macro-F1_ci_upper",
    "Macro-AUC",
    "Macro-AUC_ci_lower",
    "Macro-AUC_ci_upper",
    "Mel-AUC",
    "Mel-AUC_ci_lower",
    "Mel-AUC_ci_upper",
    "Mel-Sens",
    "Mel-Sens_ci_lower",
    "Mel-Sens_ci_upper",
    "Params",
    "FLOPs",
    "model_size_mb",
    "latency_ms_b1",
    "throughput_img_s",
    "peak_mem_mb",
    "precision",
    "config_path",
    "fallback_flag",
]


def experiment_role(cfg: dict, model_name: str) -> str:
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    if model_cfg.get("ablation_id"):
        return "ablation"
    if str(model_name).startswith("oncoderm_mambax_b"):
        return "submission_v1"
    if eval_cfg.get("p2_highres"):
        return "p2_highres"
    if eval_cfg.get("p2_second_seed"):
        return "p2_second_seed"
    if model_name.startswith("lesionmamba_loss_"):
        return "loss_ablation"
    if model_name in CORE_MODEL_NAMES:
        return "core"
    return "optional_or_extra"


def safe_metric(fn: Callable[[np.ndarray, np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], float]:
    def wrapped(labels: np.ndarray, probs: np.ndarray) -> float:
        try:
            return float(fn(labels, probs))
        except Exception:
            return float("nan")

    return wrapped


def ci_columns(run_id: str, iters: int = 1000) -> dict:
    pred_dir = Path("outputs/predictions")
    labels_path = pred_dir / f"{run_id}_test_labels.npy"
    probs_path = pred_dir / f"{run_id}_test_probs.npy"
    if not labels_path.exists() or not probs_path.exists():
        return {}
    labels = np.load(labels_path)
    probs = np.load(probs_path)
    if len(labels) == 0:
        return {}
    mel_idx = LABEL_MAP["mel"]
    funcs = {
        "Macro-F1": safe_metric(lambda y, p: f1_score(y, p.argmax(axis=1), average="macro", zero_division=0)),
        "Macro-AUC": safe_metric(
            lambda y, p: np.nanmean(
                [
                    roc_auc_score((y == i).astype(int), p[:, i])
                    for i in range(p.shape[1])
                    if len(np.unique((y == i).astype(int))) > 1
                ]
            )
        ),
        "Mel-AUC": safe_metric(lambda y, p: roc_auc_score((y == mel_idx).astype(int), p[:, mel_idx])),
        "Mel-Sens": safe_metric(
            lambda y, p: (
                ((p.argmax(axis=1) == mel_idx) & (y == mel_idx)).sum() / max((y == mel_idx).sum(), 1)
            )
        ),
    }
    out = {}
    for name, fn in funcs.items():
        ci = bootstrap_ci(labels, probs, fn, iters=iters)
        out[f"{name}_ci_lower"] = ci["ci_lower"]
        out[f"{name}_ci_upper"] = ci["ci_upper"]
    return out


def run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    return load_yaml(cfg_path) if cfg_path.exists() else {}


def collect_all_run_rows() -> pd.DataFrame:
    rows = []
    run_dirs = [p for p in Path("outputs/runs").glob("*") if p.is_dir()]
    for run_dir in run_dirs:
        cfg = run_config(run_dir)
        metrics = read_json(run_dir / "metrics_test.json", default={})
        env = read_json(run_dir / "environment.json", default={})
        model_meta = env.get("model_metadata", {})
        model_name = model_meta.get("model_name", cfg.get("model", {}).get("name", run_dir.name.split("_seed")[0]))
        bootstrap_iters = int(cfg.get("eval", {}).get("bootstrap_iters", 1000))
        row = {
            "run_id": run_dir.name,
            "Model": model_name,
            "experiment_role": experiment_role(cfg, model_name),
            "ablation_id": cfg.get("model", {}).get("ablation_id"),
            "Acc": metrics.get("accuracy"),
            "BAcc": metrics.get("balanced_accuracy"),
            "Macro-F1": metrics.get("macro_f1"),
            "Macro-AUC": metrics.get("macro_auc"),
            "Mel-AUC": metrics.get("mel_auc"),
            "Mel-Sens": metrics.get("mel_sensitivity"),
            "config_path": str(run_dir / "config.yaml"),
            "fallback_flag": model_meta.get("not_formal_vmamba")
            or ("torch_fallback" in model_meta.get("effective_ssm_backends", {}))
            or ("torch_fallback" in str(model_meta.get("effective_ssm_backend", ""))),
        }
        row.update(ci_columns(run_dir.name, bootstrap_iters))
        rows.append(row)
        if "confusion_matrix" in metrics:
            save_confusion_matrix(
                metrics["confusion_matrix"],
                list(LABEL_MAP.keys()),
                Path("figures/confusion_matrix") / f"{run_dir.name}.png",
            )
        labels_path = Path("outputs/predictions") / f"{run_dir.name}_test_labels.npy"
        probs_path = Path("outputs/predictions") / f"{run_dir.name}_test_probs.npy"
        if labels_path.exists() and probs_path.exists():
            labels = np.load(labels_path)
            probs = np.load(probs_path)
            save_binary_roc_pr(
                (labels == LABEL_MAP["mel"]).astype(int),
                probs[:, LABEL_MAP["mel"]],
                Path("figures/roc_pr_curves") / run_dir.name,
                f"{model_name} melanoma endpoint",
            )
    all_df = pd.DataFrame(rows)
    if all_df.empty:
        all_df = pd.DataFrame(columns=BASE_TABLE_COLUMNS)
    eff_path = Path("results/efficiency_table.csv")
    if eff_path.exists() and not all_df.empty:
        try:
            eff = pd.read_csv(eff_path)
            keep = [
                "run_id",
                "params",
                "flops",
                "model_size_mb",
                "latency_ms_b1",
                "throughput_img_s",
                "peak_mem_mb",
                "precision",
            ]
            if not eff.empty and "run_id" in eff.columns:
                all_df = all_df.merge(eff[[c for c in keep if c in eff.columns]], on="run_id", how="left")
                all_df = all_df.rename(columns={"params": "Params", "flops": "FLOPs"})
        except pd.errors.EmptyDataError:
            pass
    for col in BASE_TABLE_COLUMNS:
        if col not in all_df.columns:
            all_df[col] = pd.NA
    all_df.to_csv("results/all_runs_metrics_table.csv", index=False)
    return all_df


def make_main_table(all_df: pd.DataFrame) -> pd.DataFrame:
    main_df = all_df[all_df["experiment_role"] == "core"].copy()
    if main_df.empty:
        main_df = pd.DataFrame(columns=BASE_TABLE_COLUMNS)
    else:
        main_df = main_df[[c for c in BASE_TABLE_COLUMNS if c in main_df.columns]]
        save_efficiency_tradeoff(main_df, "figures/efficiency/efficiency_tradeoff.png")
    main_df.to_csv("results/main_7class_table.csv", index=False)
    return main_df


def make_ablation_table(all_df: pd.DataFrame) -> None:
    rows = []
    for run_dir in Path("outputs/runs").glob("*"):
        if not run_dir.is_dir():
            continue
        cfg = run_config(run_dir)
        ablation_id = cfg.get("model", {}).get("ablation_id")
        if not ablation_id:
            continue
        model_cfg = cfg.get("model", {})
        stages = model_cfg.get("stages", {})
        stage2 = stages.get("stage2", {})
        row = {
            "run_id": run_dir.name,
            "Setting": ablation_id,
            "MPLE": bool(stage2.get("use_mple", False)),
            "DP-MSS": stage2.get("scan_mode") == "dual_path" and bool(stage2.get("use_dpmss", False)),
            "LMGF": bool(stage2.get("use_lmgf", False)),
            "Lesion Pooling": bool(model_cfg.get("use_lesion_pooling", False)),
        }
        match = all_df[all_df["run_id"] == run_dir.name]
        if not match.empty:
            for col in ["Macro-F1", "Macro-AUC", "Mel-AUC", "Params", "FLOPs"]:
                if col in match.columns:
                    row[col] = match.iloc[0].get(col)
        rows.append(row)
    ablation_df = pd.DataFrame(rows)
    if ablation_df.empty:
        ablation_df = pd.DataFrame(
            columns=["run_id", "Setting", "MPLE", "DP-MSS", "LMGF", "Lesion Pooling", "Macro-F1", "Macro-AUC", "Mel-AUC", "Params", "FLOPs"]
        )
    if not ablation_df.empty:
        ablation_df = ablation_df.sort_values("Setting")
    ablation_df.to_csv("results/ablation_table.csv", index=False)


def write_report_metadata(main_df: pd.DataFrame, all_df: pd.DataFrame) -> None:
    pretraining_status = {}
    fallback_notes = {}
    for run_dir in Path("outputs/runs").glob("*"):
        if not run_dir.is_dir():
            continue
        env = read_json(run_dir / "environment.json", default={})
        meta = env.get("model_metadata", {})
        pretraining_status[run_dir.name] = meta.get("pretrained", meta.get("pretraining_status", "unknown"))
        if (
            meta.get("not_formal_vmamba")
            or "torch_fallback" in meta.get("effective_ssm_backends", {})
            or "torch_fallback" in str(meta.get("effective_ssm_backend", ""))
        ):
            fallback_notes[run_dir.name] = meta
    write_json(
        {
            "paper_title": "OncoDerm-MambaX: Clinically Gated Cross-Scale State-Space Adaptation for Explainable Dermoscopic Melanoma Assessment",
            "legacy_project_title": "LesionMamba: A Multi-Scale State Space Architecture for Explainable Skin Lesion Classification in Dermoscopic Images",
            "submission_v1_scope": "supplementary submission-track experiments with B0-B6 OncoDerm-MambaX ablations, melanoma endpoint head, CMGF metadata fusion, CSSA, and LPAH mask auxiliary supervision",
            "dataset_names": ["HAM10000", "ISIC_external_if_available", "ISIC_masks_if_available"],
            "split_seed": 42,
            "hardware": environment_summary(),
            "software_versions": environment_summary(),
            "pretraining_status": pretraining_status,
            "fallback_notes": fallback_notes,
            "clinical_scope_note": "computer-aided dermoscopic image classification / decision support; not an autonomous diagnostic system",
            "main_table_rows": int(len(main_df)),
            "all_runs_table_rows": int(len(all_df)),
        },
        "results/report_metadata.json",
    )


def main() -> None:
    ensure_dir("results")
    all_df = collect_all_run_rows()
    main_df = make_main_table(all_df)
    make_ablation_table(all_df)
    write_report_metadata(main_df, all_df)
    write_json({"bootstrap_source": "saved test predictions", "ci": 0.95}, "results/bootstrap_meta.json")
    print(f"wrote main table rows={len(main_df)}")


if __name__ == "__main__":
    main()
