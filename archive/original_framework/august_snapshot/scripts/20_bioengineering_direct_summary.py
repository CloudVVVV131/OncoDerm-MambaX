from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from lesionmamba.analysis.bioengineering_supplement import (
    DIRECT_MODEL_SPECS,
    OPTIONAL_MODEL_SPECS,
    DirectModelSpec,
    aligned_external,
    discover_run,
    efficiency_lookup,
    external_metrics,
    file_sha256,
    internal_metrics,
    load_prediction_triplet,
    paired_bootstrap_delta,
    prediction_path,
    run_provenance,
    validation_thresholds,
)
from lesionmamba.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--bootstrap-iters", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--include-efficientnet", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-paired-bootstrap", action="store_true")
    return parser.parse_args()


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def efficiency_values(row: dict | None) -> dict[str, float | None]:
    row = row or {}
    return {
        "Params": float(row["params"]) if finite(row.get("params")) else None,
        "FLOPs": float(row["flops"]) if finite(row.get("flops")) else None,
        "Latency_ms_b1": float(row["latency_ms_b1"]) if finite(row.get("latency_ms_b1")) else None,
        "Peak_memory_MB": float(row["peak_mem_mb"]) if finite(row.get("peak_mem_mb")) else None,
    }


def make_gate(summary: pd.DataFrame) -> dict:
    if "model_id" not in summary.columns:
        return {
            "status": "not_evaluable",
            "reason": "No summarized seed42 runs are available",
            "thresholds": {
                "external_roc_auc_delta_min": -0.01,
                "external_pr_auc_delta_min": -0.01,
                "internal_macro_f1_delta_min": -0.02,
                "params_ratio_max": 1.25,
                "flops_ratio_max": 1.25,
                "latency_ratio_max": 1.50,
            },
            "expand_additional_seeds": False,
        }
    base = summary[summary["model_id"] == "b6"]
    candidate = summary[summary["model_id"] == "b6m"]
    thresholds = {
        "external_roc_auc_delta_min": -0.01,
        "external_pr_auc_delta_min": -0.01,
        "internal_macro_f1_delta_min": -0.02,
        "params_ratio_max": 1.25,
        "flops_ratio_max": 1.25,
        "latency_ratio_max": 1.50,
    }
    if base.empty or candidate.empty:
        return {
            "status": "not_evaluable",
            "reason": "B6 seed42 or B6-M seed42 is missing",
            "thresholds": thresholds,
            "expand_additional_seeds": False,
        }
    b6 = base.iloc[0]
    b6m = candidate.iloc[0]
    values = {
        "external_roc_auc_delta": float(b6m["External ROC-AUC"] - b6["External ROC-AUC"]),
        "external_pr_auc_delta": float(b6m["External PR-AUC"] - b6["External PR-AUC"]),
        "internal_macro_f1_delta": float(b6m["Macro-F1"] - b6["Macro-F1"]),
        "params_ratio": (
            float(b6m["Params"] / b6["Params"])
            if finite(b6m.get("Params")) and finite(b6.get("Params")) and float(b6["Params"]) > 0
            else None
        ),
        "flops_ratio": (
            float(b6m["FLOPs"] / b6["FLOPs"])
            if finite(b6m.get("FLOPs")) and finite(b6.get("FLOPs")) and float(b6["FLOPs"]) > 0
            else None
        ),
        "latency_ratio": (
            float(b6m["Latency_ms_b1"] / b6["Latency_ms_b1"])
            if finite(b6m.get("Latency_ms_b1"))
            and finite(b6.get("Latency_ms_b1"))
            and float(b6["Latency_ms_b1"]) > 0
            else None
        ),
    }
    checks = {
        "external_roc_auc": values["external_roc_auc_delta"] >= thresholds["external_roc_auc_delta_min"],
        "external_pr_auc": values["external_pr_auc_delta"] >= thresholds["external_pr_auc_delta_min"],
        "internal_macro_f1": values["internal_macro_f1_delta"] >= thresholds["internal_macro_f1_delta_min"],
        "finite_metrics": all(
            finite(b6m[column]) for column in ["Macro-F1", "External ROC-AUC", "External PR-AUC"]
        ),
        "params": values["params_ratio"] is not None
        and values["params_ratio"] <= thresholds["params_ratio_max"],
        "flops": values["flops_ratio"] is not None
        and values["flops_ratio"] <= thresholds["flops_ratio_max"],
        "latency": values["latency_ratio"] is not None
        and values["latency_ratio"] <= thresholds["latency_ratio_max"],
    }
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "failed",
        "thresholds": thresholds,
        "observed": values,
        "checks": checks,
        "expand_additional_seeds": passed,
        "decision_rule_frozen_before_seed2024_and_seed3407": True,
    }


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    predictions_dir = Path(args.predictions_dir)
    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(results_dir / "bioengineering_direct_baselines")
    prediction_output = ensure_dir(output_dir / "predictions")
    table_output = ensure_dir(output_dir / "tables")
    optional_output = ensure_dir(output_dir / "optional")

    specs: list[DirectModelSpec] = list(DIRECT_MODEL_SPECS)
    if args.include_efficientnet:
        specs.append(next(item for item in OPTIONAL_MODEL_SPECS if item.model_id == "efficientnet_b0"))
    specs.append(next(item for item in OPTIONAL_MODEL_SPECS if item.model_id == "b6m"))

    efficiency = efficiency_lookup(results_dir)
    summary_rows: list[dict] = []
    external_rows: list[dict] = []
    fairness_rows: list[dict] = []
    softmax_scores: list[pd.DataFrame] = []
    bundles: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    run_dirs: dict[str, Path] = {}
    provenance: list[dict] = []
    missing: list[str] = []

    for spec in specs:
        run_dir = discover_run(runs_dir, spec, seed=42)
        if run_dir is None:
            if spec.required:
                missing.append(spec.model_id)
            continue
        try:
            validation, internal, external = load_prediction_triplet(predictions_dir, run_dir, spec)
        except Exception:
            if spec.required:
                raise
            continue
        run_dirs[spec.model_id] = run_dir
        bundles[spec.model_id] = (validation, internal, external)
        intended_thresholds = validation_thresholds(validation, spec.intended_score_column)
        softmax_thresholds = validation_thresholds(validation, "prob_mel")
        intended_internal = internal_metrics(internal, spec.intended_score_column)
        intended_external = external_metrics(external, spec.intended_score_column, intended_thresholds)
        softmax_internal = internal_metrics(internal, "prob_mel")
        softmax_external = external_metrics(external, "prob_mel", softmax_thresholds)
        eff = efficiency_values(efficiency.get(run_dir.name))

        summary_rows.append(
            {
                "model_id": spec.model_id,
                "Model": spec.display_name,
                "Type": spec.model_type,
                "run_seed": 42,
                "split_seed": 42,
                "run_id": run_dir.name,
                "Macro-F1": intended_internal["macro_f1"],
                "Macro-AUC": intended_internal["macro_auc"],
                "Internal Mel-AUC": intended_internal["mel_auc"],
                "External ROC-AUC": intended_external["roc_auc"],
                "External PR-AUC": intended_external["pr_auc"],
                "External Sensitivity": intended_external["sensitivity"],
                "External Specificity": intended_external["specificity"],
                "External Brier": intended_external["brier"],
                "External ECE": intended_external["ece"],
                "intended_score_source": spec.intended_score_column,
                "source": spec.source_label,
                "validation_youden_threshold": intended_thresholds["youden"],
                "threshold_source": "internal_validation",
                **eff,
            }
        )
        external_rows.append(
            {
                "model_id": spec.model_id,
                "run_id": run_dir.name,
                "score_source": spec.intended_score_column,
                "threshold_source": "internal_validation",
                "validation_youden_threshold": intended_thresholds["youden"],
                "validation_spec90_threshold": intended_thresholds["spec90"],
                "validation_spec95_threshold": intended_thresholds["spec95"],
                **{f"External {key}": value for key, value in intended_external.items()},
            }
        )
        fairness_rows.append(
            {
                "model_id": spec.model_id,
                "Model": spec.display_name,
                "run_id": run_dir.name,
                "score_source": "seven_class_softmax_mel",
                "threshold_source": "internal_validation",
                "validation_youden_threshold": softmax_thresholds["youden"],
                "Macro-F1": softmax_internal["macro_f1"],
                "Macro-AUC": softmax_internal["macro_auc"],
                "Internal Mel-AUC": softmax_internal["mel_auc"],
                "External ROC-AUC": softmax_external["roc_auc"],
                "External PR-AUC": softmax_external["pr_auc"],
                "External Sensitivity": softmax_external["sensitivity"],
                "External Specificity": softmax_external["specificity"],
                "External Brier": softmax_external["brier"],
                "External ECE": softmax_external["ece"],
                **eff,
            }
        )
        for dataset_name, frame, label_column in [
            ("internal_test", internal, "true_label"),
            ("external_locked", external, "label"),
        ]:
            score_frame = frame[["image_id", label_column, "prob_mel"]].copy()
            score_frame = score_frame.rename(columns={label_column: "label", "prob_mel": "score"})
            score_frame.insert(0, "run_id", run_dir.name)
            score_frame.insert(0, "model_id", spec.model_id)
            score_frame.insert(0, "dataset", dataset_name)
            score_frame["score_source"] = "seven_class_softmax_mel"
            softmax_scores.append(score_frame)

        provenance_row = run_provenance(run_dir)
        provenance_row.update(
            {
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "intended_score_source": spec.intended_score_column,
                "internal_prediction_sha256": file_sha256(prediction_path(predictions_dir, run_dir, "test")),
                "validation_prediction_sha256": file_sha256(prediction_path(predictions_dir, run_dir, "val")),
                "external_prediction_sha256": file_sha256(prediction_path(predictions_dir, run_dir, "external")),
            }
        )
        provenance.append(provenance_row)

        if spec.model_id in {"swin_tiny", "dermamamba"}:
            shutil.copy2(
                prediction_path(predictions_dir, run_dir, "external"),
                prediction_output / f"{spec.model_id}_seed42_external.csv",
            )

    if missing and not args.allow_incomplete:
        raise RuntimeError(f"Required direct-comparison runs are missing: {missing}")

    summary = pd.DataFrame(summary_rows)
    external_table = pd.DataFrame(external_rows)
    fairness = pd.DataFrame(fairness_rows)
    score_table = pd.concat(softmax_scores, ignore_index=True) if softmax_scores else pd.DataFrame()

    summary.to_csv(output_dir / "baseline_summary_seed42.csv", index=False)
    external_table.to_csv(output_dir / "external_melanoma_seed42.csv", index=False)
    score_table.to_csv(output_dir / "sevenclass_derived_melanoma_scores.csv", index=False)

    primary_ids = ["b1", "swin_tiny", "dermamamba", "b2", "b4", "b6"]
    table3_columns = [
        "Model",
        "Type",
        "Params",
        "FLOPs",
        "Macro-F1",
        "Macro-AUC",
        "Internal Mel-AUC",
        "External ROC-AUC",
        "External PR-AUC",
        "run_id",
        "intended_score_source",
    ]
    table3 = (
        summary[summary["model_id"].isin(primary_ids)].copy()
        if "model_id" in summary.columns
        else pd.DataFrame(columns=table3_columns)
    )
    if not table3.empty:
        table3["_order"] = table3["model_id"].map({key: idx for idx, key in enumerate(primary_ids)})
        table3 = table3.sort_values("_order").drop(columns="_order")
    table3[[c for c in table3_columns if c in table3.columns]].to_csv(
        table_output / "Table3_direct_recent_baselines.csv", index=False
    )
    fairness_ids = list(primary_ids)
    if args.include_efficientnet:
        fairness_ids.append("efficientnet_b0")
    direct_fairness = (
        fairness[fairness["model_id"].isin(fairness_ids)].copy()
        if "model_id" in fairness.columns
        else fairness
    )
    direct_fairness.to_csv(table_output / "TableS5_architecture_fairness.csv", index=False)
    if "model_id" in fairness.columns and (fairness["model_id"] == "b6m").any():
        fairness[fairness["model_id"] == "b6m"].to_csv(
            optional_output / "B6M_architecture_fairness.csv", index=False
        )

    bootstrap_rows = []
    if not args.skip_paired_bootstrap and "b6" in bundles and "dermamamba" in bundles:
        b6_external = bundles["b6"][2]
        derma_external = bundles["dermamamba"][2]
        aligned = aligned_external(b6_external, derma_external, "prob_mel_endpoint", "prob_mel")
        for metric in ["roc_auc", "pr_auc"]:
            bootstrap_rows.append(
                paired_bootstrap_delta(
                    aligned,
                    metric,
                    iterations=args.bootstrap_iters,
                    seed=args.bootstrap_seed,
                )
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output_dir / "dermamamba_vs_b6_paired_bootstrap.csv", index=False)

    gate = make_gate(summary)
    write_json(gate, optional_output / "B6M_seed42_gate.json")
    if not summary.empty and (summary["model_id"] == "b6m").any():
        summary[summary["model_id"] == "b6m"].to_csv(optional_output / "B6M_seed42_metrics.csv", index=False)
        external_table[external_table["model_id"] == "b6m"].to_csv(
            optional_output / "B6M_seed42_external.csv", index=False
        )

    b6m_runs = []
    b6m_spec = next(item for item in OPTIONAL_MODEL_SPECS if item.model_id == "b6m")
    for seed in [42, 2024, 3407]:
        run = discover_run(runs_dir, b6m_spec, seed=seed)
        if run is None:
            continue
        try:
            validation, internal, external = load_prediction_triplet(predictions_dir, run, b6m_spec)
        except Exception:
            continue
        im = internal_metrics(internal, b6m_spec.intended_score_column)
        thresholds = validation_thresholds(validation, b6m_spec.intended_score_column)
        em = external_metrics(external, b6m_spec.intended_score_column, thresholds)
        b6m_runs.append(
            {
                "seed": seed,
                "run_id": run.name,
                "Macro-F1": im["macro_f1"],
                "Internal Mel-AUC": im["mel_auc"],
                "External ROC-AUC": em["roc_auc"],
                "External PR-AUC": em["pr_auc"],
            }
        )
    b6m_run_table = pd.DataFrame(b6m_runs)
    b6m_run_table.to_csv(optional_output / "B6M_stability_runs.csv", index=False)
    if not b6m_run_table.empty:
        metric_columns = ["Macro-F1", "Internal Mel-AUC", "External ROC-AUC", "External PR-AUC"]
        stability = {"n_seeds": int(len(b6m_run_table)), "seeds": ",".join(map(str, b6m_run_table["seed"]))}
        for column in metric_columns:
            stability[f"{column}_mean"] = float(b6m_run_table[column].mean())
            stability[f"{column}_std"] = float(b6m_run_table[column].std(ddof=1)) if len(b6m_run_table) > 1 else None
        pd.DataFrame([stability]).to_csv(optional_output / "B6M_stability_mean_std.csv", index=False)

    crosssplit_rows = []
    if "model_id" in summary.columns and (summary["model_id"] == "b6").any():
        base = summary[summary["model_id"] == "b6"].iloc[0]
        crosssplit_rows.append(
            {
                "cohort": "fixed_split_seed42",
                "run_seed": int(base["run_seed"]),
                "split_seed": int(base["split_seed"]),
                "run_id": base["run_id"],
                "Macro-F1": base["Macro-F1"],
                "Internal Mel-AUC": base["Internal Mel-AUC"],
                "External ROC-AUC": base["External ROC-AUC"],
                "External PR-AUC": base["External PR-AUC"],
            }
        )
    cross_spec = next(item for item in OPTIONAL_MODEL_SPECS if item.model_id == "b6_crosssplit2024")
    cross_run = discover_run(runs_dir, cross_spec, seed=42, expected_split_seed=2024)
    if cross_run is not None:
        cross_validation, cross_internal, cross_external = load_prediction_triplet(
            predictions_dir, cross_run, cross_spec
        )
        cross_im = internal_metrics(cross_internal, cross_spec.intended_score_column)
        cross_thresholds = validation_thresholds(cross_validation, cross_spec.intended_score_column)
        cross_em = external_metrics(cross_external, cross_spec.intended_score_column, cross_thresholds)
        crosssplit_rows.append(
            {
                "cohort": "independent_split_seed2024",
                "run_seed": 42,
                "split_seed": 2024,
                "run_id": cross_run.name,
                "Macro-F1": cross_im["macro_f1"],
                "Internal Mel-AUC": cross_im["mel_auc"],
                "External ROC-AUC": cross_em["roc_auc"],
                "External PR-AUC": cross_em["pr_auc"],
            }
        )
        cross_provenance = run_provenance(cross_run)
        cross_provenance.update(
            {
                "model_id": cross_spec.model_id,
                "display_name": cross_spec.display_name,
                "intended_score_source": cross_spec.intended_score_column,
                "internal_prediction_sha256": file_sha256(
                    prediction_path(predictions_dir, cross_run, "test")
                ),
                "validation_prediction_sha256": file_sha256(
                    prediction_path(predictions_dir, cross_run, "val")
                ),
                "external_prediction_sha256": file_sha256(
                    prediction_path(predictions_dir, cross_run, "external")
                ),
            }
        )
        provenance.append(cross_provenance)
    crosssplit_table = pd.DataFrame(crosssplit_rows)
    crosssplit_table.to_csv(optional_output / "B6_crosssplit2024_metrics.csv", index=False)

    source_lock_path = Path("configs/bioengineering/source_lock.json")
    manifest_path = output_dir / "direct_baseline_manifest.json"
    status_path = output_dir / "status.json"
    manifest = {
        "status": "completed" if not missing else "completed_with_missing_required_runs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_performed_by_summary_script": False,
        "fixed_split_seed": 42,
        "fixed_run_seed_for_direct_table": 42,
        "include_efficientnet": bool(args.include_efficientnet),
        "external_tuning_performed": False,
        "operating_threshold_policy": "derive once on each run's internal validation set; apply unchanged to external data",
        "required_models": primary_ids,
        "missing_required_models": missing,
        "score_policy": {
            "Table3": "model-intended melanoma score; seven-class metrics always use the softmax head",
            "TableS5": "seven-class softmax MEL score for every direct-comparison architecture",
        },
        "paired_bootstrap": {
            "comparison": "B6 minus DermaMamba resource-matched architecture port",
            "iterations": args.bootstrap_iters,
            "seed": args.bootstrap_seed,
            "resampling_unit": "external_image_id",
            "p_value_method": "two-sided empirical paired bootstrap with plus-one correction",
        },
        "cross_split_repeat": {
            "requested_split_seed": 2024,
            "run_seed": 42,
            "available": cross_run is not None,
            "interpretation": "descriptive sensitivity analysis; internal test cohorts differ and are not paired",
        },
        "source_lock": {
            "path": str(source_lock_path),
            "sha256": file_sha256(source_lock_path) if source_lock_path.exists() else None,
        },
        "runs": provenance,
    }
    status_payload = {
        "status": manifest["status"],
        "models_summarized": [row["model_id"] for row in summary_rows],
        "missing_required_models": missing,
        "b6m_gate": gate.get("status"),
        "b6m_expand_additional_seeds": gate.get("expand_additional_seeds", False),
        "paired_bootstrap_rows": len(bootstrap),
        "crosssplit2024_available": cross_run is not None,
    }
    write_json(status_payload, status_path)
    outputs = [str(path) for path in sorted(output_dir.rglob("*")) if path.is_file()]
    outputs.append(str(manifest_path))
    manifest["outputs"] = sorted(set(outputs))
    write_json(manifest, manifest_path)
    print(
        f"bioengineering_summary status={manifest['status']} models={len(summary)} "
        f"bootstrap_rows={len(bootstrap)} b6m_gate={gate.get('status')}"
    )


if __name__ == "__main__":
    main()
