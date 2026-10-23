from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from lesionmamba.analysis.bioengineering_supplement import (
    DIRECT_MODEL_SPECS,
    OPTIONAL_MODEL_SPECS,
    audit_ham_split_files,
    dermamamba_protocol_ok,
    file_sha256,
    load_prediction_triplet,
    validation_thresholds,
)
from lesionmamba.data.datasets import LABEL_MAP
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, write_json


REQUIRED_OUTPUTS = (
    "baseline_summary_seed42.csv",
    "external_melanoma_seed42.csv",
    "sevenclass_derived_melanoma_scores.csv",
    "tables/Table3_direct_recent_baselines.csv",
    "tables/TableS5_architecture_fairness.csv",
    "dermamamba_vs_b6_paired_bootstrap.csv",
    "direct_baseline_manifest.json",
    "status.json",
)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(self, name: str, passed: bool, detail: Any = None) -> None:
        self.checks.append({"name": name, "status": "passed" if passed else "failed", "detail": detail})
        if not passed:
            self.errors.append(f"{name}: {detail}")

    def warn(self, name: str, detail: Any) -> None:
        self.checks.append({"name": name, "status": "warning", "detail": detail})
        self.warnings.append(f"{name}: {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--metadata-csv", default="data/raw/HAM10000/HAM10000_metadata.csv")
    parser.add_argument("--external-labels-csv", default="data/raw/ISIC_external/labels_harmonized.csv")
    parser.add_argument("--minimum-bootstrap-iters", type=int, default=10_000)
    parser.add_argument("--include-efficientnet", action="store_true")
    parser.add_argument("--require-b6m", action="store_true")
    parser.add_argument("--require-crosssplit", action="store_true")
    parser.add_argument("--require-mask-status", action="store_true")
    return parser.parse_args()


def read_json(path: Path, audit: Audit, name: str) -> dict[str, Any]:
    if not path.is_file():
        audit.check(name, False, f"missing file: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        audit.check(name, False, f"invalid JSON: {exc}")
        return {}
    audit.check(name, isinstance(payload, dict), str(path))
    return payload if isinstance(payload, dict) else {}


def read_csv(path: Path, audit: Audit, name: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        audit.check(name, False, f"missing or empty file: {path}")
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        audit.check(name, False, f"invalid CSV: {exc}")
        return pd.DataFrame()
    audit.check(name, not frame.empty, {"path": str(path), "rows": len(frame)})
    return frame


def close(a: Any, b: Any, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def column(frame: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index, name=name)


def architecture_metadata_ok(model_id: str, metadata: dict[str, Any], locked_revision: str | None) -> tuple[bool, dict[str, Any]]:
    expected: dict[str, Any]
    if model_id in {"b1", "b2", "b4", "b6", "b6_crosssplit2024"}:
        expected = {"source": "oncoderm_mambax", "timm_name": "convnext_tiny", "pretrained": True}
    elif model_id == "swin_tiny":
        expected = {"source": "timm", "timm_name": "swin_tiny_patch4_window7_224", "pretrained": True}
    elif model_id == "efficientnet_b0":
        expected = {"source": "timm", "timm_name": "efficientnet_b0", "pretrained": True}
    elif model_id == "dermamamba":
        expected = {
            "source": "clean_room_architecture_faithful_port",
            "comparison_label": "DermaMamba (resource-matched architecture port)",
            "source_revision": locked_revision,
            "official_checkpoint_used": False,
            "pretrained": True,
            "embed_dim": 192,
            "depth": 6,
        }
    elif model_id == "b6m":
        expected = {
            "source": "oncoderm_mambax",
            "timm_name": "convnext_tiny",
            "pretrained": True,
            "formal_mamba_block": True,
            "effective_ssm_backend": "state_spaces_mamba_ssm",
            "mamba_scope": "F4_only",
        }
    else:
        return False, {"reason": "unknown model_id", "model_id": model_id}
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    return not mismatches, {"expected": expected, "mismatches": mismatches}


def prediction_matches_source(
    prediction: pd.DataFrame,
    source_csv: Path,
    prediction_label: str,
    source_label: str,
    label_map: dict[str, int] | None = None,
) -> tuple[bool, dict[str, Any]]:
    if not source_csv.is_file():
        return False, {"reason": "source CSV missing", "path": str(source_csv)}
    source = pd.read_csv(source_csv)
    required_prediction = {"image_id", prediction_label}
    required_source = {"image_id", source_label}
    if not required_prediction.issubset(prediction.columns) or not required_source.issubset(source.columns):
        return False, {
            "reason": "required columns missing",
            "prediction_columns": list(prediction.columns),
            "source_columns": list(source.columns),
        }
    if source["image_id"].astype(str).duplicated().any():
        return False, {"reason": "source image_id is not unique", "path": str(source_csv)}
    predicted = prediction[["image_id", prediction_label]].copy()
    expected = source[["image_id", source_label]].copy()
    predicted["image_id"] = predicted["image_id"].astype(str)
    expected["image_id"] = expected["image_id"].astype(str)
    if label_map is not None:
        expected[source_label] = expected[source_label].map(label_map)
    expected[source_label] = pd.to_numeric(expected[source_label], errors="coerce")
    predicted[prediction_label] = pd.to_numeric(predicted[prediction_label], errors="coerce")
    merged = predicted.merge(expected, on="image_id", how="outer", indicator=True, validate="one_to_one")
    matched = merged["_merge"].eq("both")
    labels_equal = bool(
        matched.all()
        and merged[prediction_label].notna().all()
        and merged[source_label].notna().all()
        and np.array_equal(
            merged[prediction_label].astype(int).to_numpy(),
            merged[source_label].astype(int).to_numpy(),
        )
    )
    return labels_equal, {
        "prediction_rows": len(predicted),
        "source_rows": len(expected),
        "matched_rows": int(matched.sum()),
        "prediction_only": int((merged["_merge"] == "left_only").sum()),
        "source_only": int((merged["_merge"] == "right_only").sum()),
        "labels_equal": labels_equal,
        "source_csv": str(source_csv),
    }


def finite_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> tuple[bool, dict[str, Any]]:
    missing = [name for name in columns if name not in frame]
    if missing:
        return False, {"missing_columns": missing}
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).all()), {"columns": columns, "rows": len(frame)}


def audit_mask_status(output_dir: Path, audit: Audit) -> None:
    status_path = output_dir / "mask200_status.json"
    status = read_json(status_path, audit, "mask200_status_json")
    if not status:
        return
    state = status.get("status")
    table_path = output_dir / "mask200_table.csv"
    if state == "not_run_insufficient_pairs":
        available = int(status.get("available_pairs", -1))
        required = int(status.get("required_pairs", -1))
        audit.check(
            "mask200_infeasibility_is_real_and_auditable",
            required == 200 and 0 <= available < required and not table_path.exists(),
            {"available_pairs": available, "required_pairs": required, "table_exists": table_path.exists()},
        )
        return
    if state == "completed_or_partial":
        available = int(status.get("available_image_mask_pairs", -1))
        evaluated = int(status.get("evaluated_mask_count", -1))
        run_states = status.get("runs", [])
        completed = [row for row in run_states if row.get("status") == "completed"]
        enough_per_run = bool(completed) and all(int(row.get("matched_masks", -1)) >= 200 for row in completed)
        audit.check(
            "mask200_completed_with_real_pairs",
            available >= 200 and evaluated == 200 and enough_per_run and table_path.is_file(),
            {
                "available_pairs": available,
                "evaluated_masks": evaluated,
                "completed_runs": len(completed),
                "table_exists": table_path.exists(),
            },
        )
        return
    audit.check("mask200_status_is_recognized", False, state)


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    predictions_dir = Path(args.predictions_dir)
    output_dir = ensure_dir(Path(args.results_dir) / "bioengineering_direct_baselines")
    audit_path = output_dir / "completion_audit.json"
    audit = Audit()

    for relative in REQUIRED_OUTPUTS:
        path = output_dir / relative
        audit.check(f"required_output:{relative}", path.is_file() and path.stat().st_size > 0, str(path))

    manifest = read_json(output_dir / "direct_baseline_manifest.json", audit, "manifest_json")
    status = read_json(output_dir / "status.json", audit, "summary_status_json")
    audit.check("summary_completed", status.get("status") == "completed", status.get("status"))
    audit.check("manifest_completed", manifest.get("status") == "completed", manifest.get("status"))
    audit.check("external_tuning_disabled", manifest.get("external_tuning_performed") is False, None)
    audit.check(
        "threshold_policy_is_validation_locked",
        "internal validation" in str(manifest.get("operating_threshold_policy", "")).lower(),
        manifest.get("operating_threshold_policy"),
    )
    split42 = audit_ham_split_files(args.split_dir, seed=42, metadata_csv=args.metadata_csv)
    audit.check("fixed_split42_integrity", split42.get("status") == "passed", split42)

    source_lock_path = Path("configs/bioengineering/source_lock.json")
    source_lock = read_json(source_lock_path, audit, "source_lock_json")
    locked_revision = source_lock.get("dermamamba", {}).get("revision") if source_lock else None
    derma_config_path = Path("configs/bioengineering/dermamamba_architecture_faithful_seed42.yaml")
    derma_config = load_yaml(derma_config_path) if derma_config_path.exists() else {}
    configured_revision = derma_config.get("model", {}).get("dermamamba", {}).get("source_revision")
    audit.check(
        "dermamamba_revision_matches_source_lock",
        bool(locked_revision) and configured_revision == locked_revision,
        {"locked": locked_revision, "configured": configured_revision},
    )
    source_protocol_ok, source_protocol_detail = dermamamba_protocol_ok(
        derma_config, locked_revision
    )
    audit.check(
        "dermamamba_source_config_matches_frozen_adapter",
        source_protocol_ok,
        source_protocol_detail,
    )
    manifest_lock = manifest.get("source_lock", {})
    audit.check(
        "manifest_source_lock_hash_matches",
        source_lock_path.is_file() and manifest_lock.get("sha256") == file_sha256(source_lock_path),
        manifest_lock,
    )

    listed_outputs = {str(item) for item in manifest.get("outputs", [])}
    missing_listed = [item for item in listed_outputs if not Path(item).is_file()]
    audit.check("manifest_output_files_exist", not missing_listed, missing_listed[:10])
    required_listed = [str(output_dir / item) for item in REQUIRED_OUTPUTS]
    audit.check(
        "manifest_lists_required_outputs",
        all(item in listed_outputs for item in required_listed),
        [item for item in required_listed if item not in listed_outputs],
    )

    summary = read_csv(output_dir / "baseline_summary_seed42.csv", audit, "baseline_summary_csv")
    external_table = read_csv(output_dir / "external_melanoma_seed42.csv", audit, "external_summary_csv")
    score_table = read_csv(output_dir / "sevenclass_derived_melanoma_scores.csv", audit, "softmax_score_csv")
    table3 = read_csv(output_dir / "tables/Table3_direct_recent_baselines.csv", audit, "table3_csv")
    table_s5 = read_csv(output_dir / "tables/TableS5_architecture_fairness.csv", audit, "table_s5_csv")
    bootstrap = read_csv(
        output_dir / "dermamamba_vs_b6_paired_bootstrap.csv", audit, "paired_bootstrap_csv"
    )

    required_ids = [spec.model_id for spec in DIRECT_MODEL_SPECS]
    expected_fairness_ids = [*required_ids, *(["efficientnet_b0"] if args.include_efficientnet else [])]
    if not summary.empty and "model_id" in summary:
        summary_ids = summary["model_id"].astype(str).tolist()
        audit.check("required_direct_models_present_once", all(summary_ids.count(item) == 1 for item in required_ids), summary_ids)
        audit.check("summary_model_ids_unique", not summary["model_id"].duplicated().any(), summary_ids)
        direct = summary[summary["model_id"].isin(required_ids)]
        audit.check(
            "direct_models_use_fixed_run_and_split_seed42",
            len(direct) == len(required_ids)
            and (direct["run_seed"].astype(int) == 42).all()
            and (direct["split_seed"].astype(int) == 42).all(),
            direct[["model_id", "run_seed", "split_seed"]].to_dict("records") if not direct.empty else [],
        )
    else:
        audit.check("summary_has_model_id", False, list(summary.columns))

    if not table3.empty:
        table3_runs = set(table3.get("run_id", pd.Series(dtype=str)).astype(str))
        expected_runs = set(
            summary.loc[column(summary, "model_id").isin(required_ids), "run_id"].astype(str)
        ) if "model_id" in summary and "run_id" in summary else set()
        audit.check("table3_contains_exact_direct_runs", len(table3) == 6 and table3_runs == expected_runs, sorted(table3_runs))
        expected_order = (
            summary.set_index("model_id").loc[required_ids, "run_id"].astype(str).tolist()
            if set(required_ids).issubset(set(column(summary, "model_id").astype(str)))
            else []
        )
        audit.check(
            "table3_direct_model_order_is_frozen",
            table3["run_id"].astype(str).tolist() == expected_order,
            table3["run_id"].astype(str).tolist(),
        )
        numeric_ok, numeric_detail = finite_numeric_columns(
            table3,
            [
                "Params",
                "FLOPs",
                "Macro-F1",
                "Macro-AUC",
                "Internal Mel-AUC",
                "External ROC-AUC",
                "External PR-AUC",
            ],
        )
        positive_efficiency = (
            (pd.to_numeric(column(table3, "Params"), errors="coerce") > 0).all()
            and (pd.to_numeric(column(table3, "FLOPs"), errors="coerce") > 0).all()
        )
        audit.check(
            "table3_numeric_metrics_and_efficiency_are_complete",
            numeric_ok and positive_efficiency,
            numeric_detail,
        )

    if not table_s5.empty and "model_id" in table_s5:
        actual_fairness_ids = table_s5["model_id"].astype(str).tolist()
        audit.check(
            "table_s5_contains_exact_fair_architectures",
            sorted(actual_fairness_ids) == sorted(expected_fairness_ids),
            actual_fairness_ids,
        )
        audit.check(
            "table_s5_uses_softmax_mel_for_every_model",
            set(column(table_s5, "score_source").astype(str)) == {"seven_class_softmax_mel"},
            sorted(set(column(table_s5, "score_source").astype(str))),
        )
        fairness_numeric_ok, fairness_numeric_detail = finite_numeric_columns(
            table_s5,
            [
                "Macro-F1",
                "Macro-AUC",
                "Internal Mel-AUC",
                "External ROC-AUC",
                "External PR-AUC",
                "External Sensitivity",
                "External Specificity",
                "External Brier",
                "External ECE",
            ],
        )
        audit.check("table_s5_metrics_are_complete", fairness_numeric_ok, fairness_numeric_detail)

    if not external_table.empty:
        audit.check(
            "external_operating_points_are_validation_derived",
            set(column(external_table, "threshold_source").astype(str)) == {"internal_validation"},
            sorted(set(column(external_table, "threshold_source").astype(str))),
        )

    if not score_table.empty:
        required_score_columns = {"dataset", "model_id", "image_id", "score"}
        score_columns_ok = required_score_columns.issubset(score_table.columns)
        duplicate_keys = (
            score_table.duplicated(["dataset", "model_id", "image_id"]).any()
            if score_columns_ok
            else True
        )
        scores = pd.to_numeric(column(score_table, "score"), errors="coerce").to_numpy()
        audit.check("softmax_scores_have_unique_image_keys", not duplicate_keys, None)
        audit.check(
            "softmax_scores_are_finite_probabilities",
            np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
            {"rows": len(scores)},
        )
        present = set(column(score_table, "model_id").astype(str))
        audit.check("softmax_scores_cover_direct_models", set(required_ids).issubset(present), sorted(present))

    all_specs = {spec.model_id: spec for spec in (*DIRECT_MODEL_SPECS, *OPTIONAL_MODEL_SPECS)}
    provenance_by_model = {row.get("model_id"): row for row in manifest.get("runs", [])}
    if not summary.empty and "model_id" in summary:
        for _, row in summary.iterrows():
            model_id = str(row["model_id"])
            spec = all_specs.get(model_id)
            if spec is None:
                audit.check(f"known_model_spec:{model_id}", False, model_id)
                continue
            run_dir = runs_dir / str(row["run_id"])
            try:
                validation, internal, external = load_prediction_triplet(predictions_dir, run_dir, spec)
            except Exception as exc:
                audit.check(f"prediction_triplet_valid:{model_id}", False, str(exc))
                continue
            audit.check(f"prediction_triplet_valid:{model_id}", True, run_dir.name)
            split_seed = int(row.get("split_seed", 42))
            for split_name, frame in [("val", validation), ("test", internal)]:
                matches, detail = prediction_matches_source(
                    frame,
                    Path(args.split_dir) / f"ham10000_seed{split_seed}_{split_name}.csv",
                    "true_label",
                    "dx",
                    LABEL_MAP,
                )
                audit.check(f"prediction_matches_split:{model_id}:{split_name}", matches, detail)
            external_matches, external_detail = prediction_matches_source(
                external,
                Path(args.external_labels_csv),
                "label",
                "melanoma",
            )
            audit.check(
                f"external_prediction_matches_locked_labels:{model_id}",
                external_matches,
                external_detail,
            )
            intended_threshold = validation_thresholds(validation, spec.intended_score_column)["youden"]
            audit.check(
                f"validation_threshold_reproduces:{model_id}",
                close(row.get("validation_youden_threshold"), intended_threshold),
                {"reported": row.get("validation_youden_threshold"), "recomputed": intended_threshold},
            )
            provenance = provenance_by_model.get(model_id, {})
            metadata_ok, metadata_detail = architecture_metadata_ok(
                model_id, provenance.get("model_metadata", {}), locked_revision
            )
            audit.check(f"architecture_identity_matches_protocol:{model_id}", metadata_ok, metadata_detail)
            config_path = run_dir / "config.yaml"
            audit.check(
                f"config_hash_matches_manifest:{model_id}",
                config_path.is_file() and provenance.get("config_sha256") == file_sha256(config_path),
                provenance.get("config_sha256"),
            )
            if model_id == "dermamamba":
                run_config = load_yaml(config_path) if config_path.is_file() else {}
                protocol_ok, protocol_detail = dermamamba_protocol_ok(
                    run_config, locked_revision
                )
                audit.check(
                    "dermamamba_run_matches_frozen_training_protocol",
                    protocol_ok,
                    protocol_detail,
                )
            for split, key in [
                ("val", "validation_prediction_sha256"),
                ("test", "internal_prediction_sha256"),
                ("external", "external_prediction_sha256"),
            ]:
                prediction = predictions_dir / f"{run_dir.name}_{split}_predictions.csv"
                audit.check(
                    f"prediction_hash_matches_manifest:{model_id}:{split}",
                    prediction.is_file() and provenance.get(key) == file_sha256(prediction),
                    provenance.get(key),
                )

    if not bootstrap.empty:
        metrics = set(column(bootstrap, "metric").astype(str))
        requested = pd.to_numeric(column(bootstrap, "iterations_requested"), errors="coerce")
        valid = pd.to_numeric(column(bootstrap, "iterations_valid"), errors="coerce")
        p_values = pd.to_numeric(column(bootstrap, "p_value_two_sided"), errors="coerce")
        audit.check("paired_bootstrap_has_roc_and_pr", metrics == {"roc_auc", "pr_auc"}, sorted(metrics))
        audit.check(
            "paired_bootstrap_iteration_budget_met",
            len(bootstrap) == 2
            and requested.notna().all()
            and (requested >= args.minimum_bootstrap_iters).all()
            and valid.notna().all()
            and (valid > 0).all()
            and (valid <= requested).all(),
            bootstrap[["metric", "iterations_requested", "iterations_valid"]].to_dict("records"),
        )
        audit.check(
            "paired_bootstrap_uses_finite_sample_corrected_p_values",
            p_values.notna().all()
            and ((p_values > 0) & (p_values <= 1)).all()
            and set(column(bootstrap, "p_value_method").astype(str))
            == {"two-sided empirical paired bootstrap with plus-one correction"},
            p_values.tolist(),
        )
        audit.check(
            "paired_bootstrap_is_image_aligned_b6_vs_dermamamba",
            set(column(bootstrap, "system_a").astype(str)) == {"b6"}
            and set(column(bootstrap, "system_b").astype(str)) == {"dermamamba"}
            and set(column(bootstrap, "resampling_unit").astype(str)) == {"external_image_id"}
            and column(bootstrap, "n").nunique() == 1,
            bootstrap.reindex(columns=["metric", "system_a", "system_b", "resampling_unit", "n"]).to_dict("records"),
        )

    gate_path = output_dir / "optional/B6M_seed42_gate.json"
    gate = read_json(gate_path, audit, "b6m_gate_json") if args.require_b6m else {}
    if args.require_b6m:
        b6m_present = not summary.empty and "b6m" in set(summary["model_id"].astype(str))
        audit.check("b6m_seed42_present", b6m_present, None)
        audit.check("b6m_gate_is_evaluable", gate.get("status") in {"passed", "failed"}, gate.get("status"))
        smoke = read_json(output_dir / "mamba_backend_smoke.json", audit, "mamba_backend_smoke_json")
        expected_mamba = str(source_lock.get("mamba_ssm", {}).get("release", "")).lstrip("v")
        audit.check(
            "mamba_backend_smoke_passed_at_locked_version",
            smoke.get("status") == "passed"
            and smoke.get("mamba_ssm") == expected_mamba
            and smoke.get("forward_and_backward_finite") is True,
            smoke,
        )
        expected_gate_checks = {
            "external_roc_auc",
            "external_pr_auc",
            "internal_macro_f1",
            "finite_metrics",
            "params",
            "flops",
            "latency",
        }
        audit.check(
            "b6m_gate_covers_performance_and_resources",
            expected_gate_checks == set(gate.get("checks", {})),
            sorted(gate.get("checks", {})),
        )
        stability = read_csv(output_dir / "optional/B6M_stability_runs.csv", audit, "b6m_stability_csv")
        seeds = set(pd.to_numeric(column(stability, "seed"), errors="coerce").dropna().astype(int))
        audit.check("b6m_stability_contains_seed42", 42 in seeds, sorted(seeds))
        if gate.get("expand_additional_seeds"):
            audit.check("b6m_gate_expansion_completed", {42, 2024, 3407}.issubset(seeds), sorted(seeds))
        elif seeds - {42}:
            audit.warn("b6m_additional_seeds_despite_failed_gate", sorted(seeds))

    if args.require_crosssplit:
        cross = read_csv(
            output_dir / "optional/B6_crosssplit2024_metrics.csv", audit, "b6_crosssplit_csv"
        )
        split_seeds = set(pd.to_numeric(column(cross, "split_seed"), errors="coerce").dropna().astype(int))
        run_seeds = set(pd.to_numeric(column(cross, "run_seed"), errors="coerce").dropna().astype(int))
        audit.check(
            "b6_crosssplit_repeat_is_distinct_and_correct",
            split_seeds == {42, 2024} and run_seeds == {42} and len(cross) == 2,
            {"split_seeds": sorted(split_seeds), "run_seeds": sorted(run_seeds), "rows": len(cross)},
        )
        independent = cross[pd.to_numeric(column(cross, "split_seed"), errors="coerce") == 2024]
        if len(independent) == 1:
            cross_run_id = str(independent.iloc[0]["run_id"])
            cross_run_dir = runs_dir / cross_run_id
            cross_spec = all_specs["b6_crosssplit2024"]
            try:
                cross_validation, cross_internal, cross_external = load_prediction_triplet(
                    predictions_dir, cross_run_dir, cross_spec
                )
                audit.check("crosssplit_prediction_triplet_valid", True, cross_run_id)
                for split_name, frame in [("val", cross_validation), ("test", cross_internal)]:
                    matches, detail = prediction_matches_source(
                        frame,
                        Path(args.split_dir) / f"ham10000_seed2024_{split_name}.csv",
                        "true_label",
                        "dx",
                        LABEL_MAP,
                    )
                    audit.check(f"crosssplit_prediction_matches_split:{split_name}", matches, detail)
                external_matches, external_detail = prediction_matches_source(
                    cross_external,
                    Path(args.external_labels_csv),
                    "label",
                    "melanoma",
                )
                audit.check(
                    "crosssplit_external_prediction_matches_locked_labels",
                    external_matches,
                    external_detail,
                )
            except Exception as exc:
                audit.check("crosssplit_prediction_triplet_valid", False, str(exc))
            cross_provenance = provenance_by_model.get("b6_crosssplit2024", {})
            metadata_ok, metadata_detail = architecture_metadata_ok(
                "b6_crosssplit2024", cross_provenance.get("model_metadata", {}), locked_revision
            )
            audit.check("crosssplit_architecture_identity_matches_protocol", metadata_ok, metadata_detail)
            cross_config_path = cross_run_dir / "config.yaml"
            audit.check(
                "crosssplit_config_hash_matches_manifest",
                cross_config_path.is_file()
                and cross_provenance.get("config_sha256") == file_sha256(cross_config_path),
                cross_provenance.get("config_sha256"),
            )
        else:
            audit.check("crosssplit_has_one_independent_run", False, {"rows": len(independent)})
        split2024 = audit_ham_split_files(args.split_dir, seed=2024, metadata_csv=args.metadata_csv)
        audit.check("crosssplit2024_integrity", split2024.get("status") == "passed", split2024)
        audit.check(
            "crosssplit2024_assignment_differs_from_split42",
            split2024.get("sha256") != split42.get("sha256"),
            {"split42": split42.get("sha256"), "split2024": split2024.get("sha256")},
        )

    if args.require_mask_status:
        audit_mask_status(output_dir, audit)

    payload = {
        "status": "passed" if not audit.errors else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "minimum_bootstrap_iterations": args.minimum_bootstrap_iters,
        "requirements": {
            "include_efficientnet": args.include_efficientnet,
            "require_b6m": args.require_b6m,
            "require_crosssplit": args.require_crosssplit,
            "require_mask_status": args.require_mask_status,
        },
        "checks": audit.checks,
        "errors": audit.errors,
        "warnings": audit.warnings,
    }
    write_json(payload, audit_path)
    print(
        f"bioengineering_completion_audit status={payload['status']} "
        f"checks={len(audit.checks)} errors={len(audit.errors)} warnings={len(audit.warnings)}"
    )
    if audit.errors:
        for error in audit.errors:
            print(f"[AUDIT ERROR] {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
