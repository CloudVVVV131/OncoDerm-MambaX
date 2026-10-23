from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from lesionmamba.data.datasets import LABEL_MAP
from lesionmamba.data.leakage import leakage_report
from lesionmamba.metrics.classification import multiclass_metrics
from lesionmamba.metrics.melanoma_endpoint import (
    endpoint_metrics_from_scores,
    threshold_for_specificity,
    threshold_youden,
)
from lesionmamba.utils.config import load_yaml


PROBABILITY_COLUMNS = ["prob_akiec", "prob_bcc", "prob_bkl", "prob_df", "prob_mel", "prob_nv", "prob_vasc"]


@dataclass(frozen=True)
class DirectModelSpec:
    model_id: str
    display_name: str
    model_type: str
    run_prefix: str
    intended_score_column: str
    source_label: str
    required: bool = True


DIRECT_MODEL_SPECS: tuple[DirectModelSpec, ...] = (
    DirectModelSpec(
        "b1",
        "ConvNeXt-Tiny (B1)",
        "CNN",
        "oncoderm_mambax_b1_melanoma_head",
        "prob_mel_endpoint",
        "project implementation",
    ),
    DirectModelSpec(
        "swin_tiny",
        "Swin-Tiny",
        "Transformer",
        "swin_tiny",
        "prob_mel",
        "timm ImageNet pretrained",
    ),
    DirectModelSpec(
        "dermamamba",
        "DermaMamba (resource-matched architecture port)",
        "CNN + selective SSM",
        "dermamamba_architecture_faithful",
        "prob_mel",
        "resource-matched clean-room port of Glow945/Skin_Lesions",
    ),
    DirectModelSpec(
        "b2",
        "OncoDerm-MambaX B2",
        "CNN + metadata",
        "oncoderm_mambax_b2_metadata_cmgf",
        "prob_mel_endpoint",
        "project implementation",
    ),
    DirectModelSpec(
        "b4",
        "OncoDerm-MambaX B4",
        "CNN + lesion prior",
        "oncoderm_mambax_b4_lpah_mask_aux",
        "prob_mel_endpoint",
        "project implementation",
    ),
    DirectModelSpec(
        "b6",
        "OncoDerm-MambaX B6",
        "Full proposed model",
        "oncoderm_mambax_b6_full",
        "prob_mel_endpoint",
        "project implementation",
    ),
)


OPTIONAL_MODEL_SPECS: tuple[DirectModelSpec, ...] = (
    DirectModelSpec(
        "efficientnet_b0",
        "EfficientNet-B0",
        "CNN",
        "efficientnet_b0",
        "prob_mel",
        "timm ImageNet pretrained",
        required=False,
    ),
    DirectModelSpec(
        "b6m",
        "OncoDerm-MambaX B6-M",
        "Full model + official Mamba block",
        "oncoderm_mambax_b6m_true_mamba",
        "prob_mel_endpoint",
        "state-spaces/mamba adapter",
        required=False,
    ),
    DirectModelSpec(
        "b6_crosssplit2024",
        "OncoDerm-MambaX B6 (split seed 2024)",
        "Full proposed model",
        "oncoderm_mambax_b6_full_crosssplit2024",
        "prob_mel_endpoint",
        "project implementation; independent lesion split",
        required=False,
    ),
)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dermamamba_protocol_ok(
    config: dict[str, Any], locked_revision: str | None
) -> tuple[bool, dict[str, Any]]:
    expected: dict[str, Any] = {
        "project.seed": 42,
        "project.split_seed": 42,
        "data.input_size": 224,
        "data.metadata.enabled": False,
        "data.mask_aux.enabled": False,
        "train.epochs": 100,
        "train.batch_size": 16,
        "train.optimizer": "adam",
        "train.lr": 1e-4,
        "train.weight_decay": 5e-5,
        "train.warmup_epochs": 10,
        "train.scheduler": "cosine",
        "train.grad_clip": 1.0,
        "train.early_stopping_patience": 15,
        "train.selection_metric": "macro_f1",
        "train.selection_tie_breaker": "mel_auc",
        "loss.name": "ce",
        "loss.label_smoothing": 0.0,
        "loss.lambda_sparse": 0.0,
        "loss.lambda_tv": 0.0,
        "model.source": "dermamamba_architecture_faithful",
        "model.pretrained": True,
        "model.allow_random_init_fallback": False,
        "model.num_classes": 7,
        "model.dermamamba.source_revision": locked_revision,
        "model.dermamamba.patch_size": 16,
        "model.dermamamba.embed_dim": 192,
        "model.dermamamba.depth": 6,
        "model.dermamamba.d_state": 16,
        "model.dermamamba.d_conv": 4,
        "model.dermamamba.expand": 2,
        "model.dermamamba.fusion_dim": 1024,
        "model.dermamamba.dropout": 0.2,
    }

    def nested_value(dotted: str) -> Any:
        value: Any = config
        for key in dotted.split("."):
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    mismatches: dict[str, dict[str, Any]] = {}
    for key, expected_value in expected.items():
        observed = nested_value(key)
        if isinstance(expected_value, bool):
            matches = observed is expected_value
        elif isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            matches = bool(np.isclose(observed, expected_value)) if observed is not None else False
        else:
            matches = observed == expected_value
        if not matches:
            mismatches[key] = {"expected": expected_value, "observed": observed}
    return not mismatches, {"expected": expected, "mismatches": mismatches}


def audit_ham_split_files(
    split_dir: str | Path = "splits",
    seed: int = 42,
    metadata_csv: str | Path = "data/raw/HAM10000/HAM10000_metadata.csv",
) -> dict[str, Any]:
    split_root = Path(split_dir)
    paths = {name: split_root / f"ham10000_seed{seed}_{name}.csv" for name in ["train", "val", "test"]}
    issues: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.is_file():
            issues.append(f"missing_{name}_split")
            continue
        frame = pd.read_csv(path)
        frames[name] = frame
        required = {"image_id", "lesion_id", "dx"}
        if not required.issubset(frame.columns):
            issues.append(f"{name}_missing_columns:{sorted(required - set(frame.columns))}")
        if "image_id" in frame and frame["image_id"].astype(str).duplicated().any():
            issues.append(f"{name}_duplicate_image_id")
        if "dx" in frame and not set(frame["dx"].dropna().astype(str)).issubset(LABEL_MAP):
            issues.append(f"{name}_unknown_class")

    leakage: dict[str, Any] = {"ok": False, "overlaps": {}}
    metadata_coverage: dict[str, Any] = {}
    if len(frames) == 3 and all({"image_id", "lesion_id", "dx"}.issubset(frame.columns) for frame in frames.values()):
        leakage = leakage_report(frames["train"], frames["val"], frames["test"])
        if not leakage.get("ok"):
            issues.append("cross_split_image_or_group_leakage")
        combined = pd.concat(frames.values(), ignore_index=True)
        if combined["image_id"].astype(str).duplicated().any():
            issues.append("cross_split_duplicate_image_id")
        metadata_path = Path(metadata_csv)
        if metadata_path.is_file():
            metadata = pd.read_csv(metadata_path)
            metadata = metadata[metadata["dx"].isin(LABEL_MAP)].copy() if "dx" in metadata else metadata
            if "image_id" not in metadata:
                issues.append("metadata_missing_image_id")
            else:
                split_ids = set(combined["image_id"].astype(str))
                metadata_ids = set(metadata["image_id"].astype(str))
                missing_from_splits = sorted(metadata_ids - split_ids)
                unknown_in_splits = sorted(split_ids - metadata_ids)
                metadata_coverage = {
                    "eligible_metadata_rows": int(len(metadata)),
                    "combined_split_rows": int(len(combined)),
                    "missing_from_splits_count": len(missing_from_splits),
                    "unknown_in_splits_count": len(unknown_in_splits),
                    "missing_from_splits_preview": missing_from_splits[:20],
                    "unknown_in_splits_preview": unknown_in_splits[:20],
                }
                if missing_from_splits or unknown_in_splits or len(metadata) != len(combined):
                    issues.append("split_rows_do_not_exactly_cover_eligible_metadata")

    return {
        "status": "passed" if not issues else "failed",
        "seed": int(seed),
        "issues": issues,
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
        "sha256": {name: file_sha256(path) for name, path in paths.items() if path.is_file()},
        "paths": {name: str(path) for name, path in paths.items()},
        "leakage": leakage,
        "metadata_coverage": metadata_coverage,
    }


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def run_seed(run_dir: Path) -> int | None:
    try:
        return int(load_yaml(run_dir / "config.yaml").get("project", {}).get("seed"))
    except Exception:
        return None


def split_seed(run_dir: Path) -> int | None:
    try:
        project = load_yaml(run_dir / "config.yaml").get("project", {})
        return int(project.get("split_seed", project.get("seed", 42)))
    except Exception:
        return None


def discover_run(
    runs_dir: Path,
    spec: DirectModelSpec,
    seed: int = 42,
    expected_split_seed: int = 42,
) -> Path | None:
    candidates = []
    predictions_dir = runs_dir.parent / "predictions"
    for path in runs_dir.glob(f"{spec.run_prefix}_seed{seed}_*"):
        if not path.is_dir() or not (path / "config.yaml").exists():
            continue
        if run_seed(path) != seed:
            continue
        if split_seed(path) != expected_split_seed:
            continue
        checkpoint = path / "checkpoints" / "best_val_macro_f1.pth"
        if not checkpoint.exists():
            checkpoint = path / "checkpoints" / "last.pth"
        prediction_triplet = all(
            prediction_path(predictions_dir, path, split).is_file()
            for split in ("val", "test", "external")
        )
        if checkpoint.exists() and ((path / "TRAINING_COMPLETE").is_file() or prediction_triplet):
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.name)[-1] if candidates else None


def prediction_path(predictions_dir: Path, run_dir: Path, split: str) -> Path:
    return predictions_dir / f"{run_dir.name}_{split}_predictions.csv"


def validate_internal_predictions(df: pd.DataFrame, source: Path) -> None:
    required = {"image_id", "true_label", *PROBABILITY_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing internal prediction columns: {missing}")
    if df["image_id"].astype(str).duplicated().any():
        raise ValueError(f"{source} contains duplicate image_id values")
    probs = df[PROBABILITY_COLUMNS].astype(float).to_numpy()
    if not np.isfinite(probs).all():
        raise ValueError(f"{source} contains non-finite probabilities")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=2e-4):
        raise ValueError(f"{source} seven-class probabilities do not sum to one")


def validate_external_predictions(df: pd.DataFrame, source: Path) -> None:
    required = {"image_id", "label", "prob_mel"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing external prediction columns: {missing}")
    if df["image_id"].astype(str).duplicated().any():
        raise ValueError(f"{source} contains duplicate external image_id values")
    labels = set(df["label"].astype(int).unique().tolist())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(f"{source} must contain both binary external classes, got {sorted(labels)}")
    scores = df["prob_mel"].astype(float).to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"{source} contains invalid melanoma probabilities")


def validate_score_column(df: pd.DataFrame, score_column: str, source: Path) -> None:
    if score_column not in df.columns:
        raise ValueError(f"{source} is missing required score column {score_column}")
    scores = df[score_column].astype(float).to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"{source} contains invalid values in {score_column}")


def load_prediction_triplet(
    predictions_dir: Path,
    run_dir: Path,
    spec: DirectModelSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_path = prediction_path(predictions_dir, run_dir, "val")
    internal_path = prediction_path(predictions_dir, run_dir, "test")
    external_path = prediction_path(predictions_dir, run_dir, "external")
    paths = [validation_path, internal_path, external_path]
    if any(not path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError(f"Missing prediction files for {run_dir.name}: {missing}")
    validation = pd.read_csv(validation_path)
    internal = pd.read_csv(internal_path)
    external = pd.read_csv(external_path)
    validate_internal_predictions(validation, validation_path)
    validate_internal_predictions(internal, internal_path)
    validate_external_predictions(external, external_path)
    for frame, source in [
        (validation, validation_path),
        (internal, internal_path),
        (external, external_path),
    ]:
        validate_score_column(frame, spec.intended_score_column, source)
    return validation, internal, external


def load_prediction_bundle(
    predictions_dir: Path,
    run_dir: Path,
    spec: DirectModelSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, internal, external = load_prediction_triplet(predictions_dir, run_dir, spec)
    return internal, external


def internal_metrics(df: pd.DataFrame, score_column: str) -> dict[str, float]:
    labels = df["true_label"].astype(int).to_numpy()
    probs = df[PROBABILITY_COLUMNS].astype(float).to_numpy()
    metrics = multiclass_metrics(labels, probs)
    binary = (labels == LABEL_MAP["mel"]).astype(int)
    score = df[score_column].astype(float).to_numpy()
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_auc": float(metrics["macro_auc"]),
        "mel_auc": float(roc_auc_score(binary, score)),
        "n": int(len(df)),
        "melanoma_positive": int(binary.sum()),
    }


def validation_thresholds(df: pd.DataFrame, score_column: str) -> dict[str, float]:
    labels = (df["true_label"].astype(int).to_numpy() == LABEL_MAP["mel"]).astype(int)
    scores = df[score_column].astype(float).to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError("Validation predictions must contain melanoma and non-melanoma samples")
    return {
        "youden": threshold_youden(labels, scores),
        "spec90": threshold_for_specificity(labels, scores, 0.90),
        "spec95": threshold_for_specificity(labels, scores, 0.95),
    }


def external_metrics(
    df: pd.DataFrame,
    score_column: str,
    thresholds: dict[str, float],
) -> dict[str, float]:
    labels = df["label"].astype(int).to_numpy()
    scores = df[score_column].astype(float).to_numpy()
    labels7 = np.where(labels == 1, LABEL_MAP["mel"], LABEL_MAP["nv"])
    metrics, _ = endpoint_metrics_from_scores(labels7, scores, thresholds=thresholds)
    return {
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
        "brier": float(metrics["brier"]),
        "ece": float(metrics["ece"]),
        "n": int(len(df)),
        "melanoma_positive": int(labels.sum()),
    }


def aligned_external(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_score: str,
    second_score: str,
) -> pd.DataFrame:
    a = first[["image_id", "label", first_score]].rename(
        columns={"label": "label_a", first_score: "score_a"}
    )
    b = second[["image_id", "label", second_score]].rename(
        columns={"label": "label_b", second_score: "score_b"}
    )
    merged = a.merge(b, on="image_id", how="inner", validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError(f"External prediction alignment is incomplete: {len(a)} vs {len(b)} vs {len(merged)}")
    if not np.array_equal(merged["label_a"].to_numpy(), merged["label_b"].to_numpy()):
        raise ValueError("External labels disagree after image_id alignment")
    return merged.rename(columns={"label_a": "label"}).drop(columns=["label_b"])


def paired_bootstrap_delta(
    aligned: pd.DataFrame,
    metric_name: str,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    metric_functions: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "roc_auc": lambda y, s: float(roc_auc_score(y, s)),
        "pr_auc": lambda y, s: float(average_precision_score(y, s)),
    }
    if metric_name not in metric_functions:
        raise ValueError(f"Unsupported paired-bootstrap metric: {metric_name}")
    metric = metric_functions[metric_name]
    labels = aligned["label"].astype(int).to_numpy()
    score_a = aligned["score_a"].astype(float).to_numpy()
    score_b = aligned["score_b"].astype(float).to_numpy()
    observed_a = metric(labels, score_a)
    observed_b = metric(labels, score_b)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        index = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[index]
        if len(np.unique(sampled_labels)) < 2:
            continue
        deltas.append(metric(sampled_labels, score_a[index]) - metric(sampled_labels, score_b[index]))
    values = np.asarray(deltas, dtype=float)
    if len(values) == 0:
        raise RuntimeError("No valid paired bootstrap replicate was produced")
    # The plus-one correction prevents an impossible p=0 from a finite number
    # of Monte Carlo bootstrap replicates.
    lower_tail = (int(np.count_nonzero(values <= 0)) + 1) / (len(values) + 1)
    upper_tail = (int(np.count_nonzero(values >= 0)) + 1) / (len(values) + 1)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return {
        "metric": metric_name,
        "system_a": "b6",
        "system_b": "dermamamba",
        "estimate_a": observed_a,
        "estimate_b": observed_b,
        "delta_a_minus_b": observed_a - observed_b,
        "ci_lower": float(np.percentile(values, 2.5)),
        "ci_upper": float(np.percentile(values, 97.5)),
        "p_value_two_sided": p_value,
        "p_value_method": "two-sided empirical paired bootstrap with plus-one correction",
        "iterations_requested": int(iterations),
        "iterations_valid": int(len(values)),
        "resampling_unit": "external_image_id",
        "seed": int(seed),
        "n": int(len(labels)),
        "positive_n": int(labels.sum()),
    }


def efficiency_lookup(results_dir: Path) -> dict[str, dict[str, Any]]:
    path = results_dir / "efficiency_table.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "run_id" not in frame.columns:
        return {}
    return {str(row["run_id"]): row.to_dict() for _, row in frame.iterrows()}


def run_provenance(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    environment_path = run_dir / "environment.json"
    config = load_yaml(config_path)
    environment = read_json(environment_path, {})
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(
            run_dir / "checkpoints" / "best_val_macro_f1.pth"
            if (run_dir / "checkpoints" / "best_val_macro_f1.pth").exists()
            else run_dir / "checkpoints" / "last.pth"
        ),
        "run_seed": int(config.get("project", {}).get("seed", 42)),
        "split_seed": int(config.get("project", {}).get("split_seed", 42)),
        "input_size": int(config.get("data", {}).get("input_size", 224)),
        "model_metadata": environment.get("model_metadata", {}),
        "environment": {k: v for k, v in environment.items() if k != "model_metadata"},
    }
