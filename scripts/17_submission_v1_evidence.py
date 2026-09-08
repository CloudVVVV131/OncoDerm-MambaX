from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import platform
import re
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import ensure_dir, write_json


MEL_INDEX = 4
EPS = 1e-7

MODEL_INFO = {
    "b2": {
        "name": "B2 + CMGF metadata",
        "pattern": "oncoderm_mambax_b2_metadata_cmgf_seed{seed}_*",
    },
    "b4": {
        "name": "B4 + LPAH mask aux",
        "pattern": "oncoderm_mambax_b4_lpah_mask_aux_seed{seed}_*",
    },
    "b6": {
        "name": "B6 full OncoDerm-MambaX",
        "pattern": "oncoderm_mambax_b6_full_seed{seed}_*",
    },
}

METRIC_HIGHER_IS_BETTER = {
    "roc_auc": True,
    "pr_auc": True,
    "brier": False,
    "ece": False,
    "adaptive_ece": False,
    "nll": False,
}


@dataclass(frozen=True)
class EvidencePaths:
    result_root: Path
    predictions_dir: Path
    runs_dir: Path
    results_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class RunRecord:
    model_id: str
    seed: int
    run_id: str
    run_dir: Path
    config_path: Path


@dataclass
class PredictionBundle:
    system_id: str
    system_type: str
    members: list[str]
    val_scores: np.ndarray
    val_labels: np.ndarray
    test_scores: np.ndarray
    test_labels: np.ndarray
    external_scores: np.ndarray
    external_labels: np.ndarray
    external_image_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V1.2 evidence consolidation for OncoDerm-MambaX.")
    parser.add_argument("--config", default="configs/eval/submission_v1_evidence.yaml")
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--external-only-bootstrap", action="store_true")
    return parser.parse_args()


def normalize_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def resolve_paths(config: dict[str, Any], result_root_override: str | None) -> EvidencePaths:
    paths_cfg = config.get("paths", {})
    root = Path(result_root_override or paths_cfg.get("result_root", ".")).resolve()
    return EvidencePaths(
        result_root=root,
        predictions_dir=normalize_path(root, paths_cfg.get("predictions_dir", "outputs/predictions")),
        runs_dir=normalize_path(root, paths_cfg.get("runs_dir", "outputs/runs")),
        results_dir=ensure_dir(normalize_path(root, paths_cfg.get("results_dir", "results"))),
        figures_dir=ensure_dir(normalize_path(root, paths_cfg.get("figures_dir", "figures/evidence"))),
    )


def status_base(config: dict[str, Any], paths: EvidencePaths) -> dict[str, Any]:
    return {
        "status": "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_root": str(paths.result_root),
        "required_models": config.get("models", {}).get("required_models", []),
        "required_seeds": config.get("models", {}).get("required_seeds", []),
        "found_runs": {},
        "missing_runs": [],
        "duplicate_candidates": [],
        "alignment_errors": [],
        "fallbacks": [],
        "tables_written": [],
        "figures_written": [],
        "skipped": [],
        "notes": [],
    }


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()




def discover_required_runs(config: dict[str, Any], paths: EvidencePaths, status: dict[str, Any]) -> dict[tuple[str, int], RunRecord]:
    models_cfg = config.get("models", {})
    required_models = [str(m).lower() for m in models_cfg.get("required_models", ["b2", "b4", "b6"])]
    required_seeds = [int(s) for s in models_cfg.get("required_seeds", [42, 2024, 3407])]
    run_patterns = models_cfg.get("run_patterns", {})
    records: dict[tuple[str, int], RunRecord] = {}
    found_runs: dict[str, dict[str, str]] = {m: {} for m in required_models}
    for model_id in required_models:
        for seed in required_seeds:
            default_pattern = MODEL_INFO.get(model_id, {}).get("pattern", f"*{model_id}*seed{{seed}}_*")
            pattern = str(run_patterns.get(model_id, default_pattern)).format(seed=seed)
            candidates = sorted(p for p in paths.runs_dir.glob(pattern) if p.is_dir() and (p / "config.yaml").exists())
            if len(candidates) > 1:
                status["duplicate_candidates"].append(
                    {
                        "model_id": model_id,
                        "seed": seed,
                        "candidates": [p.name for p in candidates],
                    }
                )
                raise ValueError(
                    f"Ambiguous runs for {model_id} seed {seed}; narrow models.run_patterns "
                    "or use a directory containing only the intended runs."
                )
            chosen = candidates[0] if candidates else None
            if chosen is None:
                status["missing_runs"].append({"model_id": model_id, "seed": seed, "pattern": pattern})
                continue
            record = RunRecord(
                model_id=model_id,
                seed=seed,
                run_id=chosen.name,
                run_dir=chosen,
                config_path=chosen / "config.yaml",
            )
            records[(model_id, seed)] = record
            found_runs[model_id][str(seed)] = chosen.name
    status["found_runs"] = found_runs
    return records


def to_binary_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    unique = set(np.unique(labels).astype(int).tolist())
    if unique.issubset({0, 1}):
        return labels.astype(int)
    return (labels.astype(int) == MEL_INDEX).astype(int)


def load_endpoint_scores(predictions_dir: Path, run_id: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    labels_path = predictions_dir / f"{run_id}_{split}_labels.npy"
    endpoint_path = predictions_dir / f"{run_id}_{split}_endpoint_probs.npy"
    probs_path = predictions_dir / f"{run_id}_{split}_probs.npy"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    labels = to_binary_labels(np.load(labels_path))
    if endpoint_path.exists():
        scores = np.load(endpoint_path).reshape(-1).astype(float)
    elif probs_path.exists():
        probs = np.load(probs_path)
        scores = probs[:, MEL_INDEX].reshape(-1).astype(float)
    else:
        raise FileNotFoundError(endpoint_path)
    if len(labels) != len(scores):
        raise ValueError(f"{run_id} {split}: labels length {len(labels)} != scores length {len(scores)}")
    return labels, clip_scores(scores)


def load_external_frame(predictions_dir: Path, run_id: str, status: dict[str, Any]) -> pd.DataFrame:
    path = predictions_dir / f"{run_id}_external_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"image_id", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if "prob_mel_endpoint" in df.columns:
        score_col = "prob_mel_endpoint"
    elif "prob_mel" in df.columns:
        score_col = "prob_mel"
        status["fallbacks"].append({"run_id": run_id, "fallback": "prob_mel_endpoint_missing_used_prob_mel"})
    else:
        raise ValueError(f"{path} missing prob_mel_endpoint and prob_mel")
    out = df[["image_id", "label", score_col]].copy()
    out = out.rename(columns={score_col: "score"})
    out["image_id"] = out["image_id"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    out["score"] = pd.to_numeric(out["score"], errors="raise").astype(float)
    if out["image_id"].duplicated().any():
        dupes = out.loc[out["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise ValueError(f"{path} duplicated image_id examples: {dupes}")
    out["score"] = clip_scores(out["score"].to_numpy())
    return out


def clip_scores(scores: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)


def build_single_systems(records: dict[tuple[str, int], RunRecord], paths: EvidencePaths, status: dict[str, Any]) -> dict[str, PredictionBundle]:
    systems: dict[str, PredictionBundle] = {}
    for (model_id, seed), record in sorted(records.items()):
        val_labels, val_scores = load_endpoint_scores(paths.predictions_dir, record.run_id, "val")
        test_labels, test_scores = load_endpoint_scores(paths.predictions_dir, record.run_id, "test")
        external_df = load_external_frame(paths.predictions_dir, record.run_id, status).sort_values("image_id")
        system_id = f"{model_id}_seed{seed}"
        systems[system_id] = PredictionBundle(
            system_id=system_id,
            system_type="single_seed",
            members=[record.run_id],
            val_scores=val_scores,
            val_labels=val_labels,
            test_scores=test_scores,
            test_labels=test_labels,
            external_scores=external_df["score"].to_numpy(dtype=float),
            external_labels=external_df["label"].to_numpy(dtype=int),
            external_image_ids=external_df["image_id"].to_numpy(dtype=str),
        )
    return systems


def member_token_to_system_id(token: str) -> str:
    model_id, seed = token.split(":", 1)
    return f"{model_id.lower()}_seed{int(seed)}"


def assert_same_labels(reference: np.ndarray, current: np.ndarray, context: str) -> None:
    if reference.shape != current.shape or not np.array_equal(reference, current):
        raise ValueError(f"{context}: labels do not align")


def build_external_ensemble(member_bundles: list[PredictionBundle], context: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged: pd.DataFrame | None = None
    for idx, bundle in enumerate(member_bundles):
        df = pd.DataFrame(
            {
                "image_id": bundle.external_image_ids,
                f"label_{idx}": bundle.external_labels,
                f"score_{idx}": bundle.external_scores,
            }
        )
        merged = df if merged is None else merged.merge(df, on="image_id", how="inner")
    if merged is None or merged.empty:
        raise ValueError(f"{context}: empty external merge")
    expected_n = len(member_bundles[0].external_image_ids)
    if len(merged) != expected_n:
        raise ValueError(f"{context}: external merge kept {len(merged)} rows, expected {expected_n}")
    label_cols = [c for c in merged.columns if c.startswith("label_")]
    first = merged[label_cols[0]].to_numpy(dtype=int)
    for col in label_cols[1:]:
        if not np.array_equal(first, merged[col].to_numpy(dtype=int)):
            raise ValueError(f"{context}: external labels mismatch in {col}")
    score_cols = [c for c in merged.columns if c.startswith("score_")]
    scores = merged[score_cols].to_numpy(dtype=float).mean(axis=1)
    return merged["image_id"].to_numpy(dtype=str), first, clip_scores(scores)


def build_ensembles(single_systems: dict[str, PredictionBundle], config: dict[str, Any], status: dict[str, Any]) -> dict[str, PredictionBundle]:
    ensembles: dict[str, PredictionBundle] = {}
    for ensemble_id, tokens in config.get("ensembles", {}).items():
        member_ids = [member_token_to_system_id(str(token)) for token in tokens]
        missing = [member_id for member_id in member_ids if member_id not in single_systems]
        if missing:
            status["alignment_errors"].append({"system_id": ensemble_id, "missing_members": missing})
            continue
        members = [single_systems[member_id] for member_id in member_ids]
        val_labels = members[0].val_labels
        test_labels = members[0].test_labels
        for member in members[1:]:
            assert_same_labels(val_labels, member.val_labels, f"{ensemble_id} val")
            assert_same_labels(test_labels, member.test_labels, f"{ensemble_id} test")
        external_ids, external_labels, external_scores = build_external_ensemble(members, ensemble_id)
        system_type = "seed_ensemble" if len({m.split("_seed")[0] for m in member_ids}) == 1 else "model_ensemble"
        ensembles[ensemble_id] = PredictionBundle(
            system_id=ensemble_id,
            system_type=system_type,
            members=[member.system_id for member in members],
            val_scores=clip_scores(np.stack([m.val_scores for m in members], axis=0).mean(axis=0)),
            val_labels=val_labels,
            test_scores=clip_scores(np.stack([m.test_scores for m in members], axis=0).mean(axis=0)),
            test_labels=test_labels,
            external_scores=external_scores,
            external_labels=external_labels,
            external_image_ids=external_ids,
        )
    return ensembles


def expected_calibration_error_binary(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = clip_scores(scores)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (scores >= lo if idx == 0 else scores > lo) & (scores <= hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(scores[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def adaptive_ece_binary(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = clip_scores(scores)
    n = len(scores)
    if n == 0:
        return float("nan")
    order = np.argsort(scores)
    splits = np.array_split(order, min(n_bins, n))
    ece = 0.0
    for idx in splits:
        if len(idx) == 0:
            continue
        acc = float(y_true[idx].mean())
        conf = float(scores[idx].mean())
        ece += (len(idx) / n) * abs(acc - conf)
    return float(ece)


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def safe_nll(y_true: np.ndarray, scores: np.ndarray) -> float:
    scores = np.clip(scores, EPS, 1.0 - EPS)
    return float(log_loss(y_true.astype(int), scores, labels=[0, 1]))


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 15, adaptive_bins: int = 15) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = clip_scores(scores)
    return {
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
        "mean_score": float(np.mean(scores)) if len(scores) else float("nan"),
        "roc_auc": safe_roc_auc(y_true, scores),
        "pr_auc": safe_pr_auc(y_true, scores),
        "brier": float(brier_score_loss(y_true, scores)) if len(y_true) else float("nan"),
        "ece": expected_calibration_error_binary(y_true, scores, n_bins=n_bins),
        "adaptive_ece": adaptive_ece_binary(y_true, scores, n_bins=adaptive_bins),
        "nll": safe_nll(y_true, scores) if len(y_true) else float("nan"),
    }


def dataset_arrays(bundle: PredictionBundle, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "val":
        return bundle.val_labels, bundle.val_scores
    if dataset == "test":
        return bundle.test_labels, bundle.test_scores
    if dataset == "external":
        return bundle.external_labels, bundle.external_scores
    raise ValueError(f"unknown dataset: {dataset}")


def make_metric_tables(systems: dict[str, PredictionBundle], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_bins = int(config.get("calibration", {}).get("n_bins", 15))
    adaptive_bins = int(config.get("calibration", {}).get("adaptive_bins", 15))
    rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    for system_id, bundle in systems.items():
        for dataset in ["val", "test", "external"]:
            labels, scores = dataset_arrays(bundle, dataset)
            metrics = compute_binary_metrics(labels, scores, n_bins=n_bins, adaptive_bins=adaptive_bins)
            row = {
                "system_id": system_id,
                "system_type": bundle.system_type,
                "members": ";".join(bundle.members),
                "n_members": len(bundle.members),
                "dataset": dataset,
                **metrics,
            }
            rows.append(row)
            cal_rows.append(
                {
                    "system_id": system_id,
                    "system_type": bundle.system_type,
                    "dataset": dataset,
                    "n_samples": metrics["n_samples"],
                    "n_positive": metrics["n_positive"],
                    "prevalence": metrics["prevalence"],
                    "mean_score": metrics["mean_score"],
                    "brier": metrics["brier"],
                    "ece": metrics["ece"],
                    "adaptive_ece": metrics["adaptive_ece"],
                    "nll": metrics["nll"],
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        test_auc = out[out["dataset"] == "test"].set_index("system_id")["roc_auc"].to_dict()
        out["auc_drop"] = out.apply(
            lambda r: test_auc.get(r["system_id"], np.nan) - r["roc_auc"] if r["dataset"] == "external" else np.nan,
            axis=1,
        )
        out["recommended_role"] = "supportive"
    return out, pd.DataFrame(cal_rows)


def metric_value(metric_name: str, y_true: np.ndarray, scores: np.ndarray) -> float:
    if metric_name == "roc_auc":
        return safe_roc_auc(y_true, scores)
    if metric_name == "pr_auc":
        return safe_pr_auc(y_true, scores)
    if metric_name == "brier":
        return float(brier_score_loss(y_true, scores))
    if metric_name == "ece":
        return expected_calibration_error_binary(y_true, scores, n_bins=15)
    if metric_name == "adaptive_ece":
        return adaptive_ece_binary(y_true, scores, n_bins=15)
    if metric_name == "nll":
        return safe_nll(y_true, scores)
    raise ValueError(f"unknown metric: {metric_name}")


def paired_bootstrap_difference(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_name: str,
    iters: int,
    seed: int,
    ci: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    alpha = 1.0 - ci
    score_a_val = metric_value(metric_name, y_true, score_a)
    score_b_val = metric_value(metric_name, y_true, score_b)
    deltas: list[float] = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2 and metric_name in {"roc_auc", "pr_auc"}:
            continue
        try:
            va = metric_value(metric_name, y_true[idx], score_a[idx])
            vb = metric_value(metric_name, y_true[idx], score_b[idx])
        except Exception:
            continue
        delta = va - vb
        if math.isfinite(delta):
            deltas.append(float(delta))
    if not deltas:
        return {
            "score_a": score_a_val,
            "score_b": score_b_val,
            "delta": score_a_val - score_b_val,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_two_sided": float("nan"),
            "iters_requested": iters,
            "iters_effective": 0,
        }
    arr = np.asarray(deltas)
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    return {
        "score_a": score_a_val,
        "score_b": score_b_val,
        "delta": score_a_val - score_b_val,
        "ci_lower": float(np.nanpercentile(arr, 100.0 * alpha / 2.0)),
        "ci_upper": float(np.nanpercentile(arr, 100.0 * (1.0 - alpha / 2.0))),
        "p_two_sided": min(p, 1.0),
        "iters_requested": iters,
        "iters_effective": int(len(arr)),
    }


def make_bootstrap_table(systems: dict[str, PredictionBundle], config: dict[str, Any], iters_override: int | None, external_only: bool, status: dict[str, Any]) -> pd.DataFrame:
    bootstrap_cfg = config.get("bootstrap", {})
    iters = int(iters_override or bootstrap_cfg.get("iters", 10000))
    ci = float(bootstrap_cfg.get("ci", 0.95))
    random_seed = int(config.get("project", {}).get("random_seed", 20260710))
    metrics = [str(m) for m in bootstrap_cfg.get("metrics", ["roc_auc", "pr_auc", "brier", "ece", "nll"])]
    datasets = ["external"] if external_only else [str(d) for d in bootstrap_cfg.get("datasets", ["external", "test"])]
    comparisons = [(str(a), str(b)) for a, b in bootstrap_cfg.get("comparisons", [])]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for comp_idx, (system_a, system_b) in enumerate(comparisons):
            if system_a not in systems or system_b not in systems:
                status["skipped"].append(
                    {
                        "task": "paired_bootstrap",
                        "dataset": dataset,
                        "comparison": [system_a, system_b],
                        "reason": "system_missing",
                    }
                )
                continue
            labels_a, scores_a = dataset_arrays(systems[system_a], dataset)
            labels_b, scores_b = dataset_arrays(systems[system_b], dataset)
            assert_same_labels(labels_a, labels_b, f"bootstrap {dataset} {system_a} vs {system_b}")
            for metric_idx, metric in enumerate(metrics):
                result = paired_bootstrap_difference(
                    labels_a,
                    scores_a,
                    scores_b,
                    metric,
                    iters=iters,
                    seed=random_seed + 1000 * comp_idx + metric_idx,
                    ci=ci,
                )
                effective_ratio = result["iters_effective"] / max(result["iters_requested"], 1)
                if effective_ratio < 0.8:
                    status["notes"].append(
                        f"Low bootstrap effective ratio for {dataset} {system_a} vs {system_b} {metric}: {effective_ratio:.3f}"
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison_id": f"{system_a}_vs_{system_b}",
                        "system_a": system_a,
                        "system_b": system_b,
                        "metric": metric,
                        "higher_is_better": METRIC_HIGHER_IS_BETTER.get(metric),
                        "n_samples": int(len(labels_a)),
                        "n_positive": int(labels_a.sum()),
                        "bootstrap_ci": ci,
                        "random_seed": random_seed,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def binary_confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    y_true = y_true.astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "accuracy": float(acc),
        "balanced_accuracy": float((sens + spec) / 2.0),
        "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([np.array([0.0, 1.0]), clip_scores(scores)]))


def select_threshold(y_val: np.ndarray, scores_val: np.ndarray, policy: str) -> dict[str, Any]:
    best_t = 0.5
    best_score = -float("inf")
    feasible = True
    thresholds = candidate_thresholds(scores_val)
    if policy == "youden":
        for t in thresholds:
            stats = binary_confusion(y_val, scores_val, float(t))
            score = stats["sensitivity"] + stats["specificity"] - 1.0
            if score > best_score:
                best_score = score
                best_t = float(t)
    elif policy.startswith("specificity_"):
        target = float(policy.split("_", 1)[1]) / 100.0
        feasible = False
        for t in thresholds:
            stats = binary_confusion(y_val, scores_val, float(t))
            if stats["specificity"] >= target:
                feasible = True
                score = stats["sensitivity"]
                if score > best_score:
                    best_score = score
                    best_t = float(t)
    elif policy.startswith("sensitivity_"):
        target = float(policy.split("_", 1)[1]) / 100.0
        feasible = False
        for t in thresholds:
            stats = binary_confusion(y_val, scores_val, float(t))
            if stats["sensitivity"] >= target:
                feasible = True
                score = stats["specificity"]
                if score > best_score:
                    best_score = score
                    best_t = float(t)
    else:
        raise ValueError(f"unknown threshold policy: {policy}")
    return {"threshold": best_t, "feasible": bool(feasible), "selection_score": float(best_score) if feasible else float("nan")}


def make_threshold_table(systems: dict[str, PredictionBundle], config: dict[str, Any]) -> pd.DataFrame:
    thresh_cfg = config.get("thresholds", {})
    policies = [str(p) for p in thresh_cfg.get("policies", ["youden", "specificity_90", "specificity_95", "sensitivity_80"])]
    target_splits = [str(s) for s in thresh_cfg.get("target_splits", ["val", "test", "external"])]
    rows: list[dict[str, Any]] = []
    for system_id, bundle in systems.items():
        for policy in policies:
            selected = select_threshold(bundle.val_labels, bundle.val_scores, policy)
            for dataset in target_splits:
                labels, scores = dataset_arrays(bundle, dataset)
                stats = binary_confusion(labels, scores, selected["threshold"]) if selected["feasible"] else {}
                rows.append(
                    {
                        "system_id": system_id,
                        "system_type": bundle.system_type,
                        "threshold_policy": policy,
                        "threshold_source": "val",
                        "threshold": selected["threshold"],
                        "feasible": selected["feasible"],
                        "selection_score_on_val": selected["selection_score"],
                        "dataset": dataset,
                        "n_samples": int(len(labels)),
                        "n_positive": int(labels.sum()),
                        **stats,
                    }
                )
    return pd.DataFrame(rows)








def maybe_make_figures(systems: dict[str, PredictionBundle], bootstrap_df: pd.DataFrame, threshold_df: pd.DataFrame, paths: EvidencePaths) -> list[Path]:
    written: list[Path] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return written
    ensure_dir(paths.figures_dir)
    plot_systems = [s for s in ["b2_seed_ensemble", "b6_seed_ensemble", "b2_b6_all6", "b2_b4_b6_all9"] if s in systems]
    if plot_systems:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
        for system_id in plot_systems:
            bundle = systems[system_id]
            if len(np.unique(bundle.external_labels)) < 2:
                continue
            fpr, tpr, _ = roc_curve(bundle.external_labels, bundle.external_scores)
            precision, recall, _ = precision_recall_curve(bundle.external_labels, bundle.external_scores)
            axes[0].plot(fpr, tpr, label=f"{system_id} AUC={safe_roc_auc(bundle.external_labels, bundle.external_scores):.3f}")
            axes[1].plot(recall, precision, label=f"{system_id} AP={safe_pr_auc(bundle.external_labels, bundle.external_scores):.3f}")
        axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate")
        axes[0].set_title("External ROC")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("External PR")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        fig.tight_layout()
        out = paths.figures_dir / "submission_v1_external_roc_pr.png"
        fig.savefig(out)
        plt.close(fig)
        written.append(out)

    if not bootstrap_df.empty:
        sub = bootstrap_df[(bootstrap_df["dataset"] == "external") & (bootstrap_df["metric"].isin(["roc_auc", "pr_auc", "brier", "ece"]))]
        if not sub.empty:
            labels = [f"{r.system_a} vs {r.system_b}\n{r.metric}" for r in sub.itertuples()]
            y = np.arange(len(sub))
            fig_h = max(4, 0.35 * len(sub))
            fig, ax = plt.subplots(figsize=(8, fig_h), dpi=160)
            x = sub["delta"].astype(float).to_numpy()
            lo = sub["ci_lower"].astype(float).to_numpy()
            hi = sub["ci_upper"].astype(float).to_numpy()
            ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="o", markersize=3, capsize=3)
            ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Metric delta (system A - system B)")
            ax.set_title("Paired bootstrap external differences")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            out = paths.figures_dir / "submission_v1_bootstrap_delta_forest.png"
            fig.savefig(out)
            plt.close(fig)
            written.append(out)

    if not threshold_df.empty:
        sub = threshold_df[
            (threshold_df["dataset"] == "external")
            & (threshold_df["system_id"].isin(["b6_seed_ensemble", "b2_b6_all6"]))
            & (threshold_df["threshold_policy"].isin(["youden", "specificity_90", "specificity_95"]))
        ].copy()
        if not sub.empty:
            sub["label"] = sub["system_id"] + "\n" + sub["threshold_policy"]
            metrics = ["sensitivity", "specificity", "ppv", "npv"]
            fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
            x = np.arange(len(sub))
            width = 0.18
            for idx, metric in enumerate(metrics):
                ax.bar(x + (idx - 1.5) * width, sub[metric].astype(float), width=width, label=metric)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["label"], rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1.02)
            ax.set_title("External operating points")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            out = paths.figures_dir / "submission_v1_threshold_operating_points.png"
            fig.savefig(out)
            plt.close(fig)
            written.append(out)

    reliability_systems = [s for s in ["b2_seed_ensemble", "b6_seed_ensemble", "b2_b6_all6"] if s in systems]
    if reliability_systems:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
        for system_id in reliability_systems:
            bundle = systems[system_id]
            bins = np.linspace(0.0, 1.0, 11)
            xs, ys = [], []
            for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
                mask = (bundle.external_scores >= lo if idx == 0 else bundle.external_scores > lo) & (bundle.external_scores <= hi)
                if mask.any():
                    xs.append(float(bundle.external_scores[mask].mean()))
                    ys.append(float(bundle.external_labels[mask].mean()))
            ax.plot(xs, ys, marker="o", label=system_id)
        ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive rate")
        ax.set_title("External reliability")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = paths.figures_dir / "submission_v1_external_reliability.png"
        fig.savefig(out)
        plt.close(fig)
        written.append(out)
    return written


def make_manifest(config_path: Path, config: dict[str, Any], paths: EvidencePaths, output_files: list[Path], bootstrap_iters: int | None) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/17_submission_v1_evidence.py",
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "result_root": str(paths.result_root),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "bootstrap_seed": config.get("project", {}).get("random_seed"),
        "bootstrap_iters_override": bootstrap_iters,
        "training_performed": False,
        "no_external_weight_tuning": True,
        "output_files": [str(p) for p in output_files],
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    if args.bootstrap_iters is not None:
        config.setdefault("bootstrap", {})["iters"] = int(args.bootstrap_iters)
    if args.no_figures:
        config.setdefault("outputs", {})["make_figures"] = False
    paths = resolve_paths(config, args.result_root)
    status = status_base(config, paths)
    status_path = paths.results_dir / "submission_v1_evidence_status.json"
    try:
        for input_dir, label in [(paths.predictions_dir, "predictions_dir"), (paths.runs_dir, "runs_dir")]:
            if not input_dir.exists():
                raise FileNotFoundError(f"{label} does not exist: {input_dir}")
        records = discover_required_runs(config, paths, status)
        if status["missing_runs"]:
            raise RuntimeError(f"Missing required runs: {status['missing_runs']}")
        single_systems = build_single_systems(records, paths, status)
        ensemble_systems = build_ensembles(single_systems, config, status)
        systems = {**single_systems, **ensemble_systems}
        if status["alignment_errors"]:
            raise RuntimeError(f"Alignment errors: {status['alignment_errors']}")

        ensemble_df, calibration_df = make_metric_tables(systems, config)
        bootstrap_df = make_bootstrap_table(systems, config, args.bootstrap_iters, args.external_only_bootstrap, status)
        threshold_df = make_threshold_table(systems, config)

        output_files: list[Path] = []
        table_paths = {
            "ensemble": paths.results_dir / "submission_v1_evidence_ensemble_table.csv",
            "bootstrap": paths.results_dir / "submission_v1_evidence_paired_bootstrap.csv",
            "thresholds": paths.results_dir / "submission_v1_evidence_threshold_operating_points.csv",
            "calibration": paths.results_dir / "submission_v1_evidence_calibration_table.csv",
        }
        ensemble_df.to_csv(table_paths["ensemble"], index=False)
        bootstrap_df.to_csv(table_paths["bootstrap"], index=False)
        threshold_df.to_csv(table_paths["thresholds"], index=False)
        calibration_df.to_csv(table_paths["calibration"], index=False)
        output_files.extend(table_paths.values())


        if bool(config.get("outputs", {}).get("make_figures", True)):
            figure_files = maybe_make_figures(systems, bootstrap_df, threshold_df, paths)
            status["figures_written"] = [str(p) for p in figure_files]
            output_files.extend(figure_files)
        else:
            status["skipped"].append({"task": "figures", "reason": "disabled"})

        manifest = make_manifest(config_path, config, paths, output_files, args.bootstrap_iters)
        manifest_path = paths.results_dir / "submission_v1_evidence_manifest.json"
        write_json(manifest, manifest_path)
        output_files.append(manifest_path)

        status["tables_written"] = [str(p) for p in output_files if p.suffix.lower() in {".csv", ".json"}]
        status["status"] = "completed_with_warnings" if status["fallbacks"] or status["skipped"] or status["notes"] else "completed"
        write_json(status, status_path)
        print(f"submission_v1_evidence status={status['status']} systems={len(systems)} bootstrap_rows={len(bootstrap_df)}")
    except Exception as exc:
        status["status"] = "failed"
        status["notes"].append(str(exc))
        write_json(status, status_path)
        raise


if __name__ == "__main__":
    main()
