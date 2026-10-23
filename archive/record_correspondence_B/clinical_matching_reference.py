"""Reference numerical kernels for OncoDerm original-model clinical reassignment.

No file discovery, torch imports, model loading or access to project data occurs here.
The caller must bind original checkpoint/image/metadata identities before execution.
Tested with synthetic data only by ChatGPT. These are NOT patient/model results.

Array contract:
    table: (6, N, U) probabilities for ONE head; rows are fixed input images;
           U includes every observed 22-D clinical vector and E_NULL.
    types: (N,) original clinical-vector indices, all integers.
    strata: tuple of int64 arrays, each shape (m, n), grouping the canonical
            original image rows by patient/case block size n. Every row occurs once.
    model order: B2 42/2024/3407, B6 42/2024/3407.

Bootstrap resamples complete recipient blocks WITH clinical records/labels, within
original block-size strata. Reassignment is regenerated on resampled block COPIES.
Copies of the same original subject may serve as each other's donors in bootstrap;
this is an empirical bootstrap artifact, not an original-data reassignment rule.
The original 19 singleton-size SIIM blocks remain fixed. Interpret uncertainty
conditional on the observed size-stratum counts and these singleton blocks.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Sequence
import numpy as np

BASE_SEED = 20261020
K_REASSIGNMENTS = 50
B_BOOTSTRAP = 2000
COHORT_CODES = {"SIIM": 1, "Derm7pt": 2}


def stream(cohort: int, phase: int, draw: int = 0, repeat: int = 0,
           attempt: int = 0) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [BASE_SEED, int(cohort), int(phase), int(draw), int(repeat), int(attempt)])))


def make_strata(blocks: Sequence[np.ndarray], n_rows: int) -> tuple[np.ndarray, ...]:
    """Caller supplies canonically sorted blocks and sorted rows within each block."""
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for b in blocks:
        a = np.asarray(b, dtype=np.int64)
        if a.ndim != 1 or len(a) == 0:
            raise ValueError("Every block must be a nonempty one-dimensional index array")
        grouped[len(a)].append(a)
    flat = np.concatenate([np.asarray(b, dtype=np.int64) for b in blocks])
    if len(flat) != n_rows or not np.array_equal(np.sort(flat), np.arange(n_rows)):
        raise ValueError("Blocks must partition 0..N-1 exactly once")
    return tuple(np.stack(grouped[size]) for size in sorted(grouped))


def derangement(m: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform permutation conditional on no self donor; no score/label access."""
    if m < 2:
        raise ValueError("No derangement exists for fewer than two block copies")
    if m == 2:
        return np.array([1, 0], dtype=np.int64)
    identity = np.arange(m)
    for _ in range(10000):
        p = rng.permutation(m)
        if np.all(p != identity):
            return p
    raise RuntimeError("Unexpected RNG rejection limit; do not change the design")


def donor_slots(strata: Sequence[np.ndarray], n_rows: int,
                rng: np.random.Generator) -> np.ndarray:
    """Return recipient-slot -> donor-slot bijection. Singleton-size blocks stay native.

    For each size stratum: derange whole blocks, then independently randomize the
    order of complete records inside each assigned donor block. No record is
    replicated/dropped; fields never permute separately. Row order randomization
    is part of this specified stress intervention, NOT a null exchangeability claim.
    """
    result = np.arange(n_rows, dtype=np.int64)
    for a in strata:
        m, n = a.shape
        if m == 1:
            continue
        donors = a[derangement(m, rng)]
        within = rng.permuted(np.broadcast_to(np.arange(n), (m, n)), axis=1)
        result[a] = np.take_along_axis(donors, within, axis=1)
    return result


def resample_rows(strata: Sequence[np.ndarray], n_rows: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Recipient slot -> original row. Sample m entire blocks from each size stratum."""
    out = np.empty(n_rows, dtype=np.int64)
    for a in strata:
        m, _ = a.shape
        if m == 1:
            out[a] = a  # explicit conditional singleton treatment
        else:
            sampled = rng.integers(0, m, size=m)
            out[a] = a[sampled]
    return out


def ap_auc(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Tie-aware noninterpolated AP and ROC-AUC; matches sklearn definitions."""
    y = np.asarray(y)
    score = np.asarray(score)
    if y.ndim != 1 or score.shape != y.shape:
        raise ValueError("Expected equal-length one-dimensional inputs")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("Labels must be binary and scores finite")
    p = float(np.sum(y, dtype=np.float64))
    n = len(y) - p
    if p == 0 or n == 0:
        raise ValueError("Two classes required for joint AP/AUC output")
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(y) - 1]
    tp = np.cumsum(yy, dtype=np.float64)[ends]
    fp = (ends + 1) - tp
    prev_tp, prev_fp = np.r_[0.0, tp[:-1]], np.r_[0.0, fp[:-1]]
    ap = np.sum((tp - prev_tp) * tp / (ends + 1)) / p
    auc = np.sum((fp - prev_fp) * (tp + prev_tp)) / (2.0 * p * n)
    return np.array([ap, auc], dtype=np.float64)


def run_metrics(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[0] != 6 or scores.shape[1] != len(y):
        raise ValueError("Expected scores shape (6,N) in fixed model order")
    return np.stack([ap_auc(y, p) for p in scores])


def family_mean(per_run: np.ndarray) -> np.ndarray:
    if per_run.shape != (6, 2):
        raise ValueError("Expected six runs by AP/AUC")
    return per_run.reshape(2, 3, 2).mean(axis=1)


def observed_reassignments(table: np.ndarray, types: np.ndarray, null_type: int,
                          y: np.ndarray, strata: Sequence[np.ndarray], cohort: int,
                          repeats: int = K_REASSIGNMENTS) -> dict[str, np.ndarray]:
    """Only evaluation. This function never averages probabilities across mappings."""
    n = len(y)
    rows = np.arange(n)
    matched = run_metrics(y, table[:, rows, types])
    null = run_metrics(y, table[:, rows, null_type])
    shuffled = np.empty((repeats, 6, 2), dtype=np.float64)
    for k in range(repeats):
        mapping = donor_slots(strata, n, stream(cohort, 1, repeat=k))
        shuffled[k] = run_metrics(y, table[:, rows, types[mapping]])
    mu = shuffled.mean(axis=0)
    return {"matched_per_run": matched, "null_per_run": null,
            "shuffled_per_repeat_per_run": shuffled,
            "matched_minus_shuffled_per_run": matched - mu,
            "shuffled_minus_null_per_run": mu - null}


def bootstrap_one(table: np.ndarray, types: np.ndarray, null_type: int,
                  y: np.ndarray, strata: Sequence[np.ndarray], cohort: int,
                  draw: int, repeats: int = K_REASSIGNMENTS) -> dict[str, np.ndarray | int]:
    """One deterministic, resumable nested replicate. No use of fixed old donor scores.

    Rebuilds both the recipient sample and the donor pool from resampled complete
    patient/case blocks, then averages the SAME K independent reassignments used in
    the point-estimator definition. Same rows and mappings apply to all six fits.
    Callers run draw=0..1999; invalid one-class samples are redrawn independently.
    """
    n = len(y)
    for attempt in range(10000):
        rows = resample_rows(strata, n, stream(cohort, 2, draw=draw, attempt=attempt))
        yy = y[rows]
        if 0 < np.sum(yy) < n:
            break
    else:
        raise RuntimeError("Cannot obtain a two-class bootstrap sample")
    matched = run_metrics(yy, table[:, rows, types[rows]])
    null = run_metrics(yy, table[:, rows, null_type])
    mean_shuffled = np.zeros((6, 2), dtype=np.float64)
    for k in range(repeats):
        slots = donor_slots(strata, n, stream(cohort, 3, draw=draw, repeat=k, attempt=attempt))
        # Both image and clinical source indices come from the resampled sample.
        donor_rows = rows[slots]
        mean_shuffled += run_metrics(yy, table[:, rows, types[donor_rows]]) / repeats
    return {"matched_minus_shuffled": family_mean(matched - mean_shuffled),
            "shuffled_minus_null": family_mean(mean_shuffled - null),
            "rejected_one_class_samples": attempt}
