from __future__ import annotations

from typing import Any

import pandas as pd


def _overlap(a: pd.Series, b: pd.Series) -> list[str]:
    aa = set(a.dropna().astype(str))
    bb = set(b.dropna().astype(str))
    aa.discard("")
    bb.discard("")
    return sorted(aa.intersection(bb))


def leakage_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_col: str = "image_id",
    lesion_col: str = "lesion_id",
    patient_col: str = "patient_id",
) -> dict[str, Any]:
    splits = {"train": train_df, "val": val_df, "test": test_df}
    report: dict[str, Any] = {"ok": True, "overlaps": {}}
    for col in [image_col, lesion_col, patient_col]:
        if col not in train_df.columns:
            continue
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            ov = _overlap(splits[a][col], splits[b][col])
            report["overlaps"][f"{col}:{a}-{b}"] = ov[:50]
            if ov:
                report["ok"] = False
    return report


def class_distribution(df: pd.DataFrame, label_col: str = "dx") -> dict[str, Any]:
    counts = df[label_col].value_counts().sort_index()
    total = int(counts.sum())
    return {
        "total": total,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "ratio": {str(k): float(v / total) if total else 0.0 for k, v in counts.items()},
    }
