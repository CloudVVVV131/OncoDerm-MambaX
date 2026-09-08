from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lesionmamba.data.datasets import LABEL_MAP
from lesionmamba.data.leakage import class_distribution, leakage_report
from lesionmamba.utils.io import ensure_dir, write_json


def configured_split_seed(config: dict[str, Any]) -> int:
    project_cfg = config.get("project", {})
    return int(project_cfg.get("split_seed", project_cfg.get("seed", 42)))


def _choose_group_column(df: pd.DataFrame, lesion_col: str, patient_col: str) -> str:
    if patient_col in df.columns and df[patient_col].notna().all():
        values = df[patient_col].astype(str).str.strip()
        if values.ne("").all():
            return patient_col
    if lesion_col in df.columns:
        return lesion_col
    return ""


def _group_label_table(df: pd.DataFrame, label_col: str, group_col: str) -> pd.DataFrame:
    if group_col:
        rows = []
        for gid, g in df.groupby(group_col):
            counts = g[label_col].value_counts()
            rows.append({"group": gid, "label": counts.idxmax(), "n": len(g)})
        return pd.DataFrame(rows)
    return pd.DataFrame({"group": df.index, "label": df[label_col], "n": 1})


def grouped_stratified_split(
    df: pd.DataFrame,
    label_col: str = "dx",
    group_col: str = "lesion_id",
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    group_table = _group_label_table(df, label_col, group_col)
    train_groups: list[Any] = []
    val_groups: list[Any] = []
    test_groups: list[Any] = []
    for label, sub in group_table.groupby("label"):
        groups = sub["group"].to_numpy()
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train_groups.extend(groups[:n_train])
        val_groups.extend(groups[n_train : n_train + n_val])
        test_groups.extend(groups[n_train + n_val :])
    if group_col:
        train_df = df[df[group_col].isin(train_groups)].copy()
        val_df = df[df[group_col].isin(val_groups)].copy()
        test_df = df[df[group_col].isin(test_groups)].copy()
    else:
        train_df = df.loc[train_groups].copy()
        val_df = df.loc[val_groups].copy()
        test_df = df.loc[test_groups].copy()
    return train_df, val_df, test_df


def make_splits(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    metadata_csv = Path(data_cfg["metadata_csv"])
    image_dir = Path(data_cfg["image_dir"])
    split_dir = ensure_dir(data_cfg.get("split_dir", "splits"))
    label_col = data_cfg.get("label_column", "dx")
    image_col = data_cfg.get("image_id_column", "image_id")
    lesion_col = data_cfg.get("group_column", "lesion_id")
    patient_col = data_cfg.get("patient_id_column", "patient_id")
    seed = configured_split_seed(config)
    df = pd.read_csv(metadata_csv)
    df = df[df[label_col].isin(LABEL_MAP)].copy()
    group_col = _choose_group_column(df, lesion_col, patient_col)
    ratios = data_cfg.get("split", {})
    split_ratios = (float(ratios.get("train", 0.70)), float(ratios.get("val", 0.15)), float(ratios.get("test", 0.15)))
    train_df, val_df, test_df = grouped_stratified_split(df, label_col, group_col, seed, split_ratios)
    prefix = f"ham10000_seed{seed}"
    paths = {
        "train": split_dir / f"{prefix}_train.csv",
        "val": split_dir / f"{prefix}_val.csv",
        "test": split_dir / f"{prefix}_test.csv",
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    missing = []
    if image_dir.exists():
        indexed = {p.stem for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        missing = [str(x) for x in df[image_col].astype(str).tolist() if str(x) not in indexed]
    leak = leakage_report(train_df, val_df, test_df, image_col, lesion_col, patient_col)
    report = {
        "seed": seed,
        "split_seed": seed,
        "run_seed": int(config.get("project", {}).get("seed", seed)),
        "group_col": group_col,
        "paths": {k: str(v) for k, v in paths.items()},
        "leakage": leak,
        "missing_images_count": len(missing),
        "missing_images_preview": missing[:50],
        "distribution": {
            "train": class_distribution(train_df, label_col),
            "val": class_distribution(val_df, label_col),
            "test": class_distribution(test_df, label_col),
        },
    }
    write_json(report, split_dir / f"split_report_seed{seed}.json")
    write_json(LABEL_MAP, split_dir / "label_map.json")
    table_rows = []
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        row = {"split": name, "n": len(sdf), "melanoma_positive": int((sdf[label_col] == "mel").sum())}
        for label in LABEL_MAP:
            row[label] = int((sdf[label_col] == label).sum())
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    ensure_dir("results")
    table.to_csv("results/dataset_split_table.csv", index=False)
    return report


def split_paths(config: dict[str, Any]) -> dict[str, Path]:
    seed = configured_split_seed(config)
    split_dir = Path(config["data"].get("split_dir", "splits"))
    return {
        "train": split_dir / f"ham10000_seed{seed}_train.csv",
        "val": split_dir / f"ham10000_seed{seed}_val.csv",
        "test": split_dir / f"ham10000_seed{seed}_test.csv",
    }
