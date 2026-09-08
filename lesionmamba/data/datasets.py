from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

DEFAULT_SEX_VALUES = ["male", "female", "unknown"]
DEFAULT_LOCALIZATION_VALUES = [
    "abdomen",
    "acral",
    "back",
    "chest",
    "ear",
    "face",
    "foot",
    "genital",
    "hand",
    "lower extremity",
    "neck",
    "scalp",
    "trunk",
    "unknown",
    "upper extremity",
]


def metadata_dim(metadata_cfg: dict[str, Any] | None = None) -> int:
    cfg = metadata_cfg or {}
    if not cfg.get("enabled", False):
        return 0
    sex_values = cfg.get("sex_values", DEFAULT_SEX_VALUES)
    loc_values = cfg.get("localization_values", DEFAULT_LOCALIZATION_VALUES)
    include_missing = bool(cfg.get("include_missing_indicators", True))
    # age scalar + sex one-hot + localization one-hot + optional missingness flags.
    return 1 + len(sex_values) + len(loc_values) + (3 if include_missing else 0)


def _norm_text(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"


def encode_metadata(row: pd.Series, metadata_cfg: dict[str, Any] | None = None) -> torch.Tensor:
    cfg = metadata_cfg or {}
    dim = metadata_dim(cfg)
    if dim == 0:
        return torch.empty(0, dtype=torch.float32)

    age_column = cfg.get("age_column", "age")
    sex_column = cfg.get("sex_column", "sex")
    loc_column = cfg.get("localization_column", cfg.get("site_column", "localization"))
    age_scale = float(cfg.get("age_scale", 100.0))
    sex_values = list(cfg.get("sex_values", DEFAULT_SEX_VALUES))
    loc_values = list(cfg.get("localization_values", DEFAULT_LOCALIZATION_VALUES))
    include_missing = bool(cfg.get("include_missing_indicators", True))

    age_raw = row.get(age_column, None)
    age_missing = pd.isna(age_raw)
    try:
        age = 0.0 if age_missing else float(age_raw) / max(age_scale, 1.0)
    except (TypeError, ValueError):
        age = 0.0
        age_missing = True

    sex = _norm_text(row.get(sex_column, "unknown"))
    if sex not in sex_values:
        sex = "unknown" if "unknown" in sex_values else sex_values[-1]
    loc = _norm_text(row.get(loc_column, "unknown"))
    if loc not in loc_values:
        loc = "unknown" if "unknown" in loc_values else loc_values[-1]

    values: list[float] = [age]
    values.extend(1.0 if sex == item else 0.0 for item in sex_values)
    values.extend(1.0 if loc == item else 0.0 for item in loc_values)
    if include_missing:
        values.extend(
            [
                1.0 if age_missing else 0.0,
                1.0 if _norm_text(row.get(sex_column, "unknown")) == "unknown" else 0.0,
                1.0 if _norm_text(row.get(loc_column, "unknown")) == "unknown" else 0.0,
            ]
        )
    return torch.tensor(values, dtype=torch.float32)


def _index_image_files(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not root.exists():
        return index
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            index[p.stem] = p
    return index


def _canonical_mask_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ["_segmentation", "_mask", "_lesion", "_gt", "_groundtruth"]:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_binary_mask(path: Path, size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    tensor = torch.from_numpy(np.array(mask, dtype="float32") / 255.0).unsqueeze(0)
    return (tensor > 0.5).float()


class HAM10000Dataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        transform: Any = None,
        label_column: str = "dx",
        image_id_column: str = "image_id",
        group_column: str = "lesion_id",
        patient_id_column: str = "patient_id",
        metadata_config: dict[str, Any] | None = None,
        mask_config: dict[str, Any] | None = None,
        input_size: int = 224,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        self.label_column = label_column
        self.image_id_column = image_id_column
        self.group_column = group_column
        self.patient_id_column = patient_id_column
        self.metadata_config = metadata_config or {}
        self.mask_config = mask_config or {}
        self.input_size = int(input_size)
        self.mask_index: dict[str, Path] = {}
        self._index_images()
        self._index_masks()

    def _index_images(self) -> None:
        self.image_index = _index_image_files(self.image_dir)

    def _index_masks(self) -> None:
        mask_dir = self.mask_config.get("mask_dir")
        if not mask_dir:
            return
        root = Path(mask_dir)
        if not root.exists():
            return
        for p in root.rglob("*"):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                self.mask_index[_canonical_mask_stem(p)] = p

    def __len__(self) -> int:
        return len(self.df)

    def _find_image(self, image_id: str) -> Path:
        if image_id in self.image_index:
            return self.image_index[image_id]
        for ext in [".jpg", ".jpeg", ".png"]:
            p = self.image_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found for image_id={image_id} under {self.image_dir}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image_id = str(row[self.image_id_column])
        label_name = str(row[self.label_column])
        path = self._find_image(image_id)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        lesion_id = str(row.get(self.group_column, ""))
        patient_id = row.get(self.patient_id_column, None)
        patient_id = "" if pd.isna(patient_id) else str(patient_id)
        metadata = encode_metadata(row, self.metadata_config)
        mask = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
        has_mask = torch.tensor(0.0, dtype=torch.float32)
        mask_path = self.mask_index.get(image_id)
        if mask_path is not None:
            mask = _load_binary_mask(mask_path, self.input_size)
            has_mask = torch.tensor(1.0, dtype=torch.float32)
        return {
            "image": image,
            "label": int(LABEL_MAP[label_name]),
            "image_id": image_id,
            "lesion_id": lesion_id,
            "patient_id": patient_id,
            "path": str(path),
            "metadata": metadata,
            "mask": mask,
            "has_mask": has_mask,
        }


class ExternalEndpointDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        transform: Any = None,
        metadata_config: dict[str, Any] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        self.metadata_config = metadata_config or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        path: Path | None = None
        raw_path = row.get("path", None) if "path" in row else None
        if raw_path is not None and pd.notna(raw_path) and str(raw_path).strip():
            candidate_path = Path(str(raw_path))
            if candidate_path.is_file():
                path = candidate_path
        if path is None:
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = self.image_dir / f"{image_id}{ext}"
                if candidate.is_file():
                    path = candidate
                    break
        if path is None:
            raise FileNotFoundError(f"External image not found: {image_id}")
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row.get("melanoma", row.get("label", 0)))
        return {
            "image": image,
            "label": label,
            "image_id": image_id,
            "lesion_id": str(row.get("lesion_id", "")),
            "patient_id": str(row.get("patient_id", "")),
            "path": str(path),
            "metadata": encode_metadata(row, self.metadata_config),
        }


class SkinLesionMaskDataset(Dataset):
    """Auxiliary image-mask dataset for lesion prior supervision.

    Supplies image-mask pairs to the auxiliary LPAH branch during training.
    The classification branch uses images and configured metadata at inference.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform: Any = None,
        input_size: int = 224,
        metadata_dim_value: int = 0,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.input_size = int(input_size)
        self.metadata_dim_value = int(metadata_dim_value)
        images = _index_image_files(self.image_dir)
        masks: dict[str, Path] = {}
        for p in self.mask_dir.rglob("*") if self.mask_dir.exists() else []:
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                masks[_canonical_mask_stem(p)] = p
        self.rows = [(key, img, masks[key]) for key, img in images.items() if key in masks]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id, image_path, mask_path = self.rows[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        mask = _load_binary_mask(mask_path, self.input_size)
        return {
            "image": image,
            "label": -1,
            "image_id": image_id,
            "lesion_id": "",
            "patient_id": "",
            "path": str(image_path),
            "metadata": torch.zeros(self.metadata_dim_value, dtype=torch.float32),
            "mask": mask,
            "has_mask": torch.tensor(1.0, dtype=torch.float32),
        }
