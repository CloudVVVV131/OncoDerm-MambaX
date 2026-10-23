from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader, Subset

from lesionmamba.data.datasets import HAM10000Dataset
from lesionmamba.data.splits import make_splits
from lesionmamba.data.transforms import build_transforms
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import project_root, write_json


def write_dataloader_sanity(config: dict, report: dict) -> None:
    data_cfg = config["data"]
    train_csv = Path(report["paths"]["train"])
    image_dir = Path(data_cfg["image_dir"])
    out_path = Path(data_cfg.get("split_dir", "splits")) / f"dataloader_sanity_seed{config.get('project', {}).get('seed', 42)}.json"
    if not train_csv.exists() or not image_dir.exists():
        write_json(
            {
                "status": "skipped",
                "reason": "train split or image directory is missing",
                "train_csv": str(train_csv),
                "image_dir": str(image_dir),
            },
            out_path,
        )
        return
    try:
        dataset = HAM10000Dataset(
            train_csv,
            image_dir,
            transform=build_transforms(config, "eval"),
            label_column=data_cfg.get("label_column", "dx"),
            image_id_column=data_cfg.get("image_id_column", "image_id"),
            group_column=data_cfg.get("group_column", "lesion_id"),
            patient_id_column=data_cfg.get("patient_id_column", "patient_id"),
            metadata_config=data_cfg.get("metadata", {}),
            mask_config=data_cfg.get("mask_aux", {}),
            input_size=int(data_cfg.get("input_size", 224)),
        )
        subset = Subset(dataset, list(range(min(4, len(dataset)))))
        loader = DataLoader(subset, batch_size=max(1, len(subset)), shuffle=False, num_workers=0)
        batch = next(iter(loader)) if len(subset) else None
        write_json(
            {
                "status": "ok" if batch is not None else "empty_split",
                "batch_shape": list(batch["image"].shape) if batch is not None else None,
                "labels": batch["label"].tolist() if batch is not None else [],
                "metadata_shape": list(batch["metadata"].shape) if batch is not None and "metadata" in batch else None,
                "mask_shape": list(batch["mask"].shape) if batch is not None and "mask" in batch else None,
                "has_mask": batch["has_mask"].tolist() if batch is not None and "has_mask" in batch else [],
                "image_ids": list(batch["image_id"]) if batch is not None else [],
                "paths": list(batch["path"]) if batch is not None else [],
                "dataset_size": len(dataset),
            },
            out_path,
        )
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/defaults.yaml")
    args = parser.parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    report = make_splits(config)
    write_dataloader_sanity(config, report)
    if not report["leakage"]["ok"]:
        raise SystemExit(f"Leakage detected: {report['leakage']['overlaps']}")
    print(f"Created splits with group_col={report['group_col']}")


if __name__ == "__main__":
    main()
