"""Restore the frozen HAM10000 row order by joining released image-ID manifests."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile
import shutil


def restore_fixed_splits(metadata_path: Path, manifest_dir: Path, output_dir: Path) -> None:
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not {"image_id", "dx", "lesion_id"}.issubset(fields):
        raise ValueError("HAM10000 metadata must include image_id, dx, and lesion_id.")
    by_id = {row["image_id"]: row for row in rows}
    if len(by_id) != len(rows) or "" in by_id:
        raise ValueError("Metadata image IDs must be nonempty and unique.")
    outputs = {}
    for seed in (42, 2024):
        seen_ids, seen_groups = set(), set()
        for split in ("train", "val", "test"):
            name = f"ham10000_seed{seed}_{split}"
            with (manifest_dir / f"{name}_ids.csv").open(encoding="utf-8-sig") as handle:
                ids = [row["image_id"] for row in csv.DictReader(handle)]
            if not ids or len(ids) != len(set(ids)) or seen_ids.intersection(ids):
                raise ValueError(f"Invalid or overlapping image IDs in {name}.")
            if set(ids) - by_id.keys():
                raise ValueError(f"Metadata is missing released image IDs for {name}.")
            selected = [by_id[image_id] for image_id in ids]
            groups = {row["lesion_id"] for row in selected}
            if "" in groups or seen_groups.intersection(groups):
                raise ValueError(f"Missing or overlapping lesion IDs in {name}.")
            seen_ids.update(ids)
            seen_groups.update(groups)
            outputs[f"{name}.csv"] = selected
        if seen_ids != set(by_id):
            raise ValueError(f"The seed-{seed} manifests do not partition this metadata exactly.")
    if output_dir.exists():
        raise FileExistsError(f"Split directory already exists; select a new output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".splits-", dir=output_dir.parent))
    try:
        for name, selected in outputs.items():
            with (stage / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(selected)
        stage.rename(output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=Path(__file__).resolve().parents[1] / "data/manifests")
    parser.add_argument("--output", type=Path, default=Path("splits"))
    args = parser.parse_args()
    restore_fixed_splits(args.metadata, args.manifests, args.output)
    print(f"Restored six fixed split files to {args.output}")


if __name__ == "__main__":
    main()
