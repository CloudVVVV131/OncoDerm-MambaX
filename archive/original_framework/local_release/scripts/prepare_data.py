from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import uuid
from zipfile import ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image

from lesionmamba.data.datasets import LABEL_MAP


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_HAM_ROWS = 10_015
EXPECTED_EXTERNAL_ROWS = 33_126
EXPECTED_EXTERNAL_POSITIVES = 584
EXPECTED_MASK_PAIRS = 200
EXPECTED_MASK_VALIDATION = 100
EXPECTED_MASK_TRAINING = 100

PUBLIC_SOURCES = {
    "ham10000": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/kmader/skin-cancer-mnist-ham10000",
        "bytes": 5_582_914_511,
        "md5": "fd84b842e863f3c82f2f4231da626014",
    },
    "isic2020_512": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/cdeotte/jpeg-melanoma-512x512",
        "bytes": 2_825_694_558,
        "md5": "cb02bdebe2901b7ae8c847250b277ea8",
    },
    "isic2018_validation_input": {
        "url": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip",
        "bytes": 239_231_159,
        "sha256": "0ea920fcfe512d12a6e620b50b50233c059f67b10146e1479c82be58ff15a797",
    },
    "isic2018_validation_masks": {
        "url": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
        "bytes": 759_706,
        "sha256": "f6911e9c0a64e6d687dd3ca466ca927dd5e82145cb2163b7a1e5b37d7a716285",
    },
    "isic2018_training_masks": {
        "url": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip",
        "bytes": 27_402_895,
        "md5": "ee5e5db7771d48fa2613abc7cb5c24e2",
    },
    "isic2018_training_input_remote": {
        "url": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip",
        "selection": "HTTP range extraction of the 100 lexicographically first eligible image IDs",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hashes(path: Path) -> dict[str, str | int]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            size += len(block)
            md5.update(block)
            sha256.update(block)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def validate_archive(path: Path, source_key: str) -> dict[str, str | int]:
    if not path.is_file():
        raise FileNotFoundError(f"Required public archive is missing: {path}")
    expected = PUBLIC_SOURCES[source_key]
    observed = file_hashes(path)
    if observed["bytes"] != expected["bytes"]:
        raise ValueError(
            f"{source_key} size mismatch: expected={expected['bytes']} observed={observed['bytes']}"
        )
    for algorithm in ("md5", "sha256"):
        if algorithm in expected and observed[algorithm] != expected[algorithm]:
            raise ValueError(
                f"{source_key} {algorithm} mismatch: expected={expected[algorithm]} "
                f"observed={observed[algorithm]}"
            )
    return {**observed, "url": str(expected["url"]), "archive": str(path)}


def member_path(info: ZipInfo) -> PurePosixPath:
    path = PurePosixPath(info.filename.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP member path: {info.filename}")
    return path


def image_members(
    archive: ZipFile, wanted_ids: set[str] | None = None
) -> dict[str, ZipInfo]:
    def member_sha256(info: ZipInfo) -> str:
        digest = hashlib.sha256()
        with archive.open(info, "r") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    index: dict[str, ZipInfo] = {}
    digests: dict[tuple[str, int, int], str] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = member_path(info)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem
        if wanted_ids is not None and stem not in wanted_ids:
            continue
        if stem in index:
            previous = index[stem]
            previous_path = member_path(previous)
            same_encoding = previous_path.suffix.lower() == path.suffix.lower()
            same_zip_fingerprint = (
                previous.file_size == info.file_size and previous.CRC == info.CRC
            )
            if same_encoding and same_zip_fingerprint:
                previous_key = (previous.filename, previous.file_size, previous.CRC)
                current_key = (info.filename, info.file_size, info.CRC)
                previous_digest = digests.setdefault(
                    previous_key, member_sha256(previous)
                )
                current_digest = digests.setdefault(current_key, member_sha256(info))
                if previous_digest == current_digest:
                    # Byte-identical aliases can have case-variant parent paths.
                    # Choose a deterministic representative after hash verification.
                    index[stem] = min(
                        (previous, info), key=lambda member: member.filename
                    )
                    continue
            raise ValueError(
                f"Conflicting duplicate image stem in archive: {stem} "
                f"({previous.filename}, {info.filename})"
            )
        index[stem] = info
    return index


def find_unique_basename(archive: ZipFile, basename: str) -> ZipInfo:
    matches = [
        info
        for info in archive.infolist()
        if not info.is_dir() and member_path(info).name.lower() == basename.lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {basename} in archive, found {len(matches)}"
        )
    return matches[0]


def canonical_mask_stem(name: str | Path) -> str:
    stem = Path(str(name)).stem
    for suffix in (
        "_segmentation",
        "-segmentation",
        "_ground_truth",
        "-ground-truth",
        "_groundtruth",
        "-groundtruth",
        "_mask",
        "-mask",
        "_lesion",
        "_gt",
        "-gt",
    ):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def mask_members(archive: ZipFile) -> dict[str, ZipInfo]:
    index: dict[str, ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = member_path(info)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = canonical_mask_stem(path.name)
        if stem in index:
            raise ValueError(f"Duplicate mask stem in archive: {stem}")
        index[stem] = info
    return index


def copy_member(archive: ZipFile, info: ZipInfo, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info, "r") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if destination.stat().st_size != info.file_size:
        raise IOError(
            f"Extracted member size mismatch for {info.filename}: "
            f"expected={info.file_size} observed={destination.stat().st_size}"
        )


def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_ham10000(archive_path: Path, destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    image_dir = destination / "images"
    image_dir.mkdir()
    with ZipFile(archive_path) as archive:
        metadata_info = find_unique_basename(archive, "HAM10000_metadata.csv")
        metadata_bytes = archive.read(metadata_info)
        metadata = pd.read_csv(io.BytesIO(metadata_bytes))
        required_columns = {"image_id", "lesion_id", "dx"}
        if len(metadata) != EXPECTED_HAM_ROWS or not required_columns.issubset(
            metadata.columns
        ):
            raise ValueError(
                f"Unexpected HAM10000 metadata: rows={len(metadata)} columns={list(metadata.columns)}"
            )
        ids = metadata["image_id"].astype(str)
        if ids.duplicated().any():
            raise ValueError("HAM10000 metadata contains duplicate image_id values")
        labels = set(metadata["dx"].astype(str))
        if labels != set(LABEL_MAP):
            raise ValueError(f"HAM10000 label set mismatch: {sorted(labels)}")
        images = image_members(archive, set(ids))
        missing = sorted(set(ids) - set(images))
        if missing or len(images) != EXPECTED_HAM_ROWS:
            raise ValueError(
                f"HAM10000 image alignment failed: found={len(images)} missing={missing[:10]}"
            )
        (destination / "HAM10000_metadata.csv").write_bytes(metadata_bytes)
        for image_id in sorted(images):
            info = images[image_id]
            suffix = member_path(info).suffix.lower()
            copy_member(archive, info, image_dir / f"{image_id}{suffix}")
    return {
        "rows": len(metadata),
        "images": len(images),
        "labels": {label: int((metadata["dx"] == label).sum()) for label in LABEL_MAP},
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }


def build_external(archive_path: Path, destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    image_dir = destination / "images"
    image_dir.mkdir()
    with ZipFile(archive_path) as archive:
        csv_info = find_unique_basename(archive, "train.csv")
        source_csv = archive.read(csv_info)
        frame = pd.read_csv(io.BytesIO(source_csv))
        id_column = "image_name" if "image_name" in frame.columns else "image_id"
        target_column = "target" if "target" in frame.columns else "melanoma"
        if id_column not in frame.columns or target_column not in frame.columns:
            raise ValueError(
                f"ISIC2020 train.csv lacks an ID/target column: {list(frame.columns)}"
            )
        ids = frame[id_column].astype(str).map(lambda value: Path(value).stem)
        targets = pd.to_numeric(frame[target_column], errors="raise").astype(int)
        if (
            len(frame) != EXPECTED_EXTERNAL_ROWS
            or int(targets.sum()) != EXPECTED_EXTERNAL_POSITIVES
        ):
            raise ValueError(
                f"Unexpected ISIC2020 labels: rows={len(frame)} positives={int(targets.sum())}"
            )
        if ids.duplicated().any() or set(targets.unique()) != {0, 1}:
            raise ValueError("ISIC2020 IDs are duplicated or labels are not binary")
        images = image_members(archive, set(ids))
        missing = sorted(set(ids) - set(images))
        if missing or len(images) != EXPECTED_EXTERNAL_ROWS:
            raise ValueError(
                f"ISIC2020 image alignment failed: found={len(images)} missing={missing[:10]}"
            )
        labels = pd.DataFrame({"image_id": ids, "melanoma": targets})
        labels.to_csv(destination / "labels_harmonized.csv", index=False)
        for image_id in sorted(images):
            info = images[image_id]
            suffix = member_path(info).suffix.lower()
            copy_member(archive, info, image_dir / f"{image_id}{suffix}")
    return {
        "rows": len(labels),
        "images": len(images),
        "melanoma_positive": int(labels["melanoma"].sum()),
        "selection": "labelled train partition only",
    }


def remote_training_image_members(
    remote_archive: object, wanted_ids: set[str]
) -> dict[str, object]:
    index: dict[str, object] = {}
    for info in remote_archive.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename.replace("\\", "/"))
        if path.suffix.lower() not in IMAGE_SUFFIXES or path.stem not in wanted_ids:
            continue
        if path.stem in index:
            raise ValueError(
                f"Duplicate training image stem in remote archive: {path.stem}"
            )
        index[path.stem] = info
    return index


def copy_remote_member(remote_archive: object, info: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with remote_archive.open(info) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    expected_size = int(getattr(info, "file_size"))
    if destination.stat().st_size != expected_size:
        raise IOError(
            f"Remote ZIP member size mismatch for {getattr(info, 'filename')}"
        )


def build_masks(
    validation_images_zip: Path,
    validation_masks_zip: Path,
    training_masks_zip: Path,
    training_images_url: str,
    destination: Path,
) -> dict[str, object]:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise RuntimeError(
            "remotezip==0.12.3 is required for selective ISIC2018 extraction"
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    image_dir = destination / "images"
    mask_dir = destination / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    cohort_rows: list[dict[str, object]] = []

    with (
        ZipFile(validation_images_zip) as image_archive,
        ZipFile(validation_masks_zip) as mask_archive,
    ):
        images = image_members(image_archive)
        masks = mask_members(mask_archive)
        validation_ids = sorted(set(images) & set(masks))
        if len(validation_ids) != EXPECTED_MASK_VALIDATION:
            raise ValueError(
                f"Expected {EXPECTED_MASK_VALIDATION} ISIC2018 validation pairs, found {len(validation_ids)}"
            )
        for image_id in validation_ids:
            image_info = images[image_id]
            mask_info = masks[image_id]
            image_name = f"{image_id}{member_path(image_info).suffix.lower()}"
            mask_name = (
                f"{image_id}_segmentation{member_path(mask_info).suffix.lower()}"
            )
            copy_member(image_archive, image_info, image_dir / image_name)
            copy_member(mask_archive, mask_info, mask_dir / mask_name)
            cohort_rows.append(
                {
                    "image_id": image_id,
                    "source_partition": "ISIC2018_validation",
                    "image_file": image_name,
                    "mask_file": mask_name,
                    "selection_rank": None,
                }
            )

    with ZipFile(training_masks_zip) as mask_archive:
        training_masks = mask_members(mask_archive)
        eligible_ids = sorted(set(training_masks) - set(validation_ids))
        training_ids = eligible_ids[:EXPECTED_MASK_TRAINING]
        if len(training_ids) != EXPECTED_MASK_TRAINING:
            raise ValueError(
                f"Insufficient unique ISIC2018 training masks: {len(training_ids)}"
            )
        with RemoteZip(
            training_images_url, initial_buffer_size=2 * 1024 * 1024, timeout=180
        ) as remote:
            remote_images = remote_training_image_members(remote, set(training_ids))
            missing = sorted(set(training_ids) - set(remote_images))
            if missing:
                raise ValueError(
                    f"Training images missing from official remote ZIP: {missing[:10]}"
                )
            for rank, image_id in enumerate(training_ids, start=1):
                image_info = remote_images[image_id]
                mask_info = training_masks[image_id]
                image_suffix = PurePosixPath(image_info.filename).suffix.lower()
                mask_suffix = member_path(mask_info).suffix.lower()
                image_name = f"{image_id}{image_suffix}"
                mask_name = f"{image_id}_segmentation{mask_suffix}"
                copy_remote_member(remote, image_info, image_dir / image_name)
                copy_member(mask_archive, mask_info, mask_dir / mask_name)
                cohort_rows.append(
                    {
                        "image_id": image_id,
                        "source_partition": "ISIC2018_training_deterministic_subset",
                        "image_file": image_name,
                        "mask_file": mask_name,
                        "selection_rank": rank,
                    }
                )

    cohort = pd.DataFrame(cohort_rows)
    cohort.to_csv(destination / "mask200_cohort.csv", index=False)
    validate_mask_dimensions(destination, cohort)
    return {
        "pairs": len(cohort),
        "validation_pairs": int(
            (cohort["source_partition"] == "ISIC2018_validation").sum()
        ),
        "training_pairs": int(
            (
                cohort["source_partition"] == "ISIC2018_training_deterministic_subset"
            ).sum()
        ),
        "selection_rule": (
            "all 100 official validation pairs plus the lexicographically first 100 unique official "
            "training pairs; no duplication, synthetic masks, or external-test tuning"
        ),
        "selected_training_ids_sha256": hashlib.sha256(
            "\n".join(training_ids).encode("utf-8")
        ).hexdigest(),
    }


def index_local_images(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*") if root.exists() else []:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in index:
                raise ValueError(f"Duplicate local image stem {path.stem} under {root}")
            index[path.stem] = path
    return index


def index_local_masks(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*") if root.exists() else []:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            stem = canonical_mask_stem(path)
            if stem in index:
                raise ValueError(f"Duplicate local mask stem {stem} under {root}")
            index[stem] = path
    return index


def validate_mask_dimensions(mask_root: Path, cohort: pd.DataFrame) -> None:
    for row in cohort.itertuples(index=False):
        image_path = mask_root / "images" / str(row.image_file)
        mask_path = mask_root / "masks" / str(row.mask_file)
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(
                    f"Mask dimension mismatch for {row.image_id}: image={image.size} mask={mask.size}"
                )
            image.verify()
            mask.verify()


def validate_split_alignment(
    split_dir: Path, metadata: pd.DataFrame
) -> dict[str, object]:
    metadata_by_id = metadata.assign(
        image_id=metadata["image_id"].astype(str)
    ).set_index("image_id")
    report: dict[str, object] = {}
    for seed in (42, 2024):
        split_frames: dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            path = split_dir / f"ham10000_seed{seed}_{split}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Fixed split is missing: {path}")
            frame = pd.read_csv(path)
            if (
                "image_id" not in frame
                or frame["image_id"].astype(str).duplicated().any()
            ):
                raise ValueError(f"Invalid image IDs in {path}")
            split_frames[split] = frame.assign(image_id=frame["image_id"].astype(str))
        all_rows = pd.concat(split_frames.values(), ignore_index=True)
        ids = all_rows["image_id"]
        if ids.duplicated().any() or set(ids) != set(metadata_by_id.index):
            raise ValueError(
                f"Split seed {seed} does not partition all HAM10000 image IDs exactly once"
            )
        for column in ("dx", "lesion_id"):
            if column in all_rows.columns and column in metadata_by_id.columns:
                expected = metadata_by_id.loc[ids, column].astype(str).to_numpy()
                observed = all_rows[column].astype(str).to_numpy()
                if not (expected == observed).all():
                    raise ValueError(
                        f"Split seed {seed} has {column} values inconsistent with metadata"
                    )
        for group_column in ("lesion_id", "patient_id"):
            if group_column not in all_rows.columns:
                continue
            groups = {
                name: set(frame[group_column].dropna().astype(str))
                for name, frame in split_frames.items()
            }
            if (
                groups["train"] & groups["val"]
                or groups["train"] & groups["test"]
                or groups["val"] & groups["test"]
            ):
                raise ValueError(
                    f"Split seed {seed} leaks {group_column} across partitions"
                )
        report[str(seed)] = {name: len(frame) for name, frame in split_frames.items()}
    return report


def verify_dataset(raw_root: Path, split_dir: Path) -> dict[str, object]:
    ham_root = raw_root / "HAM10000"
    metadata_path = ham_root / "HAM10000_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = pd.read_csv(metadata_path)
    ham_ids = metadata["image_id"].astype(str)
    ham_images = index_local_images(ham_root / "images")
    if len(metadata) != EXPECTED_HAM_ROWS or len(ham_images) != EXPECTED_HAM_ROWS:
        raise ValueError(
            f"HAM10000 final count mismatch: rows={len(metadata)} images={len(ham_images)}"
        )
    if ham_ids.duplicated().any() or set(ham_ids) != set(ham_images):
        raise ValueError("HAM10000 final image/metadata alignment failed")

    external_root = raw_root / "ISIC_external"
    labels = pd.read_csv(external_root / "labels_harmonized.csv")
    external_ids = labels["image_id"].astype(str)
    external_images = index_local_images(external_root / "images")
    positives = int(pd.to_numeric(labels["melanoma"], errors="raise").sum())
    if (
        len(labels) != EXPECTED_EXTERNAL_ROWS
        or positives != EXPECTED_EXTERNAL_POSITIVES
    ):
        raise ValueError(
            f"ISIC2020 final label mismatch: rows={len(labels)} positives={positives}"
        )
    if external_ids.duplicated().any() or set(external_ids) != set(external_images):
        raise ValueError("ISIC2020 final image/label alignment failed")

    mask_root = raw_root / "ISIC_masks"
    cohort = pd.read_csv(mask_root / "mask200_cohort.csv")
    mask_images = index_local_images(mask_root / "images")
    masks = index_local_masks(mask_root / "masks")
    cohort_ids = cohort["image_id"].astype(str)
    if len(cohort) != EXPECTED_MASK_PAIRS or cohort_ids.duplicated().any():
        raise ValueError(
            f"Mask cohort must contain exactly {EXPECTED_MASK_PAIRS} unique rows"
        )
    if set(cohort_ids) != set(mask_images) or set(cohort_ids) != set(masks):
        raise ValueError("Mask cohort image/mask alignment failed")
    counts = cohort["source_partition"].value_counts().to_dict()
    if (
        counts.get("ISIC2018_validation") != EXPECTED_MASK_VALIDATION
        or counts.get("ISIC2018_training_deterministic_subset")
        != EXPECTED_MASK_TRAINING
    ):
        raise ValueError(f"Mask cohort source composition mismatch: {counts}")
    validate_mask_dimensions(mask_root, cohort)

    split_report = validate_split_alignment(split_dir, metadata)
    return {
        "status": "verified",
        "verified_at_utc": utc_now(),
        "ham10000": {"rows": len(metadata), "images": len(ham_images)},
        "isic2020_external": {
            "rows": len(labels),
            "images": len(external_images),
            "melanoma_positive": positives,
        },
        "isic2018_masks": {
            "pairs": len(cohort),
            "validation_pairs": counts["ISIC2018_validation"],
            "training_pairs": counts["ISIC2018_training_deterministic_subset"],
        },
        "fixed_splits": split_report,
    }


def replace_dataset_dirs(
    raw_root: Path, stage_root: Path, names: Iterable[str]
) -> None:
    raw_root = raw_root.resolve()
    names = list(names)
    targets = [(raw_root / name).resolve() for name in names]
    if any(target.parent != raw_root or target.exists() for target in targets):
        raise ValueError("Data preparation requires new dataset directories inside the raw-data root.")
    transactions: list[Path] = []
    try:
        for name in names:
            target = (raw_root / name).resolve()
            if target.parent != raw_root:
                raise ValueError(
                    f"Dataset target must be inside the raw-data root: {target}"
                )
            staged = stage_root / name
            staged.rename(target)
            transactions.append(target)
    except Exception:
        for target in reversed(transactions):
            if target.exists():
                shutil.rmtree(target)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--manifests", type=Path, default=Path(__file__).resolve().parents[1] / "data/manifests")
    parser.add_argument("--ham-zip")
    parser.add_argument("--external-zip")
    parser.add_argument("--validation-images-zip")
    parser.add_argument("--validation-masks-zip")
    parser.add_argument("--training-masks-zip")
    parser.add_argument(
        "--training-images-url",
        default=PUBLIC_SOURCES["isic2018_training_input_remote"]["url"],
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    split_dir = Path(args.split_dir).resolve()
    if args.verify_only:
        report = verify_dataset(raw_root, split_dir)
        manifest_path = raw_root / "bioengineering_data_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "verified":
                raise ValueError(
                    f"Data manifest is not verified: {manifest.get('status')}"
                )
        print(json.dumps(report, ensure_ascii=False))
        return

    for name in ("HAM10000", "ISIC_external", "ISIC_masks"):
        if (raw_root / name).exists():
            raise FileExistsError(f"Dataset directory already exists: {raw_root / name}. Use --verify-only for prepared data.")
    if split_dir.exists():
        raise FileExistsError(f"Split directory already exists: {split_dir}. Use a new split directory.")

    required_args = {
        "ham10000": args.ham_zip,
        "isic2020_512": args.external_zip,
        "isic2018_validation_input": args.validation_images_zip,
        "isic2018_validation_masks": args.validation_masks_zip,
        "isic2018_training_masks": args.training_masks_zip,
    }
    missing = [key for key, value in required_args.items() if not value]
    if missing:
        raise SystemExit(
            f"Archive arguments are required unless --verify-only is used: {missing}"
        )

    archive_paths = {key: Path(value).resolve() for key, value in required_args.items()}
    archive_provenance = {
        key: validate_archive(path, key) for key, path in archive_paths.items()
    }
    raw_root.mkdir(parents=True, exist_ok=True)
    stage_root = raw_root / f".bioengineering_stage_{uuid.uuid4().hex}"
    stage_root.mkdir()
    try:
        build_report = {
            "ham10000": build_ham10000(
                archive_paths["ham10000"], stage_root / "HAM10000"
            ),
            "isic2020_external": build_external(
                archive_paths["isic2020_512"], stage_root / "ISIC_external"
            ),
            "isic2018_masks": build_masks(
                archive_paths["isic2018_validation_input"],
                archive_paths["isic2018_validation_masks"],
                archive_paths["isic2018_training_masks"],
                args.training_images_url,
                stage_root / "ISIC_masks",
            ),
        }
        from restore_splits import restore_fixed_splits

        staged_splits = stage_root / "fixed_splits"
        restore_fixed_splits(stage_root / "HAM10000/HAM10000_metadata.csv", args.manifests, staged_splits)
        staged_verification = verify_dataset(stage_root, staged_splits)
        split_dir.parent.mkdir(parents=True, exist_ok=True)
        staged_splits.rename(split_dir)
        replace_dataset_dirs(
            raw_root, stage_root, ["HAM10000", "ISIC_external", "ISIC_masks"]
        )
        final_verification = verify_dataset(raw_root, split_dir)
        manifest = {
            "schema_version": 1,
            "status": "verified",
            "created_at_utc": utc_now(),
            "public_data_prepared_from_archives": True,
            "archives": archive_provenance,
            "remote_training_images": {
                **PUBLIC_SOURCES["isic2018_training_input_remote"],
                "full_archive_downloaded": False,
            },
            "build": build_report,
            "staged_verification": staged_verification,
            "final_verification": final_verification,
        }
        manifest_tmp = raw_root / ".bioengineering_data_manifest.json.tmp"
        write_json(manifest, manifest_tmp)
        manifest_tmp.replace(raw_root / "bioengineering_data_manifest.json")
        print(json.dumps(final_verification, ensure_ascii=False))
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)


if __name__ == "__main__":
    main()
