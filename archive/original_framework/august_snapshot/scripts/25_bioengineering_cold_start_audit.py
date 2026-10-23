from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


REQUIRED_DATA_COUNTS = {
    "ham10000_rows": 10_015,
    "ham10000_images": 10_015,
    "external_rows": 33_126,
    "external_images": 33_126,
    "external_positive": 584,
    "mask_pairs": 200,
    "mask_validation": 100,
    "mask_training": 100,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe path in cold-start manifest: {value}")
    return Path(*pure.parts)


def read_prediction_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def read_image_ids(path: Path) -> list[str]:
    fields, rows = read_prediction_rows(path)
    if "image_id" not in fields:
        raise ValueError(f"image_id column is missing from {path}")
    return [str(row["image_id"]) for row in rows]


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []
        self.errors: list[str] = []

    def check(self, name: str, passed: bool, detail: object = None) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            self.errors.append(name)


def audit_artifacts(root: Path, manifest: dict[str, object], audit: Audit) -> None:
    files = manifest.get("files")
    audit.check(
        "manifest_has_file_inventory",
        isinstance(files, list) and bool(files),
        len(files or []),
    )
    if not isinstance(files, list):
        return

    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            audit.check("manifest_file_entry_is_object", False, entry)
            continue
        relative = str(entry.get("path", ""))
        try:
            relative_path = safe_relative_path(relative)
        except ValueError as exc:
            audit.check(f"safe_manifest_path:{relative}", False, str(exc))
            continue
        if relative in seen:
            audit.check(f"manifest_path_unique:{relative}", False, "duplicate")
            continue
        seen.add(relative)
        path = root / relative_path
        if not path.is_file():
            audit.check(f"file_exists:{relative}", False, "missing")
            continue
        expected_bytes = int(entry.get("bytes", -1))
        observed_bytes = path.stat().st_size
        audit.check(
            f"file_size:{relative}",
            observed_bytes == expected_bytes,
            {"expected": expected_bytes, "observed": observed_bytes},
        )
        expected_hash = str(entry.get("sha256", "")).lower()
        observed_hash = sha256(path)
        audit.check(
            f"file_sha256:{relative}",
            observed_hash == expected_hash,
            {"expected": expected_hash, "observed": observed_hash},
        )

    expected_runs = manifest.get("expected_historical_runs", [])
    audit.check(
        "historical_run_inventory_present",
        isinstance(expected_runs, list) and len(expected_runs) == 6,
    )
    for item in expected_runs if isinstance(expected_runs, list) else []:
        run_id = str(item.get("run_id", ""))
        checkpoint = (
            root / "outputs" / "runs" / run_id / "checkpoints" / "best_val_macro_f1.pth"
        )
        audit.check(
            f"best_checkpoint_present:{run_id}", checkpoint.is_file(), str(checkpoint)
        )
        last_checkpoint = checkpoint.parent / "last.pth"
        audit.check(
            f"duplicate_last_checkpoint_excluded:{run_id}",
            not last_checkpoint.exists(),
            str(last_checkpoint),
        )
        config = checkpoint.parents[1] / "config.yaml"
        environment = checkpoint.parents[1] / "environment.json"
        audit.check(f"run_config_present:{run_id}", config.is_file(), str(config))
        audit.check(
            f"run_environment_present:{run_id}", environment.is_file(), str(environment)
        )
        for split in ("val", "test", "external"):
            prediction = (
                root / "outputs" / "predictions" / f"{run_id}_{split}_predictions.csv"
            )
            audit.check(
                f"prediction_present:{run_id}:{split}",
                prediction.is_file(),
                str(prediction),
            )

    required_splits = manifest.get("required_fixed_splits", [])
    audit.check(
        "fixed_split_inventory_present",
        isinstance(required_splits, list) and len(required_splits) >= 6,
    )
    for name in required_splits if isinstance(required_splits, list) else []:
        path = root / "splits" / str(name)
        audit.check(f"fixed_split_present:{name}", path.is_file(), str(path))

    semantic_expectations = manifest.get("semantic_expectations", {})
    audit_prediction_alignment(
        root,
        expected_runs if isinstance(expected_runs, list) else [],
        semantic_expectations if isinstance(semantic_expectations, dict) else {},
        audit,
    )

    public_policy = manifest.get("public_data_policy", {})
    audit.check(
        "public_data_excluded_from_upload_bundle",
        isinstance(public_policy, dict)
        and public_policy.get("included_in_upload_bundle") is False,
        public_policy,
    )


def audit_prediction_alignment(
    root: Path,
    expected_runs: list[dict[str, object]],
    semantic_expectations: dict[str, object],
    audit: Audit,
) -> None:
    default_split_rows = {
        "42": {"train": 6_982, "val": 1_513, "test": 1_520},
        "2024": {"train": 7_013, "val": 1_515, "test": 1_487},
    }
    configured_split_rows = semantic_expectations.get("split_rows", default_split_rows)
    split_expectations = {
        (int(seed), split): int(rows)
        for seed, partitions in configured_split_rows.items()
        for split, rows in partitions.items()
    }
    ham_total = int(semantic_expectations.get("ham_total_rows", 10_015))
    external_rows = int(semantic_expectations.get("external_rows", 33_126))
    external_positives = int(semantic_expectations.get("external_positive", 584))
    split_ids: dict[tuple[int, str], list[str]] = {}
    for key, expected_rows in split_expectations.items():
        seed, split = key
        path = root / "splits" / f"ham10000_seed{seed}_{split}.csv"
        if not path.is_file():
            continue
        try:
            ids = read_image_ids(path)
            split_ids[key] = ids
            audit.check(
                f"fixed_split_row_count:seed{seed}:{split}",
                len(ids) == expected_rows,
                {"expected": expected_rows, "observed": len(ids)},
            )
            audit.check(
                f"fixed_split_ids_unique:seed{seed}:{split}",
                len(ids) == len(set(ids)),
                len(ids),
            )
        except Exception as exc:
            audit.check(f"fixed_split_readable:seed{seed}:{split}", False, str(exc))
    for seed in (42, 2024):
        partitions = [
            split_ids.get((seed, split), []) for split in ("train", "val", "test")
        ]
        combined = [image_id for partition in partitions for image_id in partition]
        audit.check(
            f"fixed_split_exact_partition:seed{seed}",
            len(combined) == ham_total and len(set(combined)) == ham_total,
            {"rows": len(combined), "unique_ids": len(set(combined))},
        )

    expected_internal = {
        "val": set(split_ids.get((42, "val"), [])),
        "test": set(split_ids.get((42, "test"), [])),
    }
    external_reference_ids: set[str] | None = None
    external_reference_labels: dict[str, int] | None = None
    for item in expected_runs:
        run_id = str(item.get("run_id", ""))
        for split in ("val", "test", "external"):
            path = (
                root / "outputs" / "predictions" / f"{run_id}_{split}_predictions.csv"
            )
            if not path.is_file():
                continue
            try:
                fields, rows = read_prediction_rows(path)
                required = {
                    "image_id",
                    "prob_mel",
                    "label" if split == "external" else "true_label",
                }
                audit.check(
                    f"prediction_columns:{run_id}:{split}",
                    required.issubset(fields),
                    {"required": sorted(required), "observed": fields},
                )
                ids = [str(row.get("image_id", "")) for row in rows]
                audit.check(
                    f"prediction_ids_unique:{run_id}:{split}",
                    bool(ids) and len(ids) == len(set(ids)) and "" not in ids,
                    {"rows": len(ids), "unique_ids": len(set(ids))},
                )
                scores_ok = True
                for row in rows:
                    try:
                        score = float(row.get("prob_mel", ""))
                        scores_ok = scores_ok and 0.0 <= score <= 1.0
                    except (TypeError, ValueError):
                        scores_ok = False
                        break
                audit.check(
                    f"prediction_scores_valid:{run_id}:{split}", scores_ok, len(rows)
                )
                if split in expected_internal:
                    audit.check(
                        f"prediction_matches_fixed_split:{run_id}:{split}",
                        set(ids) == expected_internal[split]
                        and len(ids) == len(expected_internal[split]),
                        {
                            "prediction_rows": len(ids),
                            "expected_rows": len(expected_internal[split]),
                        },
                    )
                else:
                    labels = {str(row["image_id"]): int(row["label"]) for row in rows}
                    audit.check(
                        f"external_prediction_cohort_size:{run_id}",
                        len(ids) == external_rows,
                        len(ids),
                    )
                    audit.check(
                        f"external_prediction_positive_count:{run_id}",
                        sum(labels.values()) == external_positives,
                        sum(labels.values()),
                    )
                    if external_reference_ids is None:
                        external_reference_ids = set(ids)
                        external_reference_labels = labels
                    else:
                        audit.check(
                            f"external_prediction_ids_consistent:{run_id}",
                            set(ids) == external_reference_ids,
                            len(ids),
                        )
                        audit.check(
                            f"external_prediction_labels_consistent:{run_id}",
                            labels == external_reference_labels,
                            len(labels),
                        )
            except Exception as exc:
                audit.check(f"prediction_semantics:{run_id}:{split}", False, str(exc))
    data_manifest_exists = (
        root / "data" / "raw" / "bioengineering_data_manifest.json"
    ).is_file()
    raw_files = [path for path in (root / "data" / "raw").rglob("*") if path.is_file()]
    audit.check(
        "upload_bundle_contains_no_public_raw_files",
        data_manifest_exists or not raw_files,
        {
            "cloud_data_already_prepared": data_manifest_exists,
            "unexpected_before_data_prep": [str(path) for path in raw_files[:10]]
            if not data_manifest_exists
            else [],
        },
    )


def audit_data(root: Path, audit: Audit) -> None:
    manifest_path = root / "data" / "raw" / "bioengineering_data_manifest.json"
    if not manifest_path.is_file():
        audit.check("cloud_data_manifest_present", False, str(manifest_path))
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        audit.check("cloud_data_manifest_valid_json", False, str(exc))
        return
    audit.check("cloud_data_manifest_valid_json", True)
    audit.check(
        "cloud_data_manifest_verified",
        manifest.get("status") == "verified",
        manifest.get("status"),
    )
    final = manifest.get("final_verification", {})
    ham = final.get("ham10000", {}) if isinstance(final, dict) else {}
    external = final.get("isic2020_external", {}) if isinstance(final, dict) else {}
    masks = final.get("isic2018_masks", {}) if isinstance(final, dict) else {}
    checks = {
        "ham10000_rows": ham.get("rows"),
        "ham10000_images": ham.get("images"),
        "external_rows": external.get("rows"),
        "external_images": external.get("images"),
        "external_positive": external.get("melanoma_positive"),
        "mask_pairs": masks.get("pairs"),
        "mask_validation": masks.get("validation_pairs"),
        "mask_training": masks.get("training_pairs"),
    }
    for name, expected in REQUIRED_DATA_COUNTS.items():
        audit.check(
            f"cloud_data_count:{name}",
            checks.get(name) == expected,
            {"expected": expected, "observed": checks.get(name)},
        )
    split_report = final.get("fixed_splits", {}) if isinstance(final, dict) else {}
    audit.check(
        "cloud_data_fixed_splits_verified",
        isinstance(split_report, dict) and set(split_report) == {"42", "2024"},
        split_report,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--artifacts-only", action="store_true")
    mode.add_argument("--data-only", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest_path = root / "COLD_START_ARTIFACT_MANIFEST.json"
    audit = Audit()
    manifest: dict[str, object] = {}
    if not args.data_only:
        if not manifest_path.is_file():
            audit.check("cold_start_manifest_present", False, str(manifest_path))
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                audit.check("cold_start_manifest_valid_json", True)
                audit.check(
                    "cold_start_manifest_schema",
                    manifest.get("schema_version") == 1,
                    manifest.get("schema_version"),
                )
                audit_artifacts(root, manifest, audit)
            except Exception as exc:
                audit.check("cold_start_manifest_valid_json", False, str(exc))
    if not args.artifacts_only:
        audit_data(root, audit)

    payload = {
        "schema_version": 1,
        "status": "passed" if not audit.errors else "failed",
        "mode": "artifacts"
        if args.artifacts_only
        else "data"
        if args.data_only
        else "full",
        "project_root": str(root),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": audit.checks,
        "errors": audit.errors,
    }
    output = (
        Path(args.output)
        if args.output
        else root / "results" / "bioengineering_cold_start_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"cold_start_audit status={payload['status']} mode={payload['mode']} "
        f"checks={len(audit.checks)} errors={len(audit.errors)} output={output}"
    )
    if audit.errors:
        for error in audit.errors[:30]:
            print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
