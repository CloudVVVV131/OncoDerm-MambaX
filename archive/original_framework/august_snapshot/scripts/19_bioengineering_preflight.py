from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from lesionmamba.analysis.bioengineering_supplement import (
    DIRECT_MODEL_SPECS,
    audit_ham_split_files,
    dermamamba_protocol_ok,
    discover_run,
    file_sha256,
)
from lesionmamba.utils.config import load_yaml
from lesionmamba.data.datasets import SkinLesionMaskDataset
from lesionmamba.utils.io import ensure_dir, write_json


def github_head(owner: str, repository: str) -> dict[str, str | None]:
    url = f"https://api.github.com/repos/{owner}/{repository}/commits/main"
    try:
        request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "OncoDerm-audit"})
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"url": url, "sha": payload.get("sha"), "error": None}
    except Exception as exc:
        return {"url": url, "sha": None, "error": str(exc)}


def image_stems(root: Path) -> tuple[set[str], list[str]]:
    paths = [path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    stems = [path.stem for path in paths]
    duplicates = sorted(pd.Series(stems)[pd.Series(stems).duplicated(keep=False)].unique().tolist()) if stems else []
    return set(stems), duplicates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--strict-historical", action="store_true")
    parser.add_argument("--require-mamba", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = ensure_dir("results/bioengineering_direct_baselines")
    required_files = [
        Path("data/raw/HAM10000/HAM10000_metadata.csv"),
        Path("splits/ham10000_seed42_train.csv"),
        Path("splits/ham10000_seed42_val.csv"),
        Path("splits/ham10000_seed42_test.csv"),
        Path("data/raw/ISIC_external/labels_harmonized.csv"),
    ]
    required_dirs = [Path("data/raw/HAM10000/images"), Path("data/raw/ISIC_external/images")]
    missing_data = [str(p) for p in required_files + required_dirs if not p.exists()]

    source_lock_path = Path("configs/bioengineering/source_lock.json")
    source_lock_error = None
    try:
        source_lock = (
            json.loads(source_lock_path.read_text(encoding="utf-8"))
            if source_lock_path.exists()
            else {}
        )
    except Exception as exc:
        source_lock = {}
        source_lock_error = str(exc)
    derma_config_path = Path("configs/bioengineering/dermamamba_architecture_faithful_seed42.yaml")
    derma_config = load_yaml(derma_config_path) if derma_config_path.exists() else {}
    locked_derma_revision = source_lock.get("dermamamba", {}).get("revision")
    configured_derma_revision = (
        derma_config.get("model", {}).get("dermamamba", {}).get("source_revision")
    )
    expected_mamba_version = str(source_lock.get("mamba_ssm", {}).get("release", "")).lstrip("v") or None
    source_lock_issues = []
    if not source_lock:
        source_lock_issues.append("source_lock_missing_or_empty")
    if source_lock_error:
        source_lock_issues.append("source_lock_invalid_json")
    if not locked_derma_revision:
        source_lock_issues.append("dermamamba_revision_missing_from_source_lock")
    if configured_derma_revision != locked_derma_revision:
        source_lock_issues.append("dermamamba_config_revision_does_not_match_source_lock")
    derma_protocol_ok, derma_protocol_detail = dermamamba_protocol_ok(
        derma_config, locked_derma_revision
    )
    if not derma_protocol_ok:
        source_lock_issues.append("dermamamba_config_does_not_match_frozen_adapter_protocol")

    split_audit = audit_ham_split_files(seed=42)
    split_rows = split_audit["rows"]
    split_hashes = split_audit["sha256"]

    data_alignment_issues: list[str] = []
    ham_metadata_path = Path("data/raw/HAM10000/HAM10000_metadata.csv")
    ham_image_dir = Path("data/raw/HAM10000/images")
    missing_ham_images: list[str] = []
    duplicate_ham_stems: list[str] = []
    if ham_metadata_path.is_file() and ham_image_dir.is_dir():
        ham_metadata = pd.read_csv(ham_metadata_path)
        if "image_id" not in ham_metadata:
            data_alignment_issues.append("ham_metadata_missing_image_id")
        else:
            ham_ids = ham_metadata["image_id"].astype(str)
            if ham_ids.duplicated().any():
                data_alignment_issues.append("ham_metadata_duplicate_image_id")
            ham_image_ids, duplicate_ham_stems = image_stems(ham_image_dir)
            missing_ham_images = sorted(set(ham_ids) - ham_image_ids)
            if duplicate_ham_stems:
                data_alignment_issues.append("ham_image_stems_are_not_unique")
            if missing_ham_images:
                data_alignment_issues.append("ham_metadata_images_missing")
    external_path = Path("data/raw/ISIC_external/labels_harmonized.csv")
    external = pd.read_csv(external_path) if external_path.exists() else pd.DataFrame()
    external_summary = {
        "rows": int(len(external)),
        "melanoma_positive": int(external["melanoma"].sum()) if "melanoma" in external.columns else None,
        "sha256": file_sha256(external_path) if external_path.exists() else None,
    }
    external_image_dir = Path("data/raw/ISIC_external/images")
    missing_external_images: list[str] = []
    duplicate_external_stems: list[str] = []
    if not external.empty:
        required_external_columns = {"image_id", "melanoma"}
        if not required_external_columns.issubset(external.columns):
            data_alignment_issues.append("external_labels_missing_required_columns")
        else:
            external_ids = external["image_id"].astype(str)
            if external_ids.duplicated().any():
                data_alignment_issues.append("external_labels_duplicate_image_id")
            labels = set(pd.to_numeric(external["melanoma"], errors="coerce").dropna().astype(int))
            if labels != {0, 1}:
                data_alignment_issues.append("external_labels_do_not_contain_both_binary_classes")
            if external_image_dir.is_dir():
                external_image_ids, duplicate_external_stems = image_stems(external_image_dir)
                missing_external_images = sorted(set(external_ids) - external_image_ids)
                if duplicate_external_stems:
                    data_alignment_issues.append("external_image_stems_are_not_unique")
                if missing_external_images:
                    data_alignment_issues.append("external_labelled_images_missing")
    external_summary.update(
        {
            "missing_labelled_images_count": len(missing_external_images),
            "missing_labelled_images_preview": missing_external_images[:20],
            "duplicate_image_stems_count": len(duplicate_external_stems),
        }
    )

    historical = {}
    for spec in DIRECT_MODEL_SPECS:
        if spec.model_id == "dermamamba":
            continue
        run = discover_run(runs_dir, spec, seed=42)
        historical[spec.model_id] = None if run is None else str(run)
    missing_historical = [key for key, value in historical.items() if value is None]

    mask_dataset = SkinLesionMaskDataset(
        image_dir="data/raw/ISIC_masks/images",
        mask_dir="data/raw/ISIC_masks/masks",
        transform=None,
        input_size=224,
    )
    mask_pairs = int(len(mask_dataset))

    try:
        mamba_version = version("mamba-ssm")
        from mamba_ssm import Mamba  # noqa: F401

        mamba_importable = True
        mamba_error = None
    except Exception as exc:
        mamba_version = None
        mamba_importable = False
        mamba_error = str(exc)

    sources = {
        "dermamamba": {
            "repository": "https://github.com/Glow945/Skin_Lesions",
            "paper_doi": "10.3390/bioengineering12101030",
            "locked_revision": locked_derma_revision,
            "configured_revision": configured_derma_revision,
            "configuration_matches_lock": configured_derma_revision == locked_derma_revision,
            "head": github_head("Glow945", "Skin_Lesions"),
            "reproduction_class": "clean_room_architecture_faithful_port",
            "official_checkpoint_available": False,
            "official_training_config_available": False,
            "repository_license_file_detected": False,
        },
        "mamba_ssm": {
            "repository": "https://github.com/state-spaces/mamba",
            "release": "v2.3.2.post1",
            "expected_installed_version": expected_mamba_version,
            "head": github_head("state-spaces", "mamba"),
            "installed_version": mamba_version,
            "installed_version_matches_lock": (
                mamba_version == expected_mamba_version if mamba_version is not None else None
            ),
            "importable": mamba_importable,
            "import_error": mamba_error,
        },
    }

    blocking = [*missing_data, *source_lock_issues, *data_alignment_issues]
    blocking.extend(f"split_integrity:{issue}" for issue in split_audit["issues"])
    if args.strict_historical:
        blocking.extend(f"historical_run:{item}" for item in missing_historical)
    if args.require_mamba and not mamba_importable:
        blocking.append("mamba_ssm_not_importable")
    if args.require_mamba and mamba_importable and mamba_version != expected_mamba_version:
        blocking.append("mamba_ssm_version_does_not_match_source_lock")
    status = {
        "status": "ready" if not blocking else "blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "blocking_issues": blocking,
        "missing_data": missing_data,
        "source_lock": {
            "path": str(source_lock_path),
            "sha256": file_sha256(source_lock_path) if source_lock_path.exists() else None,
            "issues": source_lock_issues,
            "parse_error": source_lock_error,
            "dermamamba_protocol": derma_protocol_detail,
        },
        "split_rows": split_rows,
        "split_sha256": split_hashes,
        "split_integrity": split_audit,
        "data_alignment": {
            "issues": data_alignment_issues,
            "missing_ham_images_count": len(missing_ham_images),
            "missing_ham_images_preview": missing_ham_images[:20],
            "duplicate_ham_image_stems_count": len(duplicate_ham_stems),
        },
        "external": external_summary,
        "historical_seed42_runs": historical,
        "missing_historical_runs": missing_historical,
        "mask_extension": {
            "available_pairs": mask_pairs,
            "target_pairs": 200,
            "can_run_200": mask_pairs >= 200,
            "policy": "never duplicate or synthesize source masks to reach the target",
        },
        "sources": sources,
    }
    write_json(status, output_dir / "preflight_status.json")
    print(
        f"bioengineering_preflight status={status['status']} missing_data={len(missing_data)} "
        f"missing_historical={len(missing_historical)} mask_pairs={mask_pairs} "
        f"mamba_importable={mamba_importable}"
    )
    if blocking:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
