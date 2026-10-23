from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tarfile
from typing import Iterable

import pandas as pd


SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def usable_file(path: Path, include_checkpoints: bool) -> bool:
    if not path.is_file():
        return False
    if any(part in SKIP_PARTS for part in path.parts) or path.suffix.lower() in SKIP_SUFFIXES:
        return False
    if not include_checkpoints and "checkpoints" in path.parts:
        return False
    return True


def files_under(path: Path, include_checkpoints: bool) -> Iterable[Path]:
    if usable_file(path, include_checkpoints):
        yield path
    elif path.is_dir():
        for candidate in path.rglob("*"):
            if usable_file(candidate, include_checkpoints):
                yield candidate


def collect_run_ids(project_root: Path, manifest: dict) -> list[str]:
    run_ids = {
        str(row["run_id"])
        for row in manifest.get("runs", [])
        if isinstance(row, dict) and row.get("run_id")
    }
    optional_dir = project_root / "results/bioengineering_direct_baselines/optional"
    for table_name in ["B6M_stability_runs.csv", "B6_crosssplit2024_metrics.csv"]:
        table_path = optional_dir / table_name
        if table_path.is_file() and table_path.stat().st_size:
            table = pd.read_csv(table_path)
            if "run_id" in table:
                run_ids.update(table["run_id"].dropna().astype(str))
    return sorted(run_ids)


def validate_selected_artifacts(project_root: Path, run_ids: list[str], include_checkpoints: bool) -> None:
    missing: list[str] = []
    for run_id in run_ids:
        run_dir = project_root / "outputs/runs" / run_id
        if not run_dir.is_dir():
            missing.append(str(run_dir))
            continue
        if not (run_dir / "config.yaml").is_file():
            missing.append(str(run_dir / "config.yaml"))
        if include_checkpoints and not any(
            (run_dir / "checkpoints" / name).is_file()
            for name in ["best_val_macro_f1.pth", "last.pth"]
        ):
            missing.append(f"{run_dir}/checkpoints/(best_val_macro_f1.pth|last.pth)")
        for split in ["val", "test", "external"]:
            prediction = project_root / "outputs/predictions" / f"{run_id}_{split}_predictions.csv"
            if not prediction.is_file() or prediction.stat().st_size == 0:
                missing.append(str(prediction))
    if missing:
        preview = "\n".join(f"- {item}" for item in missing[:30])
        raise RuntimeError(f"Cannot create a complete result package; artifacts are missing:\n{preview}")


def collect_files(project_root: Path, run_ids: list[str], include_checkpoints: bool) -> list[Path]:
    selected: set[Path] = set()

    def add(path: Path) -> None:
        selected.update(files_under(path, include_checkpoints))

    for path in [
        project_root / "results/bioengineering_direct_baselines",
        project_root / "results/efficiency_table.csv",
        project_root / "results/efficiency_meta.json",
        project_root / "configs",
        project_root / "lesionmamba",
        project_root / "scripts",
        project_root / "tests",
        project_root / "figures/explainability/bioengineering_direct_baselines",
    ]:
        add(path)

    for pattern in ["splits/ham10000_seed42_*.csv", "splits/ham10000_seed2024_*.csv"]:
        for path in project_root.glob(pattern):
            add(path)

    for name in [
        "README.md",
        "README_BIOENGINEERING_SUPPLEMENT.md",
        "BIOENGINEERING_SUPPLEMENT_TRACEABILITY.md",
        "requirements.txt",
        "requirements-autodl.txt",
        "requirements-bioengineering.txt",
        "run_bioengineering_supplement.sh",
        "run_bioengineering_supplement_full.sh",
    ]:
        add(project_root / name)

    for run_id in run_ids:
        add(project_root / "outputs/runs" / run_id)
        for prediction in (project_root / "outputs/predictions").glob(f"{run_id}_*"):
            add(prediction)
        add(project_root / "outputs/attention_maps" / run_id)

    for log_path in project_root.glob("autodl_bioengineering_supplement*.log"):
        add(log_path)

    return sorted(selected, key=lambda path: path.as_posix())


def parse_args() -> argparse.Namespace:
    project_default = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_default)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_default.parent / f"oncoderm_bioengineering_full_results_{stamp}.tar.gz",
    )
    parser.add_argument(
        "--exclude-checkpoints",
        action="store_true",
        help="Create a smaller analysis-only package. The default full package includes checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output = args.output.resolve()
    include_checkpoints = not args.exclude_checkpoints
    result_dir = project_root / "results/bioengineering_direct_baselines"
    audit_path = result_dir / "completion_audit.json"
    manifest_path = result_dir / "direct_baseline_manifest.json"
    audit = read_json(audit_path)
    manifest = read_json(manifest_path)
    if audit.get("status") != "passed":
        raise RuntimeError(f"Completion audit did not pass: {audit.get('status')!r}")
    if manifest.get("status") != "completed":
        raise RuntimeError(f"Direct-baseline manifest is not completed: {manifest.get('status')!r}")

    run_ids = collect_run_ids(project_root, manifest)
    if not run_ids:
        raise RuntimeError("No audited run IDs were found in the result manifest")
    validate_selected_artifacts(project_root, run_ids, include_checkpoints)

    package_manifest_path = result_dir / "result_package_manifest.json"
    package_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_packaging",
        "source_completion_audit_sha256": file_sha256(audit_path),
        "source_direct_manifest_sha256": file_sha256(manifest_path),
        "include_checkpoints": include_checkpoints,
        "selected_run_count": len(run_ids),
        "selected_run_ids": run_ids,
        "raw_images_included": False,
        "raw_data_exclusion_reason": "Raw HAM10000, SIIM-ISIC 2020, and mask images are not result artifacts.",
    }
    package_manifest_path.write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    files = collect_files(project_root, run_ids, include_checkpoints)
    if package_manifest_path not in files:
        files.append(package_manifest_path)
        files.sort(key=lambda path: path.as_posix())
    package_manifest["file_count"] = len(files)
    package_manifest["payload_bytes_excluding_package_manifest"] = sum(
        path.stat().st_size for path in files if path != package_manifest_path
    )
    package_manifest_path.write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    with tarfile.open(output, mode="w:gz", compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(project_root)
            archive.add(path, arcname=(Path("project") / relative).as_posix(), recursive=False)

    archive_hash = file_sha256(output)
    checksum_path = Path(f"{output}.sha256")
    checksum_path.write_text(f"{archive_hash}  {output.name}\n", encoding="ascii")
    print(
        f"result_package status=completed path={output} bytes={output.stat().st_size} "
        f"files={len(files)} runs={len(run_ids)} sha256={archive_hash}"
    )


if __name__ == "__main__":
    main()
