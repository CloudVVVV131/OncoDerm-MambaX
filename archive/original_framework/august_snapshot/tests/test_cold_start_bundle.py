from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from zipfile import ZipFile

import pytest

from lesionmamba.models import registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_numeric_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_entry(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "category": "test",
    }


def make_tiny_cold_start_tree(root: Path) -> None:
    run_ids = [
        f"historical_model_{index}_seed42_20260101_00000{index}" for index in range(6)
    ]
    for run_id in run_ids:
        run = root / "outputs" / "runs" / run_id
        (run / "checkpoints").mkdir(parents=True)
        (run / "config.yaml").write_text("project:\n  seed: 42\n", encoding="utf-8")
        (run / "environment.json").write_text("{}\n", encoding="utf-8")
        (run / "checkpoints" / "best_val_macro_f1.pth").write_bytes(run_id.encode())
        prediction_dir = root / "outputs" / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for split in ("val", "test", "external"):
            if split == "external":
                image_id = "external_example"
            elif split == "val":
                image_id = "seed42_val"
            else:
                image_id = "seed42_test"
            (prediction_dir / f"{run_id}_{split}_predictions.csv").write_text(
                f"image_id,label,true_label,prob_mel\n{image_id},0,0,0.1\n",
                encoding="utf-8",
            )
    split_names = [
        "ham10000_seed42_train.csv",
        "ham10000_seed42_val.csv",
        "ham10000_seed42_test.csv",
        "ham10000_seed2024_train.csv",
        "ham10000_seed2024_val.csv",
        "ham10000_seed2024_test.csv",
    ]
    (root / "splits").mkdir()
    for name in split_names:
        seed = "2024" if "seed2024" in name else "42"
        split = name.removesuffix(".csv").rsplit("_", 1)[-1]
        (root / "splits" / name).write_text(
            f"image_id,dx\nseed{seed}_{split},nv\n", encoding="utf-8"
        )
    (root / "data" / "raw").mkdir(parents=True)
    (root / "COLD_START_README.md").write_text("cold start\n", encoding="utf-8")
    files = [
        file_entry(root, path) for path in sorted(root.rglob("*")) if path.is_file()
    ]
    manifest = {
        "schema_version": 1,
        "expected_historical_runs": [{"run_id": run_id} for run_id in run_ids],
        "required_fixed_splits": split_names,
        "semantic_expectations": {
            "ham_total_rows": 3,
            "split_rows": {
                "42": {"train": 1, "val": 1, "test": 1},
                "2024": {"train": 1, "val": 1, "test": 1},
            },
            "external_rows": 1,
            "external_positive": 0,
        },
        "public_data_policy": {
            "included_in_upload_bundle": False,
            "downloaded_on_cloud": True,
        },
        "files": files,
    }
    (root / "COLD_START_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def test_zip_indexing_and_mask_canonicalization_are_strict(tmp_path: Path) -> None:
    script = load_numeric_script("24_prepare_bioengineering_cloud_data.py")
    archive_path = tmp_path / "images.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("part_a/ISIC_0001.jpg", b"a")
        archive.writestr("part_b/ISIC_0002.png", b"b")
    with ZipFile(archive_path) as archive:
        index = script.image_members(archive)
    assert sorted(index) == ["ISIC_0001", "ISIC_0002"]
    assert script.canonical_mask_stem("ISIC_0001_segmentation.png") == "ISIC_0001"
    assert script.canonical_mask_stem("ISIC_0002_ground_truth.png") == "ISIC_0002"

    duplicate_path = tmp_path / "duplicates.zip"
    with ZipFile(duplicate_path, "w") as archive:
        archive.writestr("a/ISIC_0001.jpg", b"a")
        archive.writestr("b/ISIC_0001.png", b"b")
    with (
        ZipFile(duplicate_path) as archive,
        pytest.raises(ValueError, match="Duplicate image stem"),
    ):
        script.image_members(archive)


def test_checkpoint_restore_builder_disables_pretrained_without_mutating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        registry, "build_model", lambda config: captured.append(config) or config
    )
    original = {"model": {"name": "demo", "pretrained": True}}
    restored = registry.build_model_for_checkpoint(original)
    assert original["model"]["pretrained"] is True
    assert restored["model"]["pretrained"] is False
    assert captured[0] is restored


def test_artifact_audit_and_streaming_archive_verification(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    make_tiny_cold_start_tree(project)

    audit_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "25_bioengineering_cold_start_audit.py"),
        "--project-root",
        str(project),
        "--artifacts-only",
        "--output",
        str(tmp_path / "artifact_audit.json"),
    ]
    subprocess.run(audit_command, check=True, cwd=PROJECT_ROOT)

    archive_path = tmp_path / "cold_start.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(project, arcname="project")
    verify_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "26_verify_cold_start_archive.py"),
        "--archive",
        str(archive_path),
    ]
    subprocess.run(verify_command, check=True, cwd=PROJECT_ROOT)
    qa = json.loads(Path(f"{archive_path}.qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "passed"

    prediction = next((project / "outputs" / "predictions").glob("*.csv"))
    prediction.write_text("corrupt\n", encoding="utf-8")
    failed = subprocess.run(audit_command, check=False, cwd=PROJECT_ROOT)
    assert failed.returncode != 0
