"""Synthetic unit tests for configuration, checkpoints and data preparation."""
from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ConfigTests(unittest.TestCase):
    def test_resolved_configs_match_index(self):
        from lesionmamba.utils.config import load_yaml
        from lesionmamba.data.splits import configured_split_seed
        with (ROOT / "configs/experiments.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 17)
        for row in rows:
            cfg = load_yaml(ROOT / row["config"])
            self.assertNotIn("_base_", cfg)
            self.assertEqual(configured_split_seed(cfg), int(row["split_seed"]))
            self.assertEqual(cfg["project"]["seed"], int(row["training_seed"]))
            self.assertFalse(cfg["model"]["allow_random_init_fallback"])

    def test_principal_and_optional_backends_are_distinct(self):
        from lesionmamba.utils.config import load_yaml
        base = load_yaml(ROOT / "configs/runs/oncoderm_mambax_b6_full_seed42.yaml")
        optional = load_yaml(ROOT / "configs/runs/oncoderm_mambax_b6m_true_mamba_seed42.yaml")
        self.assertEqual(optional["model"]["cssa"]["backend"], "mamba_ssm")
        self.assertNotEqual(base["model"]["cssa"].get("backend"), "mamba_ssm")
        self.assertEqual(base["data"]["mask_aux"]["expected_pairs"], 100)
        self.assertEqual(optional["data"]["mask_aux"]["expected_pairs"], 200)

    def test_restore_build_disables_download_without_mutating_config(self):
        from lesionmamba.models import registry
        cfg = {"model": {"source": "oncoderm_mambax", "pretrained": True}}
        with patch.object(registry, "build_model", return_value="model") as builder:
            self.assertEqual(registry.build_model_for_checkpoint(cfg), "model")
            self.assertFalse(builder.call_args.args[0]["model"]["pretrained"])
        self.assertTrue(cfg["model"]["pretrained"])

    def test_unreleased_model_does_not_import_external_code(self):
        from lesionmamba.models.registry import build_model
        with self.assertRaisesRegex(ValueError, "not distributed"):
            build_model({"model": {"source": "dermamamba_architecture_faithful"}})


class CheckpointTests(unittest.TestCase):
    def test_safe_tensor_checkpoint_roundtrip(self):
        from lesionmamba.engine.checkpoint import save_checkpoint, load_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pth"
            save_checkpoint({"model": {"weight": torch.ones(2)}, "version": torch.__version__}, path)
            loaded = load_checkpoint(path)
            self.assertTrue(torch.equal(loaded["model"]["weight"], torch.ones(2)))

    def test_missing_auxiliary_cohort_is_an_error(self):
        from lesionmamba.engine.train import build_aux_mask_dataloader
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"data": {"mask_aux": {"enabled": True, "image_dir": tmp + "/images", "mask_dir": tmp + "/masks"}}}
            with self.assertRaises(FileNotFoundError):
                build_aux_mask_dataloader(cfg)

    def test_missing_validation_does_not_default_to_half(self):
        module = load_script("08_external_inference.py")
        with tempfile.TemporaryDirectory() as tmp:
            old = Path.cwd()
            try:
                os.chdir(tmp)
                with self.assertRaises(FileNotFoundError):
                    module.validation_thresholds(Path(tmp) / "test-run")
            finally:
                os.chdir(old)


class EvidenceTests(unittest.TestCase):
    def test_external_summary_labels_both_score_sources(self):
        module = load_script("08_external_inference.py")
        metrics = {"roc_auc": 0.73, "pr_auc": 0.24}
        for score_source in ("binary_head_sigmoid", "seven_class_softmax_mel"):
            with self.subTest(score_source=score_source):
                row = module.build_summary_row("run", "validation", 0.91, metrics, score_source)
                self.assertEqual(row["Internal multiclass MEL-AUC"], 0.91)
                self.assertEqual(row["External endpoint ROC-AUC"], 0.73)
                self.assertEqual(row["internal_score_source"], "seven_class_softmax_mel")
                self.assertEqual(row["external_score_source"], score_source)
                self.assertEqual(row["pr_auc"], 0.24)
                self.assertNotIn("Drop", row)
        self.assertEqual(metrics, {"roc_auc": 0.73, "pr_auc": 0.24})

    def test_external_summary_preserves_missing_internal_auc(self):
        module = load_script("08_external_inference.py")
        row = module.build_summary_row("run", "validation", None, {"roc_auc": 0.73}, "binary_head_sigmoid")
        self.assertIsNone(row["Internal multiclass MEL-AUC"])
        self.assertNotIn("Drop", row)

    def test_external_summary_merge_removes_legacy_drop(self):
        import pandas as pd
        module = load_script("08_external_inference.py")
        old = pd.DataFrame([
            {"run_id": "keep", "Internal Mel-AUC": 0.91, "External Mel-AUC": 0.73, "Drop": 0.18},
            {"run_id": "replace", "Internal Mel-AUC": 0.81, "External Mel-AUC": 0.63, "Drop": 0.18},
        ])
        original = old.copy(deep=True)
        new = pd.DataFrame([module.build_summary_row(
            "replace", "validation", 0.89, {"roc_auc": 0.74}, "binary_head_sigmoid"
        )])
        merged = module.merge_summary_rows(old, new).set_index("run_id")
        self.assertEqual(list(merged.index), ["keep", "replace"])
        self.assertEqual(merged.loc["keep", "Internal multiclass MEL-AUC"], 0.91)
        self.assertEqual(merged.loc["replace", "External endpoint ROC-AUC"], 0.74)
        self.assertTrue(pd.isna(merged.loc["keep", "external_score_source"]))
        for column in ("Drop", "Internal Mel-AUC", "External Mel-AUC"):
            self.assertNotIn(column, merged)
        pd.testing.assert_frame_equal(old, original)

    def test_external_summary_merge_supports_current_schema(self):
        import pandas as pd
        module = load_script("08_external_inference.py")
        old = pd.DataFrame([module.build_summary_row(
            "keep", "validation", 0.9, {"roc_auc": 0.7}, "seven_class_softmax_mel"
        )])
        new = pd.DataFrame([module.build_summary_row(
            "new", "validation", 0.8, {"roc_auc": 0.6}, "binary_head_sigmoid"
        )])
        merged = module.merge_summary_rows(old, new)
        self.assertEqual(merged["run_id"].tolist(), ["keep", "new"])
        self.assertEqual(merged["external_score_source"].tolist(), ["seven_class_softmax_mel", "binary_head_sigmoid"])
        self.assertNotIn("Drop", merged)

    def test_external_summary_merge_upgrades_mixed_schema(self):
        import pandas as pd
        module = load_script("08_external_inference.py")
        old = pd.DataFrame([
            {"run_id": "legacy", "Internal Mel-AUC": 0.9, "External Mel-AUC": 0.7},
            {"run_id": "current", "Internal multiclass MEL-AUC": 0.8, "External endpoint ROC-AUC": 0.6},
        ])
        new = pd.DataFrame([module.build_summary_row(
            "new", "validation", 0.85, {"roc_auc": 0.65}, "binary_head_sigmoid"
        )])
        merged = module.merge_summary_rows(old, new)
        self.assertEqual(merged["Internal multiclass MEL-AUC"].tolist(), [0.9, 0.8, 0.85])
        self.assertEqual(merged["External endpoint ROC-AUC"].tolist(), [0.7, 0.6, 0.65])
        self.assertNotIn("Internal Mel-AUC", merged)
        self.assertNotIn("External Mel-AUC", merged)

    def test_ambiguous_runs_require_explicit_selection(self):
        module = load_script("17_submission_v1_evidence.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for suffix in ("a", "b"):
                run = root / "runs" / f"oncoderm_mambax_b2_metadata_cmgf_seed42_{suffix}"
                run.mkdir(parents=True)
                (run / "config.yaml").write_text("{}", encoding="utf-8")
            cfg = {"models": {"required_models": ["b2"], "required_seeds": [42]}}
            paths = module.EvidencePaths(root, root / "predictions", root / "runs", root / "results", root / "figures")
            status = module.status_base(cfg, paths)
            with self.assertRaisesRegex(ValueError, "Ambiguous runs"):
                module.discover_required_runs(cfg, paths, status)
            self.assertEqual(len(status["duplicate_candidates"]), 1)


class SplitTests(unittest.TestCase):
    def fixture(self, root):
        metadata = root / "metadata.csv"
        write_csv(metadata, [{"image_id": f"I{i}", "lesion_id": f"L{i}", "dx": "nv"} for i in range(6)],
                  ["image_id", "lesion_id", "dx"])
        for seed in (42, 2024):
            for split, ids in (("train", [2, 0]), ("val", [5, 1]), ("test", [4, 3])):
                write_csv(root / "ids" / f"ham10000_seed{seed}_{split}_ids.csv",
                          [{"image_id": f"I{i}"} for i in ids], ["image_id"])
        return metadata

    def test_preserves_order_and_all_columns(self):
        module = load_script("restore_splits.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module.restore_fixed_splits(self.fixture(root), root / "ids", root / "out")
            with (root / "out/ham10000_seed42_train.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["image_id"] for r in rows], ["I2", "I0"])
            self.assertEqual(rows[0]["lesion_id"], "L2")

    def test_overlap_is_rejected_before_output(self):
        module = load_script("restore_splits.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = self.fixture(root)
            write_csv(root / "ids/ham10000_seed42_val_ids.csv", [{"image_id": "I2"}], ["image_id"])
            with self.assertRaises(ValueError):
                module.restore_fixed_splits(metadata, root / "ids", root / "out")
            self.assertFalse((root / "out").exists())

    def test_existing_splits_are_not_overwritten(self):
        module = load_script("restore_splits.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = self.fixture(root)
            (root / "out").mkdir()
            with self.assertRaises(FileExistsError):
                module.restore_fixed_splits(metadata, root / "ids", root / "out")

    def test_released_partitions_have_expected_counts_and_disjoint_ids(self):
        counts = {42: (6982, 1513, 1520), 2024: (7013, 1515, 1487)}
        universes = []
        for seed, sizes in counts.items():
            seen = set()
            for split, size in zip(("train", "val", "test"), sizes):
                with (ROOT / f"data/manifests/ham10000_seed{seed}_{split}_ids.csv").open() as handle:
                    ids = [r["image_id"] for r in csv.DictReader(handle)]
                self.assertEqual(len(ids), size)
                self.assertEqual(len(set(ids)), size)
                self.assertFalse(seen.intersection(ids))
                seen.update(ids)
            universes.append(seen)
        self.assertEqual(universes[0], universes[1])


class ArchiveTests(unittest.TestCase):
    def test_identical_duplicate_image_is_deterministic(self):
        module = load_script("prepare_data.py")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("lower/I0.jpg", b"same")
                archive.writestr("UPPER/I0.jpg", b"same")
            with zipfile.ZipFile(path) as archive:
                result = module.image_members(archive, {"I0"})
                self.assertEqual(result["I0"].filename, "UPPER/I0.jpg")

    def test_conflicting_duplicate_is_rejected(self):
        module = load_script("prepare_data.py")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("a/I0.jpg", b"one")
                archive.writestr("b/I0.jpg", b"two")
            with zipfile.ZipFile(path) as archive:
                with self.assertRaises(ValueError):
                    module.image_members(archive, {"I0"})

    def test_archive_traversal_is_rejected(self):
        module = load_script("prepare_data.py")
        with self.assertRaises(ValueError):
            module.member_path(zipfile.ZipInfo("../outside.jpg"))

    def test_existing_dataset_is_untouched(self):
        module = load_script("prepare_data.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw/HAM10000").mkdir(parents=True)
            (root / "raw/HAM10000/keep.txt").write_text("keep")
            (root / "stage/HAM10000").mkdir(parents=True)
            with self.assertRaises(ValueError):
                module.replace_dataset_dirs(root / "raw", root / "stage", ["HAM10000"])
            self.assertEqual((root / "raw/HAM10000/keep.txt").read_text(), "keep")

    def test_partial_new_dataset_install_rolls_back(self):
        module = load_script("prepare_data.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            (root / "stage/A").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                module.replace_dataset_dirs(root / "raw", root / "stage", ["A", "B"])
            self.assertFalse((root / "raw/A").exists())


if __name__ == "__main__":
    unittest.main()
