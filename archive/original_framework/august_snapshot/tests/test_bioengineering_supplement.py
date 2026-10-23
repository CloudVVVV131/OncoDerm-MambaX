from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from lesionmamba.analysis.bioengineering_supplement import (
    DIRECT_MODEL_SPECS,
    OPTIONAL_MODEL_SPECS,
    DirectModelSpec,
    discover_run,
    external_metrics,
    paired_bootstrap_delta,
    validation_thresholds,
)
from lesionmamba.data.datasets import LABEL_MAP
from lesionmamba.engine.train import build_optimizer_scheduler
from lesionmamba.models.dermamamba_reference import (
    DermaMambaArchitectureFaithful,
    MultiDirectionalConcatenation,
    _boundary_indices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBABILITY_COLUMNS = ["prob_akiec", "prob_bcc", "prob_bkl", "prob_df", "prob_mel", "prob_nv", "prob_vasc"]


def load_numeric_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_scan_orders_are_deterministic_permutations() -> None:
    scanner = MultiDirectionalConcatenation(14, 14)
    for index in range(4):
        order = getattr(scanner, f"order_{index}").tolist()
        assert sorted(order) == list(range(196))
    assert _boundary_indices(3, 3) == [0, 1, 2, 3, 5, 6, 7, 8, 4]
    tokens = torch.arange(2 * 196 * 3, dtype=torch.float32).reshape(2, 196, 3)
    assert scanner(tokens).shape == (2, 196, 12)


def test_dermamamba_tiny_forward_and_backward() -> None:
    config = {
        "data": {"input_size": 32, "num_classes": 7},
        "model": {
            "name": "dermamamba_architecture_faithful_test",
            "source": "dermamamba_architecture_faithful",
            "num_classes": 7,
            "pretrained": False,
            "dermamamba": {
                "patch_size": 16,
                "embed_dim": 8,
                "depth": 1,
                "d_state": 4,
                "d_conv": 2,
                "expand": 1,
                "fusion_dim": 16,
                "dropout": 0.0,
            },
        },
    }
    model = DermaMambaArchitectureFaithful(config).eval()
    image = torch.randn(1, 3, 32, 32, requires_grad=True)
    output = model(image)
    assert output["logits"].shape == (1, 7)
    assert output["fusion_weights"].shape == (1, 2)
    output["logits"].sum().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert model.metadata["pretrained"] is False


def test_optimizer_selection_honors_dermamamba_adam_recipe() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer, _ = build_optimizer_scheduler(
        {
            "train": {
                "optimizer": "adam",
                "lr": 1e-4,
                "weight_decay": 5e-5,
                "epochs": 2,
                "warmup_epochs": 1,
            }
        },
        model,
        steps_per_epoch=1,
    )
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(5e-5)


def test_run_discovery_cannot_mix_split_seeds(tmp_path: Path) -> None:
    spec = DirectModelSpec("demo", "Demo", "CNN", "demo", "prob_mel", "test")
    for split_seed, stamp in [(42, "20260101_000000"), (2024, "20260102_000000")]:
        run = tmp_path / f"demo_seed42_{stamp}"
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints" / "last.pth").write_bytes(b"checkpoint")
        (run / "TRAINING_COMPLETE").touch()
        (run / "config.yaml").write_text(
            yaml.safe_dump({"project": {"seed": 42, "split_seed": split_seed}}), encoding="utf-8"
        )
    incomplete = tmp_path / "demo_seed42_20260103_000000"
    (incomplete / "checkpoints").mkdir(parents=True)
    (incomplete / "checkpoints" / "last.pth").write_bytes(b"partial checkpoint")
    (incomplete / "config.yaml").write_text(
        yaml.safe_dump({"project": {"seed": 42, "split_seed": 42}}), encoding="utf-8"
    )
    selected = discover_run(tmp_path, spec, seed=42, expected_split_seed=42)
    assert selected is not None and selected.name.endswith("20260101_000000")
    selected_cross = discover_run(tmp_path, spec, seed=42, expected_split_seed=2024)
    assert selected_cross is not None and selected_cross.name.endswith("20260102_000000")


def test_external_operating_point_is_locked_on_validation() -> None:
    validation = pd.DataFrame(
        {
            "true_label": [4, 4, 0, 0],
            "prob_mel": [0.8, 0.7, 0.6, 0.1],
        }
    )
    external = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "prob_mel": [0.55, 0.45, 0.40, 0.35],
        }
    )
    thresholds = validation_thresholds(validation, "prob_mel")
    metrics = external_metrics(external, "prob_mel", thresholds)
    assert thresholds["youden"] == pytest.approx(0.7)
    assert metrics["sensitivity"] == pytest.approx(0.0)
    assert metrics["specificity"] == pytest.approx(1.0)


def test_paired_bootstrap_is_deterministic() -> None:
    aligned = pd.DataFrame(
        {
            "label": [0, 0, 0, 1, 1, 1],
            "score_a": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "score_b": [0.2, 0.4, 0.6, 0.4, 0.6, 0.8],
        }
    )
    first = paired_bootstrap_delta(aligned, "roc_auc", iterations=200, seed=42)
    second = paired_bootstrap_delta(aligned, "roc_auc", iterations=200, seed=42)
    assert first == second
    assert first["delta_a_minus_b"] > 0
    assert 0 < first["p_value_two_sided"] <= 1
    assert "plus-one correction" in first["p_value_method"]


def test_mask_pair_deduplication_counts_unique_images(tmp_path: Path) -> None:
    script = load_numeric_script("07_eval_mask_explainability.py")
    image_index = {"a": tmp_path / "a.jpg", "b": tmp_path / "b.jpg"}
    masks = [
        tmp_path / "a_mask.png",
        tmp_path / "a_segmentation.png",
        tmp_path / "b_mask.png",
        tmp_path / "b_groundtruth.png",
        tmp_path / "unmatched_mask.png",
    ]
    paired = script.unique_paired_masks(masks, image_index)
    assert [script.image_id_from_mask(path) for path in paired] == ["a", "b"]


def test_b6m_gate_includes_flops_resource_constraint() -> None:
    script = load_numeric_script("20_bioengineering_direct_summary.py")
    summary = pd.DataFrame(
        [
            {
                "model_id": "b6",
                "Macro-F1": 0.75,
                "External ROC-AUC": 0.80,
                "External PR-AUC": 0.15,
                "Params": 10.0,
                "FLOPs": 10.0,
                "Latency_ms_b1": 10.0,
            },
            {
                "model_id": "b6m",
                "Macro-F1": 0.75,
                "External ROC-AUC": 0.80,
                "External PR-AUC": 0.15,
                "Params": 12.0,
                "FLOPs": 13.0,
                "Latency_ms_b1": 12.0,
            },
        ]
    )
    failed = script.make_gate(summary)
    assert failed["status"] == "failed"
    assert failed["checks"]["flops"] is False
    summary.loc[summary["model_id"] == "b6m", "FLOPs"] = 12.0
    passed = script.make_gate(summary)
    assert passed["status"] == "passed"
    assert passed["checks"]["flops"] is True


def _prediction_frame(
    seed: int,
    external: bool = False,
    image_ids: list[str] | None = None,
    labels: np.ndarray | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if external:
        labels = np.array([0] * 28 + [1] * 12) if labels is None else np.asarray(labels, dtype=int)
        mel = np.clip(0.15 + 0.65 * labels + rng.normal(0, 0.08, len(labels)), 0.01, 0.99)
        return pd.DataFrame(
            {
                "image_id": image_ids or [f"external_{index:03d}" for index in range(len(labels))],
                "label": labels,
                "prob_mel": mel,
                "prob_mel_endpoint": np.clip(mel + rng.normal(0, 0.02, len(labels)), 0.01, 0.99),
            }
        )
    labels = np.tile(np.arange(7), 10) if labels is None else np.asarray(labels, dtype=int)
    probs = np.full((len(labels), 7), 0.04)
    probs[np.arange(len(labels)), labels] = 0.76
    probs += rng.uniform(0, 0.01, probs.shape)
    probs /= probs.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(probs, columns=PROBABILITY_COLUMNS)
    frame.insert(0, "true_label", labels)
    frame.insert(0, "image_id", image_ids or [f"internal_{index:03d}" for index in range(len(labels))])
    frame["prob_mel_endpoint"] = np.clip(frame["prob_mel"] + rng.normal(0, 0.01, len(frame)), 0.001, 0.999)
    return frame


def _write_run(
    root: Path,
    prefix: str,
    model_id: str,
    split_seed: int,
    offset: int,
    run_seed: int = 42,
) -> str:
    run_id = f"{prefix}_seed{run_seed}_20260101_{offset:06d}"
    run_dir = root / "outputs" / "runs" / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "last.pth").write_bytes(b"checkpoint")
    intended = "prob_mel" if model_id in {"swin_tiny", "dermamamba", "efficientnet_b0"} else "prob_mel_endpoint"
    config = {
        "project": {"seed": run_seed, "split_seed": split_seed},
        "data": {"input_size": 224},
        "model": {"name": prefix, "metadata": {"test": True}},
    }
    if model_id == "dermamamba":
        config = {
            "project": {"seed": run_seed, "split_seed": split_seed},
            "data": {
                "input_size": 224,
                "metadata": {"enabled": False},
                "mask_aux": {"enabled": False},
            },
            "train": {
                "epochs": 100,
                "batch_size": 16,
                "optimizer": "adam",
                "lr": 1e-4,
                "weight_decay": 5e-5,
                "warmup_epochs": 10,
                "scheduler": "cosine",
                "grad_clip": 1.0,
                "early_stopping_patience": 15,
                "selection_metric": "macro_f1",
                "selection_tie_breaker": "mel_auc",
            },
            "loss": {
                "name": "ce",
                "label_smoothing": 0.0,
                "lambda_sparse": 0.0,
                "lambda_tv": 0.0,
            },
            "model": {
                "name": prefix,
                "source": "dermamamba_architecture_faithful",
                "pretrained": True,
                "allow_random_init_fallback": False,
                "num_classes": 7,
                "dermamamba": {
                    "source_revision": "2f5542bf30b9c5f6e0cecc6b5ac498cce76ca567",
                    "patch_size": 16,
                    "embed_dim": 192,
                    "depth": 6,
                    "d_state": 16,
                    "d_conv": 4,
                    "expand": 2,
                    "fusion_dim": 1024,
                    "dropout": 0.2,
                },
            },
        }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    if model_id in {"b1", "b2", "b4", "b6", "b6_crosssplit2024"}:
        model_metadata = {"source": "oncoderm_mambax", "timm_name": "convnext_tiny", "pretrained": True}
    elif model_id == "swin_tiny":
        model_metadata = {
            "source": "timm",
            "timm_name": "swin_tiny_patch4_window7_224",
            "pretrained": True,
        }
    elif model_id == "efficientnet_b0":
        model_metadata = {"source": "timm", "timm_name": "efficientnet_b0", "pretrained": True}
    elif model_id == "dermamamba":
        model_metadata = {
            "source": "clean_room_architecture_faithful_port",
            "comparison_label": "DermaMamba (resource-matched architecture port)",
            "source_revision": "2f5542bf30b9c5f6e0cecc6b5ac498cce76ca567",
            "official_checkpoint_used": False,
            "pretrained": True,
            "embed_dim": 192,
            "depth": 6,
        }
    elif model_id == "b6m":
        model_metadata = {
            "source": "oncoderm_mambax",
            "timm_name": "convnext_tiny",
            "pretrained": True,
            "formal_mamba_block": True,
            "effective_ssm_backend": "state_spaces_mamba_ssm",
            "mamba_scope": "F4_only",
        }
    else:
        model_metadata = {"test": True}
    (run_dir / "environment.json").write_text(
        json.dumps({"model_metadata": model_metadata}), encoding="utf-8"
    )
    prediction_dir = root / "outputs" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for split, extra in [("val", 0), ("test", 1)]:
        split_frame = pd.read_csv(root / "splits" / f"ham10000_seed{split_seed}_{split}.csv")
        labels = split_frame["dx"].map({key: value for key, value in LABEL_MAP.items()}).to_numpy()
        frame = _prediction_frame(
            offset + extra,
            image_ids=split_frame["image_id"].astype(str).tolist(),
            labels=labels,
        )
        if intended == "prob_mel":
            frame["prob_mel_endpoint"] = frame["prob_mel"]
        frame.to_csv(prediction_dir / f"{run_id}_{split}_predictions.csv", index=False)
    external_labels = pd.read_csv(root / "external_labels.csv")
    external = _prediction_frame(
        offset + 2,
        external=True,
        image_ids=external_labels["image_id"].astype(str).tolist(),
        labels=external_labels["melanoma"].astype(int).to_numpy(),
    )
    if intended == "prob_mel":
        external["prob_mel_endpoint"] = external["prob_mel"]
    external.to_csv(prediction_dir / f"{run_id}_external_predictions.csv", index=False)
    return run_id


def test_summary_script_end_to_end_with_synthetic_predictions(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split_frames = {}
    offset = 0
    for split, count in [("train", 14), ("val", 7), ("test", 7)]:
        frame = pd.DataFrame(
            {
                "image_id": [f"ham_{index:03d}" for index in range(offset, offset + count)],
                "lesion_id": [f"lesion_{index:03d}" for index in range(offset, offset + count)],
                "dx": [list(PROBABILITY_COLUMNS)[index % 7].removeprefix("prob_") for index in range(count)],
            }
        )
        frame.to_csv(split_dir / f"ham10000_seed42_{split}.csv", index=False)
        split_frames[split] = frame
        offset += count
    metadata_path = tmp_path / "HAM10000_metadata.csv"
    metadata = pd.concat(split_frames.values(), ignore_index=True)
    metadata.to_csv(metadata_path, index=False)
    permuted = metadata.iloc[np.roll(np.arange(len(metadata)), 3)].reset_index(drop=True)
    for split, frame in {
        "train": permuted.iloc[:14],
        "val": permuted.iloc[14:21],
        "test": permuted.iloc[21:],
    }.items():
        frame.to_csv(split_dir / f"ham10000_seed2024_{split}.csv", index=False)
    external_labels_path = tmp_path / "external_labels.csv"
    pd.DataFrame(
        {
            "image_id": [f"external_{index:03d}" for index in range(40)],
            "melanoma": [0] * 28 + [1] * 12,
        }
    ).to_csv(external_labels_path, index=False)

    model_runs: dict[str, str] = {}
    for offset, spec in enumerate(DIRECT_MODEL_SPECS, start=1):
        model_runs[spec.model_id] = _write_run(tmp_path, spec.run_prefix, spec.model_id, 42, offset)
    efficientnet_spec = next(spec for spec in OPTIONAL_MODEL_SPECS if spec.model_id == "efficientnet_b0")
    model_runs[efficientnet_spec.model_id] = _write_run(
        tmp_path,
        efficientnet_spec.run_prefix,
        efficientnet_spec.model_id,
        42,
        10,
    )
    b6m = _write_run(tmp_path, "oncoderm_mambax_b6m_true_mamba", "b6m", 42, 20)
    _write_run(tmp_path, "oncoderm_mambax_b6m_true_mamba", "b6m", 42, 22, run_seed=2024)
    _write_run(tmp_path, "oncoderm_mambax_b6m_true_mamba", "b6m", 42, 23, run_seed=3407)
    cross = _write_run(
        tmp_path,
        "oncoderm_mambax_b6_full_crosssplit2024",
        "b6_crosssplit2024",
        2024,
        21,
    )
    efficiency = []
    for index, run_id in enumerate([*model_runs.values(), b6m, cross], start=1):
        efficiency.append(
            {
                "run_id": run_id,
                "params": 20_000_000 + index,
                "flops": 4_000_000_000 + index,
                "latency_ms_b1": 4.0 + index / 100,
                "peak_mem_mb": 1000 + index,
            }
        )
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    pd.DataFrame(efficiency).to_csv(result_dir / "efficiency_table.csv", index=False)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "20_bioengineering_direct_summary.py"),
        "--runs-dir",
        str(tmp_path / "outputs" / "runs"),
        "--predictions-dir",
        str(tmp_path / "outputs" / "predictions"),
        "--results-dir",
        str(result_dir),
        "--bootstrap-iters",
        "100",
        "--include-efficientnet",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    output = result_dir / "bioengineering_direct_baselines"
    table3 = pd.read_csv(output / "tables" / "Table3_direct_recent_baselines.csv")
    table_s5 = pd.read_csv(output / "tables" / "TableS5_architecture_fairness.csv")
    crosssplit = pd.read_csv(output / "optional" / "B6_crosssplit2024_metrics.csv")
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert len(table3) == 6
    assert len(table_s5) == 7
    assert "efficientnet_b0" in set(table_s5["model_id"])
    assert "b6m" not in set(table_s5["model_id"])
    assert set(crosssplit["split_seed"]) == {42, 2024}
    assert status["status"] == "completed"
    assert status["paired_bootstrap_rows"] == 2

    (output / "mask200_status.json").write_text(
        json.dumps(
            {
                "status": "not_run_insufficient_pairs",
                "required_pairs": 200,
                "available_pairs": 100,
                "reason": "Only unique real pairs are eligible.",
            }
        ),
        encoding="utf-8",
    )
    (output / "mamba_backend_smoke.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "mamba_ssm": "2.3.2.post1",
                "forward_and_backward_finite": True,
                "device": "cuda",
            }
        ),
        encoding="utf-8",
    )

    audit_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "22_bioengineering_completion_audit.py"),
        "--runs-dir",
        str(tmp_path / "outputs" / "runs"),
        "--predictions-dir",
        str(tmp_path / "outputs" / "predictions"),
        "--results-dir",
        str(result_dir),
        "--split-dir",
        str(split_dir),
        "--metadata-csv",
        str(metadata_path),
        "--external-labels-csv",
        str(external_labels_path),
        "--minimum-bootstrap-iters",
        "100",
        "--require-b6m",
        "--require-crosssplit",
        "--require-mask-status",
        "--include-efficientnet",
    ]
    subprocess.run(audit_command, cwd=PROJECT_ROOT, check=True)
    audit = json.loads((output / "completion_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    assert not audit["errors"]


def test_result_packer_requires_passed_audit_and_collects_full_run(tmp_path: Path) -> None:
    project = tmp_path / "project"
    result_dir = project / "results/bioengineering_direct_baselines"
    run_id = "demo_seed42_20260101_000000"
    run_dir = project / "outputs/runs" / run_id
    prediction_dir = project / "outputs/predictions"
    (run_dir / "checkpoints").mkdir(parents=True)
    prediction_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("project:\n  seed: 42\n", encoding="utf-8")
    (run_dir / "environment.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "checkpoints/last.pth").write_bytes(b"checkpoint")
    for split in ["val", "test", "external"]:
        (prediction_dir / f"{run_id}_{split}_predictions.csv").write_text(
            "image_id,label,prob_mel\nimage_1,0,0.1\n", encoding="utf-8"
        )
    (result_dir / "completion_audit.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    (result_dir / "direct_baseline_manifest.json").write_text(
        json.dumps({"status": "completed", "runs": [{"run_id": run_id}]}), encoding="utf-8"
    )
    output = tmp_path / "full_results.tar.gz"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/23_pack_bioengineering_results.py"),
        "--project-root",
        str(project),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    assert output.is_file() and Path(f"{output}.sha256").is_file()
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        assert f"project/outputs/runs/{run_id}/checkpoints/last.pth" in names
        assert f"project/outputs/predictions/{run_id}_external_predictions.csv" in names
        package_manifest = json.load(
            archive.extractfile("project/results/bioengineering_direct_baselines/result_package_manifest.json")
        )
    assert package_manifest["selected_run_ids"] == [run_id]
    assert package_manifest["include_checkpoints"] is True
