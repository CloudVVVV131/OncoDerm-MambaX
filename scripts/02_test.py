from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lesionmamba.engine.checkpoint import load_checkpoint
from lesionmamba.engine.evaluate import evaluate_and_save
from lesionmamba.engine.train import build_dataloader
from lesionmamba.models.registry import build_model_for_checkpoint
from lesionmamba.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    run_dir = Path(args.run)
    config = load_yaml(run_dir / "config.yaml")
    ckpt_path = run_dir / "checkpoints" / "best_val_macro_f1.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Validation-selected checkpoint required: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_for_checkpoint(config).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    loader = build_dataloader(config, args.split, shuffle=False)
    metrics = evaluate_and_save(
        model,
        loader,
        device,
        run_id=run_dir.name,
        split=args.split,
        run_dir=run_dir,
        amp=bool(config["train"].get("amp", True)),
    )
    print(metrics)


if __name__ == "__main__":
    main()
