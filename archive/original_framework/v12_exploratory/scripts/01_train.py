from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesionmamba.engine.train import train_one_run
from lesionmamba.utils.config import load_yaml
from lesionmamba.utils.io import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    run_dir = train_one_run(config)
    print(run_dir)


if __name__ == "__main__":
    main()
