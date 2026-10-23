from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from lesionmamba.utils.io import ensure_dir


def save_text_figure(lines: list[str], title: str, path: Path) -> None:
    plt.figure(figsize=(12, 7))
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.text(0.03, 0.92, "\n".join(lines), va="top", family="monospace", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    out = ensure_dir("figures/architecture")
    source = ensure_dir(out / "source")
    fig1 = [
        "Dermoscopic image -> lightweight augmentation",
        "-> Morphology-Preserving Overlapping Patch Stem",
        "-> Stage 1: local texture encoding",
        "-> Stage 2: local-global LesionMamba blocks",
        "-> Stage 3: compact multi-scale selective scan",
        "-> Stage 4: global lesion context",
        "-> lesion-aware global pooling -> classification head",
        "-> prediction + internal lesion attention map",
    ]
    fig2 = [
        "DP-MSS",
        "Path A: Local Window Selective Scan (k=7)",
        "Path B: Global Cross Selective Scan (left/right/top/bottom)",
        "Fusion: alpha = sigmoid(Conv1x1([F_lw, F_gc]))",
        "F_mss = alpha * F_lw + (1 - alpha) * F_gc",
    ]
    for ext in ["png", "pdf"]:
        save_text_figure(fig1, "Figure 1. LesionMamba workflow", out / f"figure1_overall_workflow.{ext}")
        save_text_figure(fig2, "Figure 2. DP-MSS module", out / f"figure2_dpmss_module.{ext}")
    (source / "figure1_overall_workflow.txt").write_text("\n".join(fig1), encoding="utf-8")
    (source / "figure2_dpmss_module.txt").write_text("\n".join(fig2), encoding="utf-8")
    print("architecture figures exported")


if __name__ == "__main__":
    main()
