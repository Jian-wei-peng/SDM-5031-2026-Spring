#!/usr/bin/env python3
"""
Re-plot loss and score curves (epochs 0–500) for every log.txt found under
the project root.  Outputs loss.jpg and score.jpg in the same directory as
each log.txt.

Parses the final `train_score_list = [...]` / `train_loss_list = [...]`
arrays printed at the end of each training log.
"""

import os
import re
import ast
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 0. Project root & style
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(PROJECT_ROOT, "utils", "log_image_style", "style_loss_1.json")

default_config = {
    "figsize": {"x": 10, "y": 5},
    "xlim": {"min": None, "max": None},
    "ylim": {"min": None, "max": None},
    "grid": True,
}

try:
    with open(STYLE_PATH, "r") as f:
        style_config = json.load(f)
    print(f"Loaded style from {STYLE_PATH}")
except FileNotFoundError:
    style_config = default_config
    print(f"Style file not found at {STYLE_PATH}, using defaults.")


# ---------------------------------------------------------------------------
# 1. Parse log.txt → (epochs, scores, losses)
# ---------------------------------------------------------------------------
def parse_log(log_path):
    """
    Parse the final `train_score_list = [...]` and `train_loss_list = [...]`
    arrays from a log.txt file.  Values are per-epoch, epoch 1 at index 0.
    Falls back to per-line regex parsing if no final arrays are found.
    """
    try:
        with open(log_path, "r") as f:
            text = f.read()
    except Exception:
        return [], [], []

    epochs = []
    scores = []
    losses = []

    # --- Primary: parse final array dump ---
    def _extract(varname):
        pattern = re.compile(
            rf"{re.escape(varname)}\s*=\s*(\[.*?\])", re.DOTALL
        )
        matches = pattern.findall(text)
        if not matches:
            return None
        try:
            return ast.literal_eval(matches[-1])
        except Exception:
            return None

    score_list = _extract("train_score_list")
    loss_list = _extract("train_loss_list")

    if score_list is not None and loss_list is not None:
        n = min(len(score_list), len(loss_list))
        for i in range(n):
            epoch = i + 1
            if epoch > 500:
                break
            epochs.append(epoch)
            scores.append(score_list[i])
            losses.append(loss_list[i])
        return epochs, scores, losses

    # --- Fallback: parse per-line "Train (100%)" summaries ---
    pattern = re.compile(
        r"Epoch\s+(\d+):\s+Train\s*\(100%\)\s+Score:\s*([-\d.]+),\s*Loss:\s*([-\d.]+)"
    )
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                if 0 <= epoch <= 500:
                    epochs.append(epoch)
                    scores.append(float(m.group(2)))
                    losses.append(float(m.group(3)))

    return epochs, scores, losses


# ---------------------------------------------------------------------------
# 2. Plot and save
# ---------------------------------------------------------------------------
def plot_and_save(epochs, values, ylabel, out_path):
    """Plot a single metric (loss or score) with consistent style."""
    figsize = (style_config["figsize"]["x"], style_config["figsize"]["y"])
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(epochs, values, marker=".", markersize=2, linewidth=1, color="#1f77b4")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # Axis limits from style (only apply if explicitly set)
    if (
        style_config["xlim"]["min"] is not None
        or style_config["xlim"]["max"] is not None
    ):
        ax.set_xlim(style_config["xlim"]["min"], style_config["xlim"]["max"])
    if (
        style_config["ylim"]["min"] is not None
        or style_config["ylim"]["max"] is not None
    ):
        ax.set_ylim(style_config["ylim"]["min"], style_config["ylim"]["max"])

    ax.grid(style_config.get("grid", True))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# 3. Main loop
# ---------------------------------------------------------------------------
def main():
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        if "log.txt" not in filenames:
            continue

        log_path = os.path.join(dirpath, "log.txt")
        print(f"\nProcessing: {log_path}")

        epochs, scores, losses = parse_log(log_path)

        if not epochs:
            print("  ⚠ No epoch data found, skipping.")
            continue

        print(f"  Parsed {len(epochs)} epochs (range {min(epochs)}–{max(epochs)})")

        log_dir = os.path.dirname(log_path)

        # Loss plot
        loss_out = os.path.join(log_dir, "loss.jpg")
        plot_and_save(epochs, losses, "Loss", loss_out)

        # Score plot
        score_out = os.path.join(log_dir, "score.jpg")
        plot_and_save(epochs, scores, "Score", score_out)

    print("\n✅ Done! All plots generated.")


if __name__ == "__main__":
    main()
