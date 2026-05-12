#!/usr/bin/env python3
"""
Plot no_aug_gap and aug_gap vs checkpoint across all eval_batch TSV results.

Searches the workspace for eval_batch/*.tsv files, extracts the method name
from the result directory, and plots two figures:
  - plot_no_aug.png : no_aug_gap (%) vs checkpoint
  - plot_aug.png    : aug_gap (%) vs checkpoint

Usage:
    python plot_eval_batch.py [--outdir <path>]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Colors & markers for different methods (auto-cycle if more than defined)
# ---------------------------------------------------------------------------
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p"]
LINESTYLES = ["-", "--", "-.", ":"]


# ---------------------------------------------------------------------------
# Find all eval_batch TSV files
# ---------------------------------------------------------------------------
def find_tsv_files(root: Path) -> list[Path]:
    tsv_files = []
    for pattern in ["**/eval_batch/*.tsv"]:
        tsv_files.extend(root.glob(pattern))
    # Deduplicate and sort
    tsv_files = sorted(set(tsv_files))
    return tsv_files


# ---------------------------------------------------------------------------
# Extract method label from result directory name
# ---------------------------------------------------------------------------
def method_label(tsv_path: Path) -> str:
    """
    From a path like:
      TSP/POMO_model/result/20260430_110217_pjw_model_stage1_poly_pomo/eval_batch/batch_eval_xxx.tsv
    Extract "POMO_model/pjw_model_stage1_poly_pomo"
    """
    parts = tsv_path.parts
    # Find 'result' and take the directory after it
    try:
        ri = parts.index("result")
    except ValueError:
        return tsv_path.parent.parent.name  # fallback

    # parts[ri+1] is the timestamp directory name
    timestamp_dir = parts[ri + 1]
    # Remove leading timestamp prefix like "20260430_110217_"
    name = timestamp_dir
    # Strip the date_time prefix (YYYYMMDD_HHMMSS_)
    if len(name) > 16 and name[8] == "_" and name[15] == "_":
        name = name[16:]
    # Also handle train__ style
    if name.startswith("train__"):
        name = name[7:]

    # Prepend parent train dir name for disambiguation
    train_dir = parts[ri - 1] if ri > 0 else ""
    return f"{train_dir}/{name}"


# ---------------------------------------------------------------------------
# Parse a single TSV into (checkpoints, no_aug_gaps, aug_gaps)
# ---------------------------------------------------------------------------
def parse_tsv(tsv_path: Path) -> tuple[list[int], list[float], list[float]]:
    checkpoints = []
    no_aug_gaps = []
    aug_gaps = []
    with open(tsv_path, "r") as f:
        header = f.readline().strip().split("\t")
        try:
            ci = header.index("checkpoint")
            ni = header.index("no_aug_gap")
            ai = header.index("aug_gap")
        except ValueError:
            print(f"  [WARN] Missing expected columns in {tsv_path}, skipping.")
            return [], [], []

        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) <= max(ci, ni, ai):
                continue
            try:
                ckpt = int(fields[ci])
                no_val = float(fields[ni])
                aug_val = float(fields[ai])
            except (ValueError, IndexError):
                continue
            checkpoints.append(ckpt)
            no_aug_gaps.append(no_val)
            aug_gaps.append(aug_val)
    return checkpoints, no_aug_gaps, aug_gaps


# ---------------------------------------------------------------------------
# Plot one metric
# ---------------------------------------------------------------------------
def plot_metric(
    methods: list[tuple[str, list[int], list[float]]],
    metric_name: str,
    output_path: Path,
):
    """Plot one metric (no_aug_gap or aug_gap) across all methods."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, ckpts, values) in enumerate(methods):
        if not ckpts:
            continue
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ls = LINESTYLES[i % len(LINESTYLES)]

        # Sort by checkpoint
        pairs = sorted(zip(ckpts, values))
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]

        ax.plot(x, y, color=color, marker=marker, linestyle=ls,
                linewidth=1.5, markersize=6, label=label)

    ax.set_xlabel("Checkpoint", fontsize=13)
    ax.set_ylabel(f"{metric_name} (%)", fontsize=13)
    ax.set_title(f"{metric_name} vs Checkpoint", fontsize=15)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # If all checkpoints are in a similar range, set integer ticks
    all_x = [x for _, ckpts, _ in methods for x in ckpts]
    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        ax.set_xlim(x_min - 10, x_max + 10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot no_aug/aug gaps from eval_batch TSV files"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Workspace root (default: parent of this script)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for plots (default: <root>/)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    outdir = Path(args.outdir).resolve() if args.outdir else root

    print(f"Searching for eval_batch TSV files under: {root}")

    tsv_files = find_tsv_files(root)
    if not tsv_files:
        print("No eval_batch/*.tsv files found.")
        sys.exit(0)

    print(f"Found {len(tsv_files)} TSV file(s):")
    for f in tsv_files:
        print(f"  {f.relative_to(root)}")

    # Parse all
    methods_no_aug = []  # (label, ckpts, no_aug_gaps)
    methods_aug = []     # (label, ckpts, aug_gaps)
    for tsv in tsv_files:
        label = method_label(tsv)
        ckpts, no_aug, aug = parse_tsv(tsv)
        if not ckpts:
            continue
        print(f"  Parsed '{label}': {len(ckpts)} checkpoints")
        methods_no_aug.append((label, ckpts, no_aug))
        methods_aug.append((label, ckpts, aug))

    if not methods_no_aug:
        print("No valid data to plot.")
        sys.exit(0)

    # Plot
    print("\nGenerating plots...")
    plot_metric(methods_no_aug, "no_aug_gap", outdir / "plot_no_aug.png")
    plot_metric(methods_aug, "aug_gap", outdir / "plot_aug.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
