#!/usr/bin/env python3
"""
Batch evaluate checkpoints from a training result directory on the validation set.

For each checkpoint from --start to --end (step --step), this script runs test.py
TWICE: once with augmentation disabled (no_aug) and once with x8 augmentation (aug).
The SUMMARY_JSON printed by test.py is parsed and consolidated into a final report.

Usage:
    python batch_eval_checkpoints.py <result_dir> [options]

Example:
    python batch_eval_checkpoints.py TSP/POMO_train/result/20260426_191342_pjw_ft_curriculum_mixed
    python batch_eval_checkpoints.py TSP/POMO_train/result/20260426_191342_pjw_ft_curriculum_mixed --cuda_device 3 --start 50 --end 300

Output:
    - <output_dir>/batch_eval_<timestamp>.json   (full per-checkpoint results)
    - <output_dir>/batch_eval_<timestamp>.tsv    (tab-separated summary)
    - Pretty-printed summary table to stdout.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_venv_python(result_dir: Path) -> str:
    """
    Walk up from result_dir to find the repo root (the one containing .venv/).
    Returns the .venv/bin/python path if found, otherwise falls back to sys.executable.
    """
    candidate = result_dir.resolve()
    # Walk up at most 5 levels
    for _ in range(5):
        venv_py = candidate / ".venv" / "bin" / "python"
        if venv_py.is_file():
            return str(venv_py)
        parent = candidate.parent
        if parent == candidate:
            break  # reached filesystem root
        candidate = parent
    # Fallback
    return sys.executable


def find_test_py(result_dir: Path) -> Path:
    """
    Given a result directory like .../POMO_train/result/<timestamp_name>/,
    find the corresponding .../POMO_train/test.py (two levels up).
    """
    base_dir = result_dir.parent.parent  # e.g. POMO_train/
    test_py = base_dir / "test.py"
    if not test_py.exists():
        raise FileNotFoundError(
            f"Cannot find test.py at {test_py}. "
            f"Expected structure: <train_dir>/result/<run_dir>/ with test.py in <train_dir>/"
        )
    return test_py.resolve()


def find_data_val(result_dir: Path) -> Path:
    """
    Find the validation data directory (data/val) containing .tsp files.
    Tries:  <repo>/TSP/data/val  and  <train_dir>/data/val
    """
    base_dir = result_dir.parent.parent  # e.g. POMO_train/
    tsp_dir = base_dir.parent  # e.g. TSP/

    candidates = [
        tsp_dir / "data" / "val",  # TSP/data/val
        base_dir / "data" / "val",  # POMO_train/data/val (fallback)
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(f"Cannot find data/val directory. Tried: {candidates}")


def run_eval(
    python_exe: str,
    test_py: Path,
    checkpoint_path: Path,
    data_path: Path,
    *,
    augmentation_enable: bool,
    cuda_device: int = 0,
    timeout: int = 900,
) -> dict | None:
    """
    Launch test.py as a subprocess for one (checkpoint, augmentation) combination.
    Returns the parsed SUMMARY_JSON dict, or None on failure.
    """
    cmd = [
        python_exe,
        str(test_py),
        "--data_path",
        str(data_path),
        "--checkpoint_path",
        str(checkpoint_path),
        "--use_cuda",
        "true",
        "--cuda_device_num",
        str(cuda_device),
        "--augmentation_enable",
        str(augmentation_enable).lower(),
        "--aug_factor",
        "8",
        "--detailed_log",
        "false",
    ]

    aug_label = "aug(x8)" if augmentation_enable else "no_aug"
    print(f"    [{aug_label}] Running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=test_py.parent,  # test.py expects to run from its own dir
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"    [{aug_label}] TIMEOUT after {timeout}s")
        return None

    if proc.returncode != 0:
        print(f"    [{aug_label}] FAILED (exit code {proc.returncode})")
        # Print last 40 lines of stderr for diagnosis
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-40:])
        if stderr_tail:
            print(f"    [{aug_label}] stderr tail:\n{stderr_tail}")
        return None

    # test.py prints exactly one line:  SUMMARY_JSON: <compact json>
    match = re.search(r"SUMMARY_JSON:\s*(\{.*\})", proc.stdout, re.DOTALL)
    if not match:
        print(f"    [{aug_label}] WARNING: no SUMMARY_JSON line in stdout")
        return None

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        print(f"    [{aug_label}] ERROR parsing JSON: {exc}")
        return None

    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate checkpoints on TSP validation set"
    )
    parser.add_argument(
        "result_dir",
        type=str,
        help="Path to the training result directory containing checkpoint-*.pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output files (default: <result_dir>/eval_batch/)",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device number (default: 0)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=50,
        help="Starting checkpoint number (default: 50)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=500,
        help="Ending checkpoint number (default: 500)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="Step between checkpoints (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout per evaluation run in seconds (default: 900)",
    )

    args = parser.parse_args()

    # --- Resolve paths ---
    result_dir = Path(args.result_dir).resolve()
    if not result_dir.is_dir():
        print(f"Error: directory not found: {result_dir}")
        sys.exit(1)

    test_py = find_test_py(result_dir)
    data_val = find_data_val(result_dir)
    venv_python = find_venv_python(result_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = result_dir / "eval_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Result dir : {result_dir}")
    print(f"test.py    : {test_py}")
    print(f"Python     : {venv_python}")
    print(f"data/val   : {data_val}")
    print(f"Output dir : {output_dir}")
    print(f"CUDA device: {args.cuda_device}")
    print(f"Checkpoints: {list(range(args.start, args.end + 1, args.step))}")

    # --- Iterate checkpoints ---
    checkpoint_nums = list(range(args.start, args.end + 1, args.step))
    rows: list[dict] = []

    for num in checkpoint_nums:
        ckpt_path = result_dir / f"checkpoint-{num}.pt"
        if not ckpt_path.exists():
            print(f"\n[{num:>4}] SKIP — file not found: {ckpt_path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{num:>4}] Evaluating: {ckpt_path.name}")
        print(f"{'=' * 70}")

        no_aug = run_eval(
            venv_python,
            test_py,
            ckpt_path,
            data_val,
            augmentation_enable=False,
            cuda_device=args.cuda_device,
            timeout=args.timeout,
        )

        aug = run_eval(
            venv_python,
            test_py,
            ckpt_path,
            data_val,
            augmentation_enable=True,
            cuda_device=args.cuda_device,
            timeout=args.timeout,
        )

        # --- Build one row ---
        row: dict = {
            "checkpoint": num,
            "checkpoint_path": str(ckpt_path),
        }

        if no_aug is not None:
            row["no_aug_gap"] = no_aug.get("avg_no_aug_gap")
            row["no_aug_solved"] = no_aug.get("solved_instance_num")
            row["no_aug_total"] = no_aug.get("total_instance_num")
        else:
            row["no_aug_gap"] = None
            row["no_aug_solved"] = None
            row["no_aug_total"] = None

        if aug is not None:
            row["aug_gap"] = aug.get("avg_aug_gap")
            row["aug_solved"] = aug.get("solved_instance_num")
            row["aug_total"] = aug.get("total_instance_num")
        else:
            row["aug_gap"] = None
            row["aug_solved"] = None
            row["aug_total"] = None

        rows.append(row)

        # Quick per-checkpoint summary
        na_str = (
            f"{row['no_aug_gap']:.4f}%" if row["no_aug_gap"] is not None else "FAIL"
        )
        a_str = f"{row['aug_gap']:.4f}%" if row["aug_gap"] is not None else "FAIL"
        print(f"  ==> no_aug_gap: {na_str}  |  aug_gap: {a_str}")

    # --- Save outputs ---
    if not rows:
        print("\nNo checkpoints were evaluated. Exiting.")
        sys.exit(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = output_dir / f"batch_eval_{timestamp}.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nJSON report: {json_path}")

    # TSV
    tsv_path = output_dir / f"batch_eval_{timestamp}.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        headers = [
            "checkpoint",
            "no_aug_gap",
            "no_aug_solved",
            "no_aug_total",
            "aug_gap",
            "aug_solved",
            "aug_total",
            "checkpoint_path",
        ]
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")
    print(f"TSV report : {tsv_path}")

    # --- Pretty-print summary table ---
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Ckpt':<8} {'no_aug_gap':<14} {'aug_gap':<14} {'Solved (aug)':<15}")
    print(f"{'-' * 8} {'-' * 14} {'-' * 14} {'-' * 15}")
    for row in rows:
        na = f"{row['no_aug_gap']:.4f}%" if row["no_aug_gap"] is not None else "N/A"
        ag = f"{row['aug_gap']:.4f}%" if row["aug_gap"] is not None else "N/A"
        sol = (
            f"{row['aug_solved']}/{row['aug_total']}"
            if row["aug_solved"] is not None
            else "N/A"
        )
        print(f"{row['checkpoint']:<8} {na:<14} {ag:<14} {sol:<15}")

    best_row = min(
        (r for r in rows if r["aug_gap"] is not None),
        key=lambda r: r["aug_gap"],
        default=None,
    )
    if best_row is not None:
        print(
            f"\nBest checkpoint: {best_row['checkpoint']}  "
            f"(aug_gap = {best_row['aug_gap']:.4f}%)"
        )


if __name__ == "__main__":
    main()
