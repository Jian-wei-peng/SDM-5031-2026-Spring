import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare a new validation result with baseline JSON.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    new = load_json(args.new)

    baseline_by_name = {
        name: {
            "size": baseline["problem_size"][idx],
            "aug_gap": baseline["aug_gap"][idx],
            "aug_score": baseline["aug_score"][idx],
        }
        for idx, name in enumerate(baseline["instances"])
    }
    new_by_name = {
        name: {
            "size": new["problem_size"][idx],
            "aug_gap": new["aug_gap"][idx],
            "aug_score": new["aug_score"][idx],
        }
        for idx, name in enumerate(new["instances"])
    }

    lines = []
    lines.append("Baseline avg_aug_gap: {:.6f}%".format(baseline["avg_aug_gap"]))
    lines.append("New      avg_aug_gap: {:.6f}%".format(new["avg_aug_gap"]))
    lines.append("Delta                 : {:+.6f}%".format(new["avg_aug_gap"] - baseline["avg_aug_gap"]))
    lines.append("")
    lines.append("| instance | size | baseline aug_gap | new aug_gap | delta | better |")
    lines.append("|---|---:|---:|---:|---:|---|")

    better_count = 0
    common_names = [name for name in baseline["instances"] if name in new_by_name]
    for name in common_names:
        base_item = baseline_by_name[name]
        new_item = new_by_name[name]
        delta = new_item["aug_gap"] - base_item["aug_gap"]
        better = delta < 0
        better_count += int(better)
        lines.append(
            "| {} | {} | {:.6f}% | {:.6f}% | {:+.6f}% | {} |".format(
                name,
                base_item["size"],
                base_item["aug_gap"],
                new_item["aug_gap"],
                delta,
                "yes" if better else "no",
            )
        )

    lines.append("")
    lines.append("Better-than-baseline count: {}/{}".format(better_count, len(common_names)))
    lines.append("Meets 70% instance target: {}".format("yes" if better_count >= 0.7 * len(common_names) else "no"))

    text = "\n".join(lines)
    print(text)

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
