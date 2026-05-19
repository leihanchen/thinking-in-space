#!/usr/bin/env python3
"""Plot VSI-Bench metrics vs epoch for runs."""

import argparse
import csv
import json
import math
import os
import re
import sys

import matplotlib.pyplot as plt


STEPS_PER_EPOCH = 426


def parse_epoch_from_name(name: str) -> float | None:
    """Extract epoch count from a directory name."""
    epochs_match = re.search(r"(\d+)\s*epochs?", name, re.IGNORECASE)
    if epochs_match:
        return float(epochs_match.group(1))

    steps_match = re.search(r"(\d+)\s*steps", name, re.IGNORECASE)
    if steps_match:
        steps = float(steps_match.group(1))
        return steps / STEPS_PER_EPOCH

    return None


def find_candidate_dirs(logs_root: str, needle: str) -> list[str]:
    """Find subdirectories whose name contains the needle."""
    matches = set()
    for root, dirs, _files in os.walk(logs_root):
        for dirname in dirs:
            if needle in dirname:
                matches.add(os.path.join(root, dirname))
    return sorted(matches)


def load_metrics(results_path: str) -> dict[str, float]:
    """Load and filter VSI-Bench metrics from a results.json file."""
    with open(results_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    metrics = data["results"]["vsibench"]["vsibench_score,none"]
    filtered = {}
    for key, value in metrics.items():
        if key in {"tabulated_keys", "tabulated_results"}:
            continue
        try:
            filtered[key] = float(value)
        except (TypeError, ValueError):
            print(
                f"Warning: non-numeric value for {key} in {results_path}",
                file=sys.stderr,
            )
    return filtered


def build_metric_order(metric_keys: set[str]) -> list[str]:
    """Order metrics with overall first, then alphabetical."""
    ordered = []
    if "overall" in metric_keys:
        ordered.append("overall")
        metric_keys = set(metric_keys)
        metric_keys.remove("overall")
    ordered.extend(sorted(metric_keys))
    return ordered


def plot_metrics(epochs: list[float], metrics_by_epoch: dict[float, dict[str, float]], output_path: str) -> None:
    """Render a grid of subplots (one per metric)."""
    all_metric_keys = set()
    for metrics in metrics_by_epoch.values():
        all_metric_keys.update(metrics.keys())

    metric_order = build_metric_order(all_metric_keys)
    if not metric_order:
        raise ValueError("No metrics found to plot.")

    n_metrics = len(metric_order)
    ncols = 2 if n_metrics <= 4 else 3
    nrows = int(math.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.4 * nrows),
        sharex=True,
    )
    axes_list = axes if isinstance(axes, (list, tuple)) else [axes]
    if hasattr(axes, "ravel"):
        axes_list = axes.ravel().tolist()

    for idx, metric in enumerate(metric_order):
        ax = axes_list[idx]
        xs = []
        ys = []
        for epoch in epochs:
            value = metrics_by_epoch[epoch].get(metric)
            if value is None:
                continue
            xs.append(epoch)
            ys.append(value)

        ax.plot(xs, ys, marker="o", linestyle="-")
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    for idx in range(n_metrics, len(axes_list)):
        axes_list[idx].axis("off")

    fig.suptitle("VSI-Bench metrics vs epoch (videor1)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)


def write_csv(
    runs: list[tuple[float, str, dict[str, float]]],
    output_path: str,
) -> None:
    """Write CSV with one row per subdirectory."""
    if not runs:
        raise ValueError("No runs to write.")

    all_metric_keys = set()
    for _epoch, _dir, metrics in runs:
        all_metric_keys.update(metrics.keys())

    metric_order = build_metric_order(all_metric_keys)
    fieldnames = ["epoch", "directory"] + metric_order

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, directory, metrics in runs:
            row = {"epoch": epoch, "directory": os.path.basename(directory)}
            row.update(metrics)
            writer.writerow(row)



def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(description="Plot VSI-Bench metrics vs epoch.")
    parser.add_argument(
        "--logs-root",
        default=os.path.join(repo_root, "logs"),
        help="Path to the logs root directory.",
    )
    parser.add_argument(
        "--needle",
        default="videor1-",
        help="Substring to match against subdirectory names.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "videor1_vsibench_metrics.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--csv-output",
        default=os.path.join(script_dir, "videor1_vsibench_metrics.csv"),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    candidate_dirs = find_candidate_dirs(args.logs_root, args.needle)
    if not candidate_dirs:
        print("No matching directories found.", file=sys.stderr)
        return 1

    epoch_to_metrics: dict[float, dict[str, float]] = {}
    all_runs: list[tuple[float, str, dict[str, float]]] = []

    for directory in candidate_dirs:
        epoch = parse_epoch_from_name(os.path.basename(directory))
        if epoch is None:
            print(f"Warning: no epoch info in {directory}", file=sys.stderr)
            continue

        results_path = os.path.join(directory, "results.json")
        if not os.path.isfile(results_path):
            print(f"Warning: missing results.json in {directory}", file=sys.stderr)
            continue

        try:
            metrics = load_metrics(results_path)
        except (KeyError, json.JSONDecodeError) as exc:
            print(f"Warning: failed to parse {results_path}: {exc}", file=sys.stderr)
            continue

        all_runs.append((epoch, directory, metrics))
        epoch_to_metrics[epoch] = metrics

    if not all_runs:
        print("No valid results.json files found.", file=sys.stderr)
        return 1

    epochs_sorted = sorted(epoch_to_metrics.keys())
    plot_metrics(epochs_sorted, epoch_to_metrics, args.output)
    write_csv(all_runs, args.csv_output)

    print(
        "Plotted",
        len(epochs_sorted),
        "epochs from",
        len(all_runs),
        "directories. Output:",
        args.output,
        "and",
        args.csv_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
