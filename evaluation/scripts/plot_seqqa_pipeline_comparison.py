"""Plot the SeqQA pipeline comparison grouped bar chart.

Usage:
    uv run --directory evaluation python scripts/plot_seqqa_pipeline_comparison.py
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import get_dataset_split_names, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "scratch" / "seqqa_pipeline_comparison.png"
DEFAULT_SUMMARY = REPO_ROOT / "scratch" / "seqqa_pipeline_comparison.csv"
RESULTS_CONFIG = "results"
METRIC_KEYS = {
    "SeqQA_Easy": "lab_bench_seqqa_difficulty_mcf:SeqQA_Easy|0",
    "SeqQA_Medium": "lab_bench_seqqa_difficulty_mcf:SeqQA_Medium|0",
    "SeqQA_Hard": "lab_bench_seqqa_difficulty_mcf:SeqQA_Hard|0",
    "SeqQA_All": "all",
}
DEFAULT_SERIES = [
    "Gemma4-31B distilled=v01.02-step-000000759",
    "Self-distilled=v02.02-step-000000220",
    "Mixed=v03.00-step-000001448",
    "Gemma4-31B distilled (unfiltered)=v04.00-step-000002709",
    "Self-distilled (unfiltered)=v05.00-step-000001518",
]
SERIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#7da6d9", "#ffb86b"]


@dataclass(frozen=True)
class RunResult:
    split: str
    revision: str
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load SeqQA difficulty results for selected revisions from a details dataset and "
            "render the grouped pipeline comparison chart."
        )
    )
    parser.add_argument(
        "--dataset",
        default="hf-carbon/details_hf-carbon__Qwen3-4B-Instruct-2507-SFT_private",
        help="Dataset repo containing timestamped LightEval results.",
    )
    parser.add_argument(
        "--baseline-dataset",
        default="hf-carbon/details_Qwen__Qwen3-4B-Instruct-2507_private",
        help="Dataset repo used for the dashed baseline reference.",
    )
    parser.add_argument(
        "--series",
        action="append",
        default=None,
        metavar="LABEL=REVISION",
        help=(
            "Series to include in the grouped bar plot. Repeatable. "
            "Defaults to the filtered v01/v02/v03 series plus unfiltered v04/v05."
        ),
    )
    parser.add_argument(
        "--title",
        default="SeqQA Data Pipeline Comparison",
        help="Figure title.",
    )
    parser.add_argument(
        "--baseline-label",
        default="Qwen3-4B-Instruct-2507",
        help="Legend label for the dashed baseline line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="CSV summary output path.",
    )
    args = parser.parse_args()
    if args.series is None:
        args.series = list(DEFAULT_SERIES)
    return args


def parse_series_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs:
        label, separator, revision = spec.partition("=")
        if not separator or not label or not revision:
            raise ValueError(
                f"Expected series specification LABEL=REVISION, got {spec!r}."
            )
        parsed.append((label, revision))
    return parsed


def parse_results_row(row: dict) -> tuple[str | None, dict[str, float] | None]:
    config_general = json.loads(row["config_general"])
    revision = config_general.get("model_config", {}).get("revision")
    raw_results = json.loads(row["results"])
    metrics = {}

    for metric_name, result_key in METRIC_KEYS.items():
        value = raw_results.get(result_key, {}).get("acc")
        if value is None:
            return revision, None
        metrics[metric_name] = float(value)

    return revision, metrics


def load_latest_runs_by_revision(dataset_repo_id: str) -> dict[str, RunResult]:
    latest_runs: dict[str, RunResult] = {}
    splits = sorted(split for split in get_dataset_split_names(dataset_repo_id, RESULTS_CONFIG) if split != "latest")

    for split in splits:
        row = load_dataset(dataset_repo_id, RESULTS_CONFIG, split=split)[0]
        revision, metrics = parse_results_row(row)
        if revision is None or metrics is None:
            continue
        latest_runs[revision] = RunResult(split=split, revision=revision, metrics=metrics)

    return latest_runs


def load_latest_baseline_metrics(dataset_repo_id: str) -> tuple[str, dict[str, float]]:
    latest_revision = ""
    latest_metrics: dict[str, float] | None = None
    splits = sorted(split for split in get_dataset_split_names(dataset_repo_id, RESULTS_CONFIG) if split != "latest")

    for split in splits:
        row = load_dataset(dataset_repo_id, RESULTS_CONFIG, split=split)[0]
        revision, metrics = parse_results_row(row)
        if metrics is None:
            continue
        latest_revision = revision or ""
        latest_metrics = metrics

    if latest_metrics is None:
        raise RuntimeError(
            f"Could not find SeqQA difficulty metrics in baseline dataset {dataset_repo_id!r}."
        )

    return latest_revision, latest_metrics


def write_summary_csv(
    path: Path,
    series_specs: list[tuple[str, str]],
    runs_by_revision: dict[str, RunResult],
    baseline_label: str,
    baseline_revision: str,
    baseline_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "revision", "split", *METRIC_KEYS]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for label, revision in series_specs:
            run = runs_by_revision[revision]
            writer.writerow(
                {
                    "label": label,
                    "revision": revision,
                    "split": run.split,
                    **run.metrics,
                }
            )

        writer.writerow(
            {
                "label": baseline_label,
                "revision": baseline_revision,
                "split": "latest-compatible",
                **baseline_metrics,
            }
        )


def plot_grouped_bars(
    *,
    output_path: Path,
    title: str,
    series_specs: list[tuple[str, str]],
    runs_by_revision: dict[str, RunResult],
    baseline_label: str,
    baseline_metrics: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    categories = list(METRIC_KEYS)
    x = np.arange(len(categories), dtype=float)
    bar_width = 0.15
    series_count = len(series_specs)
    offsets = (np.arange(series_count) - (series_count - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    bar_containers = []

    for index, (label, revision) in enumerate(series_specs):
        metrics = runs_by_revision[revision].metrics
        heights = [metrics[category] for category in categories]
        bars = ax.bar(
            x + offsets[index],
            heights,
            width=bar_width,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            label=label,
        )
        bar_containers.append(bars)

    for bars in bar_containers:
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.008,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    half_span = max(0.45, (series_count * bar_width) / 2 + 0.03)
    for index, category in enumerate(categories):
        ax.plot(
            [x[index] - half_span, x[index] + half_span],
            [baseline_metrics[category], baseline_metrics[category]],
            color="gray",
            linestyle="--",
            linewidth=2,
        )

    ax.plot([], [], color="gray", linestyle="--", linewidth=2, label=baseline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=21, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    series_specs = parse_series_specs(args.series)
    runs_by_revision = load_latest_runs_by_revision(args.dataset)
    baseline_revision, baseline_metrics = load_latest_baseline_metrics(args.baseline_dataset)

    missing_revisions = [revision for _, revision in series_specs if revision not in runs_by_revision]
    if missing_revisions:
        raise RuntimeError(
            f"Missing requested revisions in {args.dataset!r}: {', '.join(missing_revisions)}"
        )

    write_summary_csv(
        path=args.summary_output,
        series_specs=series_specs,
        runs_by_revision=runs_by_revision,
        baseline_label=args.baseline_label,
        baseline_revision=baseline_revision,
        baseline_metrics=baseline_metrics,
    )
    plot_grouped_bars(
        output_path=args.output,
        title=args.title,
        series_specs=series_specs,
        runs_by_revision=runs_by_revision,
        baseline_label=args.baseline_label,
        baseline_metrics=baseline_metrics,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
