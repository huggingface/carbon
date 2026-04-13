"""Plot SeqQA difficulty accuracy across checkpoint steps for selected revisions.

Usage:
    uv run --directory evaluation python scripts/plot_seqqa_checkpoint_series.py \
        --dataset hf-carbon/details_hf-carbon__Qwen3-4B-Instruct-2507-SFT_private \
        --baseline-dataset hf-carbon/details_Qwen__Qwen3-4B-Instruct-2507_private \
        --version-prefix v04.00 \
        --version-prefix v05.00
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import get_dataset_split_names, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scratch" / "seqqa_checkpoint_series"
RESULTS_CONFIG = "results"
RESULT_KEYS = {
    "SeqQA_Easy": "lab_bench_seqqa_difficulty_mcf:SeqQA_Easy|0",
    "SeqQA_Medium": "lab_bench_seqqa_difficulty_mcf:SeqQA_Medium|0",
    "SeqQA_Hard": "lab_bench_seqqa_difficulty_mcf:SeqQA_Hard|0",
    "SeqQA_All": "all",
}
STEP_RE = re.compile(r"step-(\d+)")


@dataclass(frozen=True)
class RunResult:
    split: str
    revision: str
    step: int
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load LightEval detail-dataset results splits, extract SeqQA difficulty accuracies, "
            "and plot accuracy vs checkpoint step for selected version prefixes."
        )
    )
    parser.add_argument(
        "--dataset",
        default="hf-carbon/details_hf-carbon__Qwen3-4B-Instruct-2507-SFT_private",
        help="Dataset repo containing timestamped LightEval results for the checkpoint series.",
    )
    parser.add_argument(
        "--baseline-dataset",
        default="hf-carbon/details_Qwen__Qwen3-4B-Instruct-2507_private",
        help="Dataset repo used for the dashed baseline reference line.",
    )
    parser.add_argument(
        "--version-prefix",
        action="append",
        required=True,
        help="Revision prefix to include, for example v04.00 or v05.00. Repeatable.",
    )
    parser.add_argument(
        "--baseline-label",
        default="Qwen3-4B-Instruct-2507",
        help="Legend label for the dashed baseline line.",
    )
    parser.add_argument(
        "--title-override",
        action="append",
        default=[],
        metavar="VERSION_PREFIX=TITLE",
        help=(
            "Override the default figure title for a version prefix. "
            "Example: --title-override 'v04.00=Gemma4-31B Distilled (No Filter)'. Repeatable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNGs and CSV summaries will be written.",
    )
    return parser.parse_args()


def extract_step(revision: str) -> int | None:
    match = STEP_RE.search(revision)
    if match is None:
        return None
    return int(match.group(1))


def parse_results_row(row: dict) -> tuple[str | None, dict[str, float] | None]:
    config_general = json.loads(row["config_general"])
    revision = config_general.get("model_config", {}).get("revision")
    raw_results = json.loads(row["results"])

    metrics = {}
    for label, result_key in RESULT_KEYS.items():
        value = raw_results.get(result_key, {}).get("acc")
        if value is None:
            return revision, None
        metrics[label] = float(value)
    return revision, metrics


def load_latest_runs_by_revision(dataset_repo_id: str) -> dict[str, RunResult]:
    latest_runs: dict[str, RunResult] = {}
    splits = sorted(split for split in get_dataset_split_names(dataset_repo_id, RESULTS_CONFIG) if split != "latest")

    for split in splits:
        dataset = load_dataset(dataset_repo_id, RESULTS_CONFIG, split=split)
        revision, metrics = parse_results_row(dataset[0])
        if revision is None or metrics is None:
            continue

        step = extract_step(revision)
        if step is None:
            continue

        latest_runs[revision] = RunResult(split=split, revision=revision, step=step, metrics=metrics)

    return latest_runs


def load_latest_baseline_metrics(dataset_repo_id: str) -> tuple[str, dict[str, float]]:
    splits = sorted(split for split in get_dataset_split_names(dataset_repo_id, RESULTS_CONFIG) if split != "latest")
    best_revision = ""
    best_metrics: dict[str, float] | None = None

    for split in splits:
        dataset = load_dataset(dataset_repo_id, RESULTS_CONFIG, split=split)
        revision, metrics = parse_results_row(dataset[0])
        if revision is None or metrics is None:
            continue
        best_revision = revision
        best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError(
            f"Could not find a SeqQA difficulty results split in baseline dataset {dataset_repo_id!r}."
        )

    return best_revision, best_metrics


def humanize_prefix(prefix: str) -> str:
    return prefix.split(".", 1)[0]


def parse_title_overrides(raw_overrides: list[str]) -> dict[str, str]:
    parsed = {}
    for override in raw_overrides:
        version_prefix, separator, title = override.partition("=")
        if not separator or not version_prefix or not title:
            raise ValueError(
                "Each --title-override value must look like VERSION_PREFIX=TITLE, "
                f"got {override!r}."
            )
        parsed[version_prefix] = title
    return parsed


def write_summary_csv(path: Path, runs: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["revision", "split", "step", "SeqQA_Easy", "SeqQA_Medium", "SeqQA_Hard", "SeqQA_All"],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "revision": run.revision,
                    "split": run.split,
                    "step": run.step,
                    **run.metrics,
                }
            )


def plot_version_series(
    output_path: Path,
    title: str,
    version_label: str,
    runs: list[RunResult],
    baseline_label: str,
    baseline_metrics: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    figure_handles = None
    figure_labels = None

    for ax, metric_name in zip(axes.flat, RESULT_KEYS, strict=True):
        x_values = [run.step for run in runs]
        y_values = [run.metrics[metric_name] for run in runs]

        line = ax.plot(
            x_values,
            y_values,
            color="#1f77b4",
            marker="o",
            linewidth=1.75,
            label=version_label,
        )[0]
        baseline = ax.axhline(
            baseline_metrics[metric_name],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=baseline_label,
        )
        ax.set_title(metric_name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.25)
        ax.ticklabel_format(style="plain", axis="x")

        if figure_handles is None:
            figure_handles = [line, baseline]
            figure_labels = [version_label, baseline_label]

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if figure_handles is not None and figure_labels is not None:
        fig.legend(figure_handles, figure_labels, loc="lower center", ncol=2, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_by_revision = load_latest_runs_by_revision(args.dataset)
    baseline_revision, baseline_metrics = load_latest_baseline_metrics(args.baseline_dataset)
    title_overrides = parse_title_overrides(args.title_override)

    for version_prefix in args.version_prefix:
        selected_runs = sorted(
            (run for revision, run in runs_by_revision.items() if revision.startswith(version_prefix)),
            key=lambda run: run.step,
        )
        if not selected_runs:
            raise RuntimeError(
                f"No SeqQA difficulty runs found in {args.dataset!r} for version prefix {version_prefix!r}."
            )

        version_name = humanize_prefix(version_prefix)
        base_name = version_name.lower()
        csv_path = args.output_dir / f"{base_name}_summary.csv"
        png_path = args.output_dir / f"{base_name}_plot.png"
        title = title_overrides.get(
            version_prefix,
            f"SeqQA Difficulty Checkpoints ({version_name}, Qwen3-4B-Instruct-2507-SFT)",
        )

        write_summary_csv(csv_path, selected_runs)
        plot_version_series(
            output_path=png_path,
            title=title,
            version_label=version_name,
            runs=selected_runs,
            baseline_label=args.baseline_label,
            baseline_metrics=baseline_metrics,
        )

        print(
            f"{version_name}: wrote {png_path} and {csv_path} using baseline "
            f"{args.baseline_label}@{baseline_revision}"
        )


if __name__ == "__main__":
    main()
