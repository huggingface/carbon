"""Plot IRT difficulty diagnostics.

Usage:
    uv run --directory evaluation python scripts/plot_difficulty_irt.py \
        --dataset hf-carbon/seqqa-irt-difficulty \
        --subset irt_item_difficulty \
        --split train
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from huggingface_hub import HfApi

SCRATCH_ROOT = Path(__file__).resolve().parents[2] / "scratch" / "seqqa_difficulty_irt"
DIFFICULTY_COLORS = {
    "easy": "#54a24b",
    "medium": "#eeca3b",
    "hard": "#d85c27",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a published IRT item-difficulty subset, bucket items by difficulty_b, "
            "and plot difficulty_b vs gold choice length."
        )
    )
    parser.add_argument(
        "--dataset",
        default="hf-carbon/seqqa-irt-difficulty",
        help="HF dataset repo id.",
    )
    parser.add_argument(
        "--subset",
        default="irt_item_difficulty",
        help="HF dataset config/subset name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name used to resolve parquet shard names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the threshold-bucket gold-choice-length PNG plot.",
    )
    parser.add_argument(
        "--percentile-output",
        type=Path,
        default=None,
        help="Where to save the percentile-bucket gold-choice-length scatter plot.",
    )
    parser.add_argument(
        "--histogram-output",
        type=Path,
        default=None,
        help="Where to save the difficulty_b histogram.",
    )
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=None,
        help="Items with difficulty_b above this threshold are labeled hard. Defaults to the 67th percentile.",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=None,
        help="Items with difficulty_b above this threshold are labeled medium. Defaults to the 33rd percentile.",
    )
    args = parser.parse_args()
    default_outputs = make_default_output_paths(args.dataset, args.subset, args.split)
    if args.output is None:
        args.output = default_outputs["output"]
    if args.percentile_output is None:
        args.percentile_output = default_outputs["percentile_output"]
    if args.histogram_output is None:
        args.histogram_output = default_outputs["histogram_output"]
    return args


def repo_parts(dataset_repo_id: str) -> tuple[str, str]:
    dataset_org, separator, dataset_name = dataset_repo_id.partition("/")
    if not separator or not dataset_org or not dataset_name:
        raise ValueError(
            f"Expected a dataset repo id like org/name, got: {dataset_repo_id!r}"
        )
    return dataset_org, dataset_name


def make_default_output_paths(dataset_repo_id: str, subset: str, split: str) -> dict[str, Path]:
    dataset_org, dataset_name = repo_parts(dataset_repo_id)
    base_dir = SCRATCH_ROOT / dataset_org / dataset_name / subset / split
    return {
        "output": base_dir / "threshold.png",
        "percentile_output": base_dir / "percentile.png",
        "histogram_output": base_dir / "dist.png",
    }


def list_split_parquet_paths(dataset_repo_id: str, subset: str, split: str) -> list[str]:
    api = HfApi()
    prefix = f"{subset}/{split}-"
    parquet_paths = sorted(
        path
        for path in api.list_repo_files(dataset_repo_id, repo_type="dataset")
        if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet shards found for {dataset_repo_id}/{subset}/{split}"
        )
    return [f"hf://datasets/{dataset_repo_id}/{path}" for path in parquet_paths]


def load_irt_subset(dataset_repo_id: str, subset: str, split: str) -> Dataset:
    return Dataset.from_parquet(list_split_parquet_paths(dataset_repo_id, subset, split))


def difficulty_bucket(difficulty_b: float, hard_threshold: float, medium_threshold: float) -> str:
    if difficulty_b > hard_threshold:
        return "hard"
    if difficulty_b > medium_threshold:
        return "medium"
    return "easy"


def parse_options(raw_options) -> list[str]:
    if isinstance(raw_options, list):
        return [str(option) for option in raw_options]
    if isinstance(raw_options, str):
        parsed = json.loads(raw_options)
        if not isinstance(parsed, list):
            raise ValueError("Expected `options` to decode to a JSON list.")
        return [str(option) for option in parsed]
    raise TypeError(f"Unsupported options type: {type(raw_options).__name__}")


def extract_options_from_query(question_text: str) -> list[str]:
    option_matches = re.findall(r"^\s*[A-Z]\.\s*(.+?)\s*$", question_text, flags=re.MULTILINE)
    return [match.strip() for match in option_matches]


def resolve_option_texts(question_text: str, options: list[str]) -> list[str]:
    stripped_options = [option.strip() for option in options]
    if stripped_options and all(len(option) <= 2 for option in stripped_options):
        parsed_options = extract_options_from_query(question_text)
        if len(parsed_options) == len(options):
            return parsed_options
    return stripped_options


def build_rows(dataset: Dataset) -> tuple[list[dict], int]:
    rows = []
    skipped = 0

    for example in dataset:
        try:
            raw_options = example.get("options", example.get("choices"))
            options = parse_options(raw_options)
            answer_index_value = example.get("answer_index", example.get("gold_index"))
            answer_index = int(answer_index_value)
            difficulty_b = float(example["difficulty_b"])
            mean_accuracy = float(example["mean_accuracy"])
            question_text = str(example.get("question", example.get("query", ""))).strip()
            option_texts = resolve_option_texts(question_text, options)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            skipped += 1
            continue

        if answer_index < 0 or answer_index >= len(option_texts):
            skipped += 1
            continue

        rows.append(
            {
                "gold_choice_text_len": len(option_texts[answer_index]),
                "question_text_len": len(question_text),
                "difficulty_b": difficulty_b,
                "mean_accuracy": mean_accuracy,
            }
        )

    return rows, skipped


def resolve_thresholds(
    rows: list[dict], hard_threshold: float | None, medium_threshold: float | None
) -> tuple[float, float]:
    if not rows:
        raise RuntimeError("No rows available to derive difficulty thresholds.")

    difficulty_values = np.asarray([row["difficulty_b"] for row in rows], dtype=float)
    resolved_hard = (
        float(np.quantile(difficulty_values, 2 / 3))
        if hard_threshold is None
        else hard_threshold
    )
    resolved_medium = (
        float(np.quantile(difficulty_values, 1 / 3))
        if medium_threshold is None
        else medium_threshold
    )
    if resolved_hard <= resolved_medium:
        raise ValueError("--hard-threshold must be greater than --medium-threshold")
    return resolved_hard, resolved_medium


def assign_threshold_difficulties(
    rows: list[dict], hard_threshold: float, medium_threshold: float
) -> None:
    for row in rows:
        row["difficulty"] = difficulty_bucket(
            row["difficulty_b"], hard_threshold, medium_threshold
        )


def assign_percentile_difficulties(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    if not rows:
        return {}

    sorted_indices = np.argsort([row["difficulty_b"] for row in rows])
    buckets = np.array_split(sorted_indices, 3)
    labels = ("easy", "medium", "hard")
    summary = {}

    for label, bucket_indices in zip(labels, buckets, strict=True):
        bucket_values = [rows[int(index)]["difficulty_b"] for index in bucket_indices]
        for index in bucket_indices:
            rows[int(index)]["percentile_difficulty"] = label

        if bucket_values:
            summary[label] = {
                "count": len(bucket_values),
                "min_difficulty_b": float(min(bucket_values)),
                "max_difficulty_b": float(max(bucket_values)),
            }
        else:
            summary[label] = {
                "count": 0,
                "min_difficulty_b": float("nan"),
                "max_difficulty_b": float("nan"),
            }

    return summary


def compute_accuracy_by_difficulty(
    rows: list[dict], difficulty_key: str = "difficulty"
) -> dict[str, dict[str, float | int | None]]:
    stats = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in rows if row.get(difficulty_key) == difficulty]
        mean_accuracy = (
            float(np.mean([row["mean_accuracy"] for row in subset])) if subset else None
        )
        stats[difficulty] = {
            "count": len(subset),
            "accuracy": mean_accuracy,
        }
    return stats


def format_accuracy(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def plot_rows(
    rows: list[dict],
    output_path: Path,
    dataset_label: str,
    hard_threshold: float,
    medium_threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in rows if row["difficulty"] == difficulty]
        if not subset:
            continue
        ax.scatter(
            [row["gold_choice_text_len"] for row in subset],
            [row["difficulty_b"] for row in subset],
            s=40,
            alpha=0.8,
            color=DIFFICULTY_COLORS[difficulty],
            edgecolors="none",
            label=f"{difficulty} (n={len(subset)})",
        )

    ax.axhline(medium_threshold, color="#6c6c6c", linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(hard_threshold, color="#6c6c6c", linestyle="--", linewidth=1, alpha=0.8)
    difficulty_values = np.asarray([row["difficulty_b"] for row in rows], dtype=float)
    y_min = float(np.min(difficulty_values))
    y_max = float(np.max(difficulty_values))
    y_pad = max((y_max - y_min) * 0.03, 0.1)
    label_transform = ax.get_yaxis_transform()
    ax.text(
        0.98,
        hard_threshold + y_pad,
        f"hard (> {hard_threshold:.3f})",
        color=DIFFICULTY_COLORS["hard"],
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
        transform=label_transform,
    )
    ax.text(
        0.98,
        medium_threshold + y_pad,
        f"medium (> {medium_threshold:.3f})",
        color=DIFFICULTY_COLORS["medium"],
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
        transform=label_transform,
    )
    ax.text(
        0.98,
        y_min + y_pad,
        f"easy (<= {medium_threshold:.3f})",
        color=DIFFICULTY_COLORS["easy"],
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
        transform=label_transform,
    )
    ax.set_xlabel("Gold choice length (characters)")
    ax.set_ylabel("IRT difficulty")
    ax.set_title(f"{dataset_label} Difficulty vs Answer Length")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_percentile_rows(
    rows: list[dict],
    output_path: Path,
    dataset_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in rows if row.get("percentile_difficulty") == difficulty]
        if not subset:
            continue
        ax.scatter(
            [row["gold_choice_text_len"] for row in subset],
            [row["difficulty_b"] for row in subset],
            s=40,
            alpha=0.8,
            color=DIFFICULTY_COLORS[difficulty],
            edgecolors="none",
            label=f"{difficulty} (n={len(subset)})",
    )

    ax.set_xlabel("Gold choice length (characters)")
    ax.set_ylabel("IRT difficulty")
    ax.set_title(f"{dataset_label} Difficulty vs Answer Length")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_histogram(
    rows: list[dict],
    output_path: Path,
    dataset_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    difficulty_values = np.asarray([row["difficulty_b"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        difficulty_values,
        bins=30,
        color="#4c78a8",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("IRT difficulty")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset_label} Difficulty Distribution")
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = load_irt_subset(args.dataset, args.subset, args.split)
    rows, skipped = build_rows(dataset)
    if not rows:
        raise RuntimeError("No rows could be parsed from the requested dataset subset.")

    hard_threshold, medium_threshold = resolve_thresholds(
        rows, args.hard_threshold, args.medium_threshold
    )
    assign_threshold_difficulties(rows, hard_threshold, medium_threshold)
    difficulty_stats = compute_accuracy_by_difficulty(rows)
    percentile_summary = assign_percentile_difficulties(rows)
    percentile_stats = compute_accuracy_by_difficulty(rows, difficulty_key="percentile_difficulty")
    overall_accuracy = float(np.mean([row["mean_accuracy"] for row in rows]))
    question_lengths = [row["question_text_len"] for row in rows]
    choice_lengths = [row["gold_choice_text_len"] for row in rows]
    if np.std(question_lengths) == 0 or np.std(choice_lengths) == 0:
        length_correlation = None
    else:
        length_correlation = float(np.corrcoef(question_lengths, choice_lengths)[0, 1])
    dataset_label = f"{args.dataset}/{args.subset}/{args.split}"

    plot_rows(
        rows,
        args.output,
        dataset_label,
        hard_threshold,
        medium_threshold,
    )
    plot_percentile_rows(
        rows,
        args.percentile_output,
        dataset_label,
    )
    plot_histogram(
        rows,
        args.histogram_output,
        dataset_label,
    )

    counts = Counter(row["difficulty"] for row in rows)
    print(f"Loaded {len(rows)} rows from {args.dataset}/{args.subset}/{args.split}")
    print(f"Skipped {skipped} rows")
    print(
        "Difficulty counts: "
        + ", ".join(f"{label}={counts.get(label, 0)}" for label in ("easy", "medium", "hard"))
    )
    print(
        f"Difficulty thresholds: hard>{hard_threshold:.3f}, "
        f"medium>{medium_threshold:.3f}"
    )
    print(f"Overall mean accuracy: {format_accuracy(overall_accuracy)}")
    print(
        "Mean accuracy by difficulty: "
        + ", ".join(
            (
                f"{label}={format_accuracy(difficulty_stats[label]['accuracy'])}"
                f" (n={difficulty_stats[label]['count']})"
            )
            for label in ("easy", "medium", "hard")
        )
    )
    print(
        "Percentile-bucket mean accuracy: "
        + ", ".join(
            (
                f"{label}={format_accuracy(percentile_stats[label]['accuracy'])}"
                f" (n={percentile_stats[label]['count']}, "
                f"difficulty_b_range=[{percentile_summary[label]['min_difficulty_b']:.3f}, "
                f"{percentile_summary[label]['max_difficulty_b']:.3f}])"
            )
            for label in ("easy", "medium", "hard")
        )
    )
    print(
        "Question-length vs gold-choice-length correlation: "
        + ("n/a" if length_correlation is None else f"{length_correlation:.3f}")
    )
    print(f"Saved plot to {args.output}")
    print(f"Saved percentile plot to {args.percentile_output}")
    print(f"Saved histogram to {args.histogram_output}")


if __name__ == "__main__":
    main()
