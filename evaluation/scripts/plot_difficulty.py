"""Plot SeqQA difficulty diagnostics.

Usage:
    uv run --directory evaluation python scripts/plot_difficulty.py \
        --dataset hf-carbon/details_Qwen__Qwen3-4B-Base_private \
        --config lab_bench_seqqa_mcf_all_0 \
        --split latest
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

SCRATCH_ROOT = Path(__file__).resolve().parents[2] / "scratch" / "seqqa_difficulty"
OPTION_LINE_RE = re.compile(r"^\s*([A-Z])\.\s*(.*?)\s*$")
QUESTION_PREFIX = "Question:"
DIFFICULTY_COLORS = {
    "easy": "#54a24b",
    "medium": "#eeca3b",
    "hard": "#d85c27",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a LAB-Bench SeqQA details subset, compute choice-normalized p(correct) from "
            "answer-token logprobs, bucket difficulty, and plot p(correct) vs gold choice length."
        )
    )
    parser.add_argument(
        "--dataset",
        default="hf-carbon/details_Qwen__Qwen3-4B-Base_private",
        help="HF dataset repo id.",
    )
    parser.add_argument(
        "--config",
        default="lab_bench_seqqa_mcf_all_0",
        help="HF dataset config name.",
    )
    parser.add_argument(
        "--split",
        default="latest",
        help="HF dataset split name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the gold-choice-length PNG plot. Defaults under scratch/seqqa_difficulty/{org}/{model_name}.",
    )
    parser.add_argument(
        "--percentile-output",
        type=Path,
        default=None,
        help="Where to save the percentile-bucket gold-choice-length scatter plot. Defaults under scratch/seqqa_difficulty/{org}/{model_name}.",
    )
    parser.add_argument(
        "--histogram-output",
        type=Path,
        default=None,
        help="Where to save the p(correct) histogram. Defaults under scratch/seqqa_difficulty/{org}/{model_name}.",
    )
    parser.add_argument(
        "--easy-threshold",
        type=float,
        default=0.7,
        help="Examples with p(correct) above this threshold are labeled easy.",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.3,
        help="Examples with p(correct) above this threshold are labeled medium.",
    )
    args = parser.parse_args()
    default_outputs = make_default_output_paths(args.dataset)
    if args.output is None:
        args.output = default_outputs["output"]
    if args.percentile_output is None:
        args.percentile_output = default_outputs["percentile_output"]
    if args.histogram_output is None:
        args.histogram_output = default_outputs["histogram_output"]
    return args


def dataset_model_parts(dataset_repo_id: str) -> tuple[str, str]:
    _, _, repo_name = dataset_repo_id.partition("/")
    if not repo_name:
        raise ValueError(f"Expected a dataset repo id like org/name, got: {dataset_repo_id!r}")

    model_stub = repo_name
    if model_stub.startswith("details_"):
        model_stub = model_stub[len("details_") :]
    for suffix in ("_private", "_public"):
        if model_stub.endswith(suffix):
            model_stub = model_stub[: -len(suffix)]
            break

    model_org, separator, model_name = model_stub.partition("__")
    if not separator or not model_org or not model_name:
        raise ValueError(
            "Expected dataset repo name to look like details_{org}__{model_name}[_private|_public], "
            f"got: {dataset_repo_id!r}"
        )
    return model_org, model_name


def make_default_output_paths(dataset_repo_id: str) -> dict[str, Path]:
    model_org, model_name = dataset_model_parts(dataset_repo_id)
    base_dir = SCRATCH_ROOT / model_org / model_name
    return {
        "output": base_dir / "threshold.png",
        "percentile_output": base_dir / "percentile.png",
        "histogram_output": base_dir / "dist.png",
    }


def normalize_choice_probs(logprobs: list[float]) -> np.ndarray:
    values = np.asarray(logprobs, dtype=float)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def difficulty_bucket(p_correct: float, easy_threshold: float, medium_threshold: float) -> str:
    if p_correct > easy_threshold:
        return "easy"
    if p_correct > medium_threshold:
        return "medium"
    return "hard"


def parse_choice_texts(query: str, raw_choices: list[str]) -> list[str]:
    expected_labels = [choice.strip() for choice in raw_choices]
    parsed = {}
    for line in query.splitlines():
        match = OPTION_LINE_RE.match(line)
        if not match:
            continue
        label, text = match.groups()
        parsed[label] = text.strip()

    missing_labels = [label for label in expected_labels if label not in parsed]
    if missing_labels:
        raise ValueError(f"missing choice text for labels: {missing_labels}")

    return [parsed[label] for label in expected_labels]


def parse_question_text_before_qmark(query: str) -> str:
    first_line = query.splitlines()[0].strip()
    if first_line.startswith(QUESTION_PREFIX):
        first_line = first_line[len(QUESTION_PREFIX) :].strip()

    if "?" in first_line:
        return first_line.split("?", 1)[0].strip()

    return first_line.strip()


def build_rows(dataset, easy_threshold: float, medium_threshold: float) -> tuple[list[dict], int]:
    rows = []
    skipped = 0

    for example in dataset:
        doc = example["doc"]
        metric = example["metric"]
        model_response = example["model_response"]
        raw_choices = doc.get("choices") or []
        logprobs = model_response.get("logprobs") or []
        gold_index = doc.get("gold_index")

        if len(raw_choices) < 2 or len(logprobs) != len(raw_choices) or gold_index is None:
            skipped += 1
            continue

        if isinstance(gold_index, list):
            if len(gold_index) != 1:
                skipped += 1
                continue
            gold_index = gold_index[0]

        try:
            gold_index = int(gold_index)
            choice_texts = parse_choice_texts(doc["query"], raw_choices)
            question_text = parse_question_text_before_qmark(doc["query"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        if gold_index < 0 or gold_index >= len(choice_texts):
            skipped += 1
            continue

        probs = normalize_choice_probs(logprobs)
        gold_choice_text_len = len(choice_texts[gold_index].strip())
        p_correct = float(probs[gold_index])
        is_correct = None
        if isinstance(metric, dict) and metric.get("acc") is not None:
            is_correct = int(metric["acc"])
        rows.append(
            {
                "gold_choice_text_len": gold_choice_text_len,
                "question_text_len": len(question_text),
                "p_correct": p_correct,
                "difficulty": difficulty_bucket(p_correct, easy_threshold, medium_threshold),
                "is_correct": is_correct,
            }
        )

    return rows, skipped


def assign_percentile_difficulties(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    if not rows:
        return {}

    sorted_indices = np.argsort([row["p_correct"] for row in rows])
    buckets = np.array_split(sorted_indices, 3)
    labels = ("hard", "medium", "easy")
    summary = {}

    for label, bucket_indices in zip(labels, buckets, strict=True):
        bucket_values = [rows[int(index)]["p_correct"] for index in bucket_indices]
        for index in bucket_indices:
            rows[int(index)]["percentile_difficulty"] = label

        if bucket_values:
            summary[label] = {
                "count": len(bucket_values),
                "min_p_correct": float(min(bucket_values)),
                "max_p_correct": float(max(bucket_values)),
            }
        else:
            summary[label] = {
                "count": 0,
                "min_p_correct": float("nan"),
                "max_p_correct": float("nan"),
            }

    return summary


def compute_accuracy_by_difficulty(
    rows: list[dict], difficulty_key: str = "difficulty"
) -> dict[str, dict[str, float | int | None]]:
    stats = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in rows if row.get(difficulty_key) == difficulty]
        correct_rows = [row["is_correct"] for row in subset if row["is_correct"] is not None]
        accuracy = None
        if correct_rows:
            accuracy = float(sum(correct_rows) / len(correct_rows))
        stats[difficulty] = {
            "count": len(subset),
            "accuracy": accuracy,
        }
    return stats


def format_accuracy(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def plot_rows(
    rows: list[dict],
    output_path: Path,
    model_name: str,
    easy_threshold: float,
    medium_threshold: float,
    difficulty_stats: dict[str, dict[str, float | int | None]],
    overall_accuracy: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in rows if row["difficulty"] == difficulty]
        if not subset:
            continue
        ax.scatter(
            [row["gold_choice_text_len"] for row in subset],
            [row["p_correct"] for row in subset],
            s=40,
            alpha=0.8,
            color=DIFFICULTY_COLORS[difficulty],
            edgecolors="none",
            label=(
                f"{difficulty} "
                f"(n={len(subset)}, acc={format_accuracy(difficulty_stats[difficulty]['accuracy'])})"
            ),
        )

    ax.axhline(medium_threshold, color="#6c6c6c", linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(easy_threshold, color="#6c6c6c", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("Gold choice length (characters)")
    ax.set_ylabel("Choice-normalized p(correct)")
    ax.set_ylim(-0.02, 1.02)
    label_transform = ax.get_yaxis_transform()
    ax.text(
        0.98,
        min(easy_threshold + 0.03, 0.99),
        f"easy ($>$ {easy_threshold:.2f})",
        color=DIFFICULTY_COLORS["easy"],
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
        transform=label_transform,
    )
    ax.text(
        0.98,
        min(medium_threshold + 0.03, easy_threshold - 0.03),
        f"medium ($>$ {medium_threshold:.2f})",
        color=DIFFICULTY_COLORS["medium"],
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
        transform=label_transform,
    )
    ax.text(
        0.98,
        0.01,
        f"hard ($\\leq$ {medium_threshold:.2f})",
        color=DIFFICULTY_COLORS["hard"],
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
        transform=label_transform,
    )
    ax.set_title(
        f"SeqQA Difficulty for {model_name} "
        f"(overall accuracy = {format_accuracy(overall_accuracy)})"
    )
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_percentile_rows(
    rows: list[dict],
    output_path: Path,
    model_name: str,
    percentile_stats: dict[str, dict[str, float | int | None]],
    overall_accuracy: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in rows if row.get("percentile_difficulty") == difficulty]
        if not subset:
            continue
        ax.scatter(
            [row["gold_choice_text_len"] for row in subset],
            [row["p_correct"] for row in subset],
            s=40,
            alpha=0.8,
            color=DIFFICULTY_COLORS[difficulty],
            edgecolors="none",
            label=(
                f"{difficulty} "
                f"(n={len(subset)}, acc={format_accuracy(percentile_stats[difficulty]['accuracy'])})"
            ),
        )

    ax.set_xlabel("Gold choice length (characters)")
    ax.set_ylabel("Choice-normalized p(correct)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"SeqQA Difficulty for {model_name} "
        f"(overall accuracy = {format_accuracy(overall_accuracy)})"
    )
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_histogram(
    rows: list[dict],
    output_path: Path,
    model_name: str,
    overall_accuracy: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    p_correct = np.asarray([row["p_correct"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(p_correct, bins=30, color="#4c78a8", alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Choice-normalized p(correct)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"SeqQA Difficulty for {model_name} "
        f"(overall accuracy = {format_accuracy(overall_accuracy)})"
    )
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.easy_threshold <= args.medium_threshold:
        raise ValueError("--easy-threshold must be greater than --medium-threshold")

    model_org, model_name = dataset_model_parts(args.dataset)
    model_label = f"{model_org}/{model_name}"
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    rows, skipped = build_rows(dataset, args.easy_threshold, args.medium_threshold)
    if not rows:
        raise RuntimeError("No rows could be parsed from the requested dataset split.")

    difficulty_stats = compute_accuracy_by_difficulty(rows)
    percentile_summary = assign_percentile_difficulties(rows)
    percentile_stats = compute_accuracy_by_difficulty(rows, difficulty_key="percentile_difficulty")
    overall_correct = [row["is_correct"] for row in rows if row["is_correct"] is not None]
    overall_accuracy = float(sum(overall_correct) / len(overall_correct)) if overall_correct else None
    question_lengths = [row["question_text_len"] for row in rows]
    choice_lengths = [row["gold_choice_text_len"] for row in rows]
    length_correlation = float(np.corrcoef(question_lengths, choice_lengths)[0, 1])

    plot_rows(
        rows,
        args.output,
        model_label,
        args.easy_threshold,
        args.medium_threshold,
        difficulty_stats,
        overall_accuracy,
    )
    plot_percentile_rows(
        rows,
        args.percentile_output,
        model_label,
        percentile_stats,
        overall_accuracy,
    )
    plot_histogram(
        rows,
        args.histogram_output,
        model_label,
        overall_accuracy,
    )

    counts = Counter(row["difficulty"] for row in rows)
    print(f"Loaded {len(rows)} rows from {args.dataset}/{args.config}/{args.split}")
    print(f"Skipped {skipped} rows")
    print(
        "Difficulty counts: "
        + ", ".join(f"{label}={counts.get(label, 0)}" for label in ("easy", "medium", "hard"))
    )
    print(f"Overall accuracy: {format_accuracy(overall_accuracy)}")
    print(
        "Accuracy by difficulty: "
        + ", ".join(
            (
                f"{label}={format_accuracy(difficulty_stats[label]['accuracy'])}"
                f" (n={difficulty_stats[label]['count']})"
            )
            for label in ("easy", "medium", "hard")
        )
    )
    print(
        "Percentile-bucket accuracy: "
        + ", ".join(
            (
                f"{label}={format_accuracy(percentile_stats[label]['accuracy'])}"
                f" (n={percentile_stats[label]['count']}, "
                f"p_range=[{percentile_summary[label]['min_p_correct']:.3f}, "
                f"{percentile_summary[label]['max_p_correct']:.3f}])"
            )
            for label in ("easy", "medium", "hard")
        )
    )
    print(f"Question-length vs gold-choice-length correlation: {length_correlation:.3f}")
    print(f"Saved plot to {args.output}")
    print(f"Saved percentile plot to {args.percentile_output}")
    print(f"Saved histogram to {args.histogram_output}")


if __name__ == "__main__":
    main()
