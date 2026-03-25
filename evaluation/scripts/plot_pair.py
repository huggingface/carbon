"""Plot paired SeqQA p(correct) scatter diagnostics for two models.

Usage:
    uv run --directory evaluation python scripts/plot_pair.py \
        --model-a-dataset hf-carbon/details_Qwen__Qwen3-4B-Base_private \
        --model-b-dataset hf-carbon/details_abl10-mix-papers-regex-lr2e5__step_20000_private \
        --difficulty-dataset hf-carbon/details_Qwen__Qwen3.5-35B-A3B-Base_private \
        --config lab_bench_seqqa_mcf_all_0 \
        --split latest
"""

import argparse
from collections import Counter
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

SCRATCH_ROOT = Path(__file__).resolve().parents[2] / "scratch" / "seqqa_pair"
DIFFICULTY_COLORS = {
    "easy": "#54a24b",
    "medium": "#eeca3b",
    "hard": "#d85c27",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load two LAB-Bench SeqQA details subsets, compute choice-normalized p(correct) "
            "for each question, and plot a paired scatter with percentile difficulty buckets "
            "defined by a reference details dataset."
        )
    )
    parser.add_argument(
        "--model-a-dataset",
        default="hf-carbon/details_Qwen__Qwen3-4B-Base_private",
        help="HF dataset repo id for model A.",
    )
    parser.add_argument(
        "--model-b-dataset",
        required=True,
        help="HF dataset repo id for model B.",
    )
    parser.add_argument(
        "--difficulty-dataset",
        default="hf-carbon/details_Qwen__Qwen3.5-35B-A3B-Base_private",
        help="HF dataset repo id used to define the percentile difficulty buckets.",
    )
    parser.add_argument(
        "--config",
        default="lab_bench_seqqa_mcf_all_0",
        help="HF dataset config name to load from both repos.",
    )
    parser.add_argument(
        "--split",
        default="latest",
        help="HF dataset split name to load from both repos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to save the paired scatter PNG. Defaults under "
            "scratch/seqqa_pair/{difficulty_org}/{difficulty_model}__ref__"
            "{model_a_org}__{model_a_model}__vs__{model_b_org}__{model_b_model}/scatter.png."
        ),
    )
    parser.add_argument(
        "--bar-output",
        type=Path,
        default=None,
        help=(
            "Where to save the grouped accuracy bar PNG. Defaults under "
            "scratch/seqqa_pair/{difficulty_org}/{difficulty_model}__ref__"
            "{model_a_org}__{model_a_model}__vs__{model_b_org}__{model_b_model}/bar.png."
        ),
    )
    args = parser.parse_args()
    default_outputs = make_default_output_paths(
        args.model_a_dataset,
        args.model_b_dataset,
        args.difficulty_dataset,
    )
    if args.output is None:
        args.output = default_outputs["output"]
    if args.bar_output is None:
        args.bar_output = default_outputs["bar_output"]
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


def make_default_output_paths(
    model_a_dataset_repo_id: str,
    model_b_dataset_repo_id: str,
    difficulty_dataset_repo_id: str,
) -> dict[str, Path]:
    model_a_org, model_a_name = dataset_model_parts(model_a_dataset_repo_id)
    model_b_org, model_b_name = dataset_model_parts(model_b_dataset_repo_id)
    difficulty_org, difficulty_model_name = dataset_model_parts(difficulty_dataset_repo_id)
    base_dir = (
        SCRATCH_ROOT
        / difficulty_org
        / (
            f"{difficulty_model_name}__ref__"
            f"{model_a_org}__{model_a_name}__vs__{model_b_org}__{model_b_name}"
        )
    )
    return {
        "output": base_dir / "scatter.png",
        "bar_output": base_dir / "bar.png",
    }


def normalize_choice_probs(logprobs: list[float]) -> np.ndarray:
    values = np.asarray(logprobs, dtype=float)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def normalize_gold_index(gold_index: int | list[int] | None) -> int:
    if gold_index is None:
        raise ValueError("gold_index is missing")
    if isinstance(gold_index, list):
        if len(gold_index) != 1:
            raise ValueError(f"expected one gold index, got {gold_index!r}")
        gold_index = gold_index[0]
    return int(gold_index)


def question_key(doc: dict) -> tuple[str, tuple[str, ...], int]:
    query = doc["query"]
    raw_choices = doc.get("choices") or []
    if len(raw_choices) < 2:
        raise ValueError("expected at least two choices")
    gold_index = normalize_gold_index(doc.get("gold_index"))
    if gold_index < 0 or gold_index >= len(raw_choices):
        raise ValueError(f"gold index {gold_index} is out of range for {len(raw_choices)} choices")
    return (query, tuple(str(choice).strip() for choice in raw_choices), gold_index)


def build_rows(
    dataset,
) -> tuple[dict[tuple[str, tuple[str, ...], int], dict], int]:
    rows = {}
    skipped = 0

    for example in dataset:
        doc = example["doc"]
        metric = example["metric"]
        model_response = example["model_response"]
        raw_choices = doc.get("choices") or []
        logprobs = model_response.get("logprobs") or []

        if len(raw_choices) < 2 or len(logprobs) != len(raw_choices):
            skipped += 1
            continue

        try:
            key = question_key(doc)
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        if key in rows:
            raise ValueError(f"duplicate question key found for query: {key[0]!r}")

        gold_index = key[2]
        probs = normalize_choice_probs(logprobs)
        is_correct = None
        if isinstance(metric, dict) and metric.get("acc") is not None:
            is_correct = int(metric["acc"])
        rows[key] = {
            "p_correct": float(probs[gold_index]),
            "is_correct": is_correct,
        }

    return rows, skipped


def assign_percentile_difficulties(
    rows: dict[tuple[str, tuple[str, ...], int], dict],
) -> dict[str, dict[str, float | int]]:
    if not rows:
        return {}

    sorted_keys = sorted(rows, key=lambda key: rows[key]["p_correct"])
    buckets = np.array_split(np.arange(len(sorted_keys)), 3)
    labels = ("hard", "medium", "easy")
    summary = {}

    for label, bucket_indices in zip(labels, buckets, strict=True):
        bucket_keys = [sorted_keys[int(index)] for index in bucket_indices]
        bucket_values = [rows[key]["p_correct"] for key in bucket_keys]
        for key in bucket_keys:
            rows[key]["percentile_difficulty"] = label
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


def pair_rows(
    model_a_rows: dict[tuple[str, tuple[str, ...], int], dict],
    model_b_rows: dict[tuple[str, tuple[str, ...], int], dict],
    difficulty_rows: dict[tuple[str, tuple[str, ...], int], dict],
) -> tuple[list[dict], int, int, int]:
    common_keys = sorted(model_a_rows.keys() & model_b_rows.keys() & difficulty_rows.keys())
    paired_rows = [
        {
            "model_a_p_correct": model_a_rows[key]["p_correct"],
            "model_b_p_correct": model_b_rows[key]["p_correct"],
            "difficulty": difficulty_rows[key]["percentile_difficulty"],
            "model_a_is_correct": model_a_rows[key]["is_correct"],
            "model_b_is_correct": model_b_rows[key]["is_correct"],
        }
        for key in common_keys
    ]
    return (
        paired_rows,
        len(model_a_rows) - len(common_keys),
        len(model_b_rows) - len(common_keys),
        len(difficulty_rows) - len(common_keys),
    )


def compute_bucket_stats(
    paired_rows: list[dict],
) -> dict[str, dict[str, float | int | None]]:
    stats = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in paired_rows if row["difficulty"] == difficulty]
        model_a_correct = [
            row["model_a_is_correct"] for row in subset if row["model_a_is_correct"] is not None
        ]
        model_b_correct = [
            row["model_b_is_correct"] for row in subset if row["model_b_is_correct"] is not None
        ]
        model_a_accuracy = float(sum(model_a_correct) / len(model_a_correct)) if model_a_correct else None
        model_b_accuracy = float(sum(model_b_correct) / len(model_b_correct)) if model_b_correct else None
        model_a_stderr = (
            float(np.std(model_a_correct) / np.sqrt(len(model_a_correct))) if model_a_correct else None
        )
        model_b_stderr = (
            float(np.std(model_b_correct) / np.sqrt(len(model_b_correct))) if model_b_correct else None
        )
        mean_delta = float(
            np.mean([row["model_b_p_correct"] - row["model_a_p_correct"] for row in subset])
        ) if subset else None
        stats[difficulty] = {
            "count": len(subset),
            "model_a_accuracy": model_a_accuracy,
            "model_b_accuracy": model_b_accuracy,
            "model_a_stderr": model_a_stderr,
            "model_b_stderr": model_b_stderr,
            "mean_delta": mean_delta,
        }
    return stats


def compute_overall_accuracy(paired_rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in paired_rows if row[key] is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def format_accuracy(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def plot_rows(
    paired_rows: list[dict],
    output_path: Path,
    model_a_label: str,
    model_b_label: str,
    difficulty_label: str,
    bucket_stats: dict[str, dict[str, float | int | None]],
    model_a_accuracy: float | None,
    model_b_accuracy: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in paired_rows if row["difficulty"] == difficulty]
        if not subset:
            continue
        ax.scatter(
            [row["model_a_p_correct"] for row in subset],
            [row["model_b_p_correct"] for row in subset],
            s=42,
            alpha=0.82,
            color=DIFFICULTY_COLORS[difficulty],
            edgecolors="white",
            linewidths=0.4,
            label=(
                f"{difficulty} "
                f"(n={len(subset)}, mean delta={format_delta(bucket_stats[difficulty]['mean_delta'])})"
            ),
        )

    ax.plot([0, 1], [0, 1], color="#6c6c6c", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{model_a_label} p(correct)")
    ax.set_ylabel(f"{model_b_label} p(correct)")
    ax.set_title(
        fill(
            f"SeqQA Paired p(correct): {model_a_label} vs {model_b_label} "
            f"(difficulty ref: {difficulty_label})",
            width=64,
        )
    )
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="lower right")

    overall_delta = float(
        np.mean([row["model_b_p_correct"] - row["model_a_p_correct"] for row in paired_rows])
    )
    summary = "\n".join(
        [
            f"paired n={len(paired_rows)}",
            f"acc={format_accuracy(model_a_accuracy)} -> {format_accuracy(model_b_accuracy)}",
            f"mean delta={format_delta(overall_delta)}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_accuracy_bars(
    output_path: Path,
    model_a_label: str,
    model_b_label: str,
    difficulty_label: str,
    bucket_stats: dict[str, dict[str, float | int | None]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    difficulties = ("easy", "medium", "hard")
    x = np.arange(len(difficulties))
    width = 0.34
    model_a_values = [
        np.nan
        if bucket_stats[difficulty]["model_a_accuracy"] is None
        else bucket_stats[difficulty]["model_a_accuracy"]
        for difficulty in difficulties
    ]
    model_b_values = [
        np.nan
        if bucket_stats[difficulty]["model_b_accuracy"] is None
        else bucket_stats[difficulty]["model_b_accuracy"]
        for difficulty in difficulties
    ]
    model_a_stderrs = [
        np.nan
        if bucket_stats[difficulty]["model_a_stderr"] is None
        else bucket_stats[difficulty]["model_a_stderr"]
        for difficulty in difficulties
    ]
    model_b_stderrs = [
        np.nan
        if bucket_stats[difficulty]["model_b_stderr"] is None
        else bucket_stats[difficulty]["model_b_stderr"]
        for difficulty in difficulties
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    error_style = {
        "elinewidth": 1.2,
        "ecolor": "#333333",
        "capsize": 5,
        "capthick": 1.2,
    }
    model_a_bars = ax.bar(
        x - width / 2,
        model_a_values,
        width=width,
        color="#4c78a8",
        label=model_a_label,
        yerr=model_a_stderrs,
        error_kw=error_style,
    )
    model_b_bars = ax.bar(
        x + width / 2,
        model_b_values,
        width=width,
        color="#f58518",
        label=model_b_label,
        yerr=model_b_stderrs,
        error_kw=error_style,
    )

    for bars, values, errors in (
        (model_a_bars, model_a_values, model_a_stderrs),
        (model_b_bars, model_b_values, model_b_stderrs),
    ):
        for bar, value, error in zip(bars, values, errors, strict=True):
            if np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#666666",
                )
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + error + 0.02, 1.03),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )

    ax.set_xticks(x, [difficulty.title() for difficulty in difficulties])
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        fill(
            "SeqQA Accuracy by Reference Percentile Difficulty "
            f"({difficulty_label})\n{model_a_label} vs {model_b_label}",
            width=64,
        )
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    model_a_org, model_a_name = dataset_model_parts(args.model_a_dataset)
    model_b_org, model_b_name = dataset_model_parts(args.model_b_dataset)
    difficulty_org, difficulty_model_name = dataset_model_parts(args.difficulty_dataset)
    model_a_label = f"{model_a_org}/{model_a_name}"
    model_b_label = f"{model_b_org}/{model_b_name}"
    difficulty_label = f"{difficulty_org}/{difficulty_model_name}"

    model_a_dataset = load_dataset(args.model_a_dataset, args.config, split=args.split)
    model_b_dataset = load_dataset(args.model_b_dataset, args.config, split=args.split)
    difficulty_dataset = load_dataset(args.difficulty_dataset, args.config, split=args.split)

    model_a_rows, model_a_skipped = build_rows(model_a_dataset)
    model_b_rows, model_b_skipped = build_rows(model_b_dataset)
    difficulty_rows, difficulty_skipped = build_rows(difficulty_dataset)
    percentile_summary = assign_percentile_difficulties(difficulty_rows)
    paired_rows, model_a_only, model_b_only, difficulty_only = pair_rows(
        model_a_rows,
        model_b_rows,
        difficulty_rows,
    )
    if not paired_rows:
        raise RuntimeError("No paired rows could be matched between the requested dataset splits.")

    bucket_stats = compute_bucket_stats(paired_rows)
    model_a_accuracy = compute_overall_accuracy(paired_rows, "model_a_is_correct")
    model_b_accuracy = compute_overall_accuracy(paired_rows, "model_b_is_correct")

    plot_rows(
        paired_rows,
        args.output,
        model_a_label,
        model_b_label,
        difficulty_label,
        bucket_stats,
        model_a_accuracy,
        model_b_accuracy,
    )
    plot_accuracy_bars(
        args.bar_output,
        model_a_label,
        model_b_label,
        difficulty_label,
        bucket_stats,
    )

    counts = Counter(row["difficulty"] for row in paired_rows)
    print(
        f"Loaded {len(model_a_rows)} model A rows from "
        f"{args.model_a_dataset}/{args.config}/{args.split}"
    )
    print(
        f"Loaded {len(model_b_rows)} model B rows from "
        f"{args.model_b_dataset}/{args.config}/{args.split}"
    )
    print(
        "Loaded "
        f"{len(difficulty_rows)} difficulty rows from "
        f"{args.difficulty_dataset}/{args.config}/{args.split}"
    )
    print(f"Skipped model A rows: {model_a_skipped}")
    print(f"Skipped model B rows: {model_b_skipped}")
    print(f"Skipped difficulty rows: {difficulty_skipped}")
    print(f"Paired rows: {len(paired_rows)}")
    print(f"Model A-only rows after pairing: {model_a_only}")
    print(f"Model B-only rows after pairing: {model_b_only}")
    print(f"Difficulty-only rows after pairing: {difficulty_only}")
    print(
        f"Fixed percentile difficulty counts from {difficulty_label}: "
        + ", ".join(f"{label}={counts.get(label, 0)}" for label in ("easy", "medium", "hard"))
    )
    print(
        f"Overall accuracy: {model_a_label}={format_accuracy(model_a_accuracy)}, "
        f"{model_b_label}={format_accuracy(model_b_accuracy)}"
    )
    print(
        f"Mean p(correct) delta by {difficulty_label} percentile difficulty: "
        + ", ".join(
            f"{label}={format_delta(bucket_stats[label]['mean_delta'])}"
            f" (n={bucket_stats[label]['count']})"
            for label in ("easy", "medium", "hard")
        )
    )
    print(
        f"Accuracy by {difficulty_label} percentile difficulty: "
        + ", ".join(
            f"{label}="
            f"{format_accuracy(bucket_stats[label]['model_a_accuracy'])}"
            f"->{format_accuracy(bucket_stats[label]['model_b_accuracy'])}"
            f" (n={bucket_stats[label]['count']})"
            for label in ("easy", "medium", "hard")
        )
    )
    print(
        f"Reference percentile bucket p(correct) ranges from {difficulty_label}: "
        + ", ".join(
            f"{label}=[{percentile_summary[label]['min_p_correct']:.3f}, "
            f"{percentile_summary[label]['max_p_correct']:.3f}]"
            for label in ("easy", "medium", "hard")
        )
    )
    print(f"Saved plot to {args.output}")
    print(f"Saved accuracy bar plot to {args.bar_output}")


if __name__ == "__main__":
    main()
