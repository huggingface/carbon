"""Plot paired SeqQA p(correct) diagnostics using IRT difficulty buckets.

Usage:
    uv run --directory evaluation python scripts/plot_pair_irt.py \
        --model-a-dataset hf-carbon/details_Qwen__Qwen3-4B-Base_private \
        --model-b-dataset hf-carbon/details_abl10-mix-papers-regex-lr2e5__step_20000_private \
        --config lab_bench_seqqa_mcf_all_0 \
        --split latest
"""

import argparse
from collections import Counter
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

SCRATCH_ROOT = Path(__file__).resolve().parents[2] / "scratch" / "seqqa_pair_irt"
DIFFICULTY_COLORS = {
    "easy": "#54a24b",
    "medium": "#eeca3b",
    "hard": "#d85c27",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load two LAB-Bench SeqQA details subsets, compute choice-normalized p(correct) "
            "for each question, and plot paired diagnostics using IRT difficulty_b buckets."
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
        "--irt-dataset",
        default="hf-carbon/seqqa-irt-difficulty",
        help="HF dataset repo id containing the published IRT outputs.",
    )
    parser.add_argument(
        "--irt-subset",
        default="irt_item_difficulty",
        help="HF dataset config/subset name containing per-item IRT difficulty.",
    )
    parser.add_argument(
        "--irt-split",
        default="train",
        help="IRT dataset split name used to resolve parquet shard names.",
    )
    parser.add_argument(
        "--config",
        default="lab_bench_seqqa_mcf_all_0",
        help="HF dataset config name to load from both model repos.",
    )
    parser.add_argument(
        "--split",
        default="latest",
        help="HF dataset split name to load from both model repos.",
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
    parser.add_argument(
        "--discrimination-threshold",
        type=float,
        default=None,
        help="Keep only IRT rows with discrimination_a greater than or equal to this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to save the paired scatter PNG. Defaults under "
            "scratch/seqqa_pair_irt/{irt_org}/{irt_name}/{irt_subset}/{irt_split}/"
            "{model_a_org}__{model_a_model}__vs__{model_b_org}__{model_b_model}/scatter.png."
        ),
    )
    parser.add_argument(
        "--bar-output",
        type=Path,
        default=None,
        help=(
            "Where to save the grouped accuracy bar PNG. Defaults under "
            "scratch/seqqa_pair_irt/{irt_org}/{irt_name}/{irt_subset}/{irt_split}/"
            "{model_a_org}__{model_a_model}__vs__{model_b_org}__{model_b_model}/bar.png."
        ),
    )
    args = parser.parse_args()
    default_outputs = make_default_output_paths(
        args.model_a_dataset,
        args.model_b_dataset,
        args.irt_dataset,
        args.irt_subset,
        args.irt_split,
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


def repo_parts(dataset_repo_id: str) -> tuple[str, str]:
    dataset_org, separator, dataset_name = dataset_repo_id.partition("/")
    if not separator or not dataset_org or not dataset_name:
        raise ValueError(f"Expected a dataset repo id like org/name, got: {dataset_repo_id!r}")
    return dataset_org, dataset_name


def make_default_output_paths(
    model_a_dataset_repo_id: str,
    model_b_dataset_repo_id: str,
    irt_dataset_repo_id: str,
    irt_subset: str,
    irt_split: str,
) -> dict[str, Path]:
    model_a_org, model_a_name = dataset_model_parts(model_a_dataset_repo_id)
    model_b_org, model_b_name = dataset_model_parts(model_b_dataset_repo_id)
    irt_org, irt_name = repo_parts(irt_dataset_repo_id)
    base_dir = (
        SCRATCH_ROOT
        / irt_org
        / irt_name
        / irt_subset
        / irt_split
        / f"{model_a_org}__{model_a_name}__vs__{model_b_org}__{model_b_name}"
    )
    return {
        "output": base_dir / "scatter.png",
        "bar_output": base_dir / "bar.png",
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


def build_model_rows(dataset) -> tuple[dict[str, dict], int]:
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
            item_id = str(doc["id"])
            gold_index = normalize_gold_index(doc.get("gold_index"))
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        if gold_index < 0 or gold_index >= len(raw_choices):
            skipped += 1
            continue

        if item_id in rows:
            raise ValueError(f"duplicate item id found: {item_id!r}")

        probs = normalize_choice_probs(logprobs)
        is_correct = None
        if isinstance(metric, dict) and metric.get("acc") is not None:
            is_correct = int(metric["acc"])
        rows[item_id] = {
            "p_correct": float(probs[gold_index]),
            "is_correct": is_correct,
        }

    return rows, skipped


def build_irt_rows(
    dataset: Dataset, discrimination_threshold: float | None
) -> tuple[dict[str, dict], int, int]:
    rows = {}
    skipped = 0
    filtered = 0

    for example in dataset:
        try:
            item_id = str(example["item_id"])
            difficulty_b = float(example["difficulty_b"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        if item_id in rows:
            raise ValueError(f"duplicate IRT item id found: {item_id!r}")

        row = {"difficulty_b": difficulty_b}
        if discrimination_threshold is not None:
            try:
                discrimination_a = float(example["discrimination_a"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if discrimination_a < discrimination_threshold:
                filtered += 1
                continue
            row["discrimination_a"] = discrimination_a

        rows[item_id] = row

    return rows, skipped, filtered


def resolve_thresholds(
    irt_rows: dict[str, dict], hard_threshold: float | None, medium_threshold: float | None
) -> tuple[float, float]:
    if not irt_rows:
        raise RuntimeError("No IRT rows available to derive difficulty thresholds.")

    values = np.asarray([row["difficulty_b"] for row in irt_rows.values()], dtype=float)
    resolved_hard = (
        float(np.quantile(values, 2 / 3))
        if hard_threshold is None
        else hard_threshold
    )
    resolved_medium = (
        float(np.quantile(values, 1 / 3))
        if medium_threshold is None
        else medium_threshold
    )
    if resolved_hard <= resolved_medium:
        raise ValueError("--hard-threshold must be greater than --medium-threshold")
    return resolved_hard, resolved_medium


def difficulty_bucket(difficulty_b: float, hard_threshold: float, medium_threshold: float) -> str:
    if difficulty_b > hard_threshold:
        return "hard"
    if difficulty_b > medium_threshold:
        return "medium"
    return "easy"


def assign_irt_difficulties(
    irt_rows: dict[str, dict],
    hard_threshold: float,
    medium_threshold: float,
) -> dict[str, dict[str, float | int]]:
    summary = {}
    for label in ("easy", "medium", "hard"):
        summary[label] = {
            "count": 0,
            "min_difficulty_b": float("nan"),
            "max_difficulty_b": float("nan"),
        }

    bucket_values = {label: [] for label in ("easy", "medium", "hard")}
    for row in irt_rows.values():
        label = difficulty_bucket(row["difficulty_b"], hard_threshold, medium_threshold)
        row["difficulty"] = label
        bucket_values[label].append(row["difficulty_b"])

    for label, values in bucket_values.items():
        if values:
            summary[label] = {
                "count": len(values),
                "min_difficulty_b": float(min(values)),
                "max_difficulty_b": float(max(values)),
            }

    return summary


def pair_rows(
    model_a_rows: dict[str, dict],
    model_b_rows: dict[str, dict],
    irt_rows: dict[str, dict],
) -> tuple[list[dict], int, int, int]:
    common_keys = sorted(model_a_rows.keys() & model_b_rows.keys() & irt_rows.keys(), key=int)
    paired_rows = [
        {
            "model_a_p_correct": model_a_rows[key]["p_correct"],
            "model_b_p_correct": model_b_rows[key]["p_correct"],
            "difficulty": irt_rows[key]["difficulty"],
            "difficulty_b": irt_rows[key]["difficulty_b"],
            "model_a_is_correct": model_a_rows[key]["is_correct"],
            "model_b_is_correct": model_b_rows[key]["is_correct"],
        }
        for key in common_keys
    ]
    return (
        paired_rows,
        len(model_a_rows) - len(common_keys),
        len(model_b_rows) - len(common_keys),
        len(irt_rows) - len(common_keys),
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
        mean_delta = (
            float(np.mean([row["model_b_p_correct"] - row["model_a_p_correct"] for row in subset]))
            if subset
            else None
        )
        mean_difficulty = (
            float(np.mean([row["difficulty_b"] for row in subset])) if subset else None
        )
        stats[difficulty] = {
            "count": len(subset),
            "model_a_accuracy": model_a_accuracy,
            "model_b_accuracy": model_b_accuracy,
            "model_a_stderr": model_a_stderr,
            "model_b_stderr": model_b_stderr,
            "mean_delta": mean_delta,
            "mean_difficulty_b": mean_difficulty,
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
            f"SeqQA Paired p(correct) by IRT Difficulty",
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
            "SeqQA Accuracy by IRT Difficulty",
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
    model_a_label = f"{model_a_org}/{model_a_name}"
    model_b_label = f"{model_b_org}/{model_b_name}"
    difficulty_label = f"{args.irt_dataset}/{args.irt_subset}/{args.irt_split}"

    model_a_dataset = load_dataset(args.model_a_dataset, args.config, split=args.split)
    model_b_dataset = load_dataset(args.model_b_dataset, args.config, split=args.split)
    irt_dataset = load_irt_subset(args.irt_dataset, args.irt_subset, args.irt_split)

    model_a_rows, model_a_skipped = build_model_rows(model_a_dataset)
    model_b_rows, model_b_skipped = build_model_rows(model_b_dataset)
    irt_rows, irt_skipped, irt_filtered = build_irt_rows(
        irt_dataset, args.discrimination_threshold
    )
    if args.discrimination_threshold is not None and not irt_rows:
        raise RuntimeError(
            "No IRT rows remain after applying "
            f"--discrimination-threshold={args.discrimination_threshold}."
        )
    hard_threshold, medium_threshold = resolve_thresholds(
        irt_rows, args.hard_threshold, args.medium_threshold
    )
    irt_summary = assign_irt_difficulties(irt_rows, hard_threshold, medium_threshold)
    paired_rows, model_a_only, model_b_only, irt_only = pair_rows(
        model_a_rows,
        model_b_rows,
        irt_rows,
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
    irt_load_summary = (
        f"Loaded {len(irt_rows)} IRT rows from "
        f"{args.irt_dataset}/{args.irt_subset}/{args.irt_split}"
    )
    if args.discrimination_threshold is not None:
        irt_load_summary += " after discrimination filtering"
    print(irt_load_summary)
    print(f"Skipped model A rows: {model_a_skipped}")
    print(f"Skipped model B rows: {model_b_skipped}")
    print(f"Skipped IRT rows: {irt_skipped}")
    if args.discrimination_threshold is not None:
        print(
            "IRT discrimination filter: "
            f"discrimination_a>={args.discrimination_threshold:.3f} "
            f"(filtered {irt_filtered} rows)"
        )
    print(f"Paired rows: {len(paired_rows)}")
    print(f"Model A-only rows after pairing: {model_a_only}")
    print(f"Model B-only rows after pairing: {model_b_only}")
    print(f"IRT-only rows after pairing: {irt_only}")
    print(
        "IRT difficulty counts: "
        + ", ".join(f"{label}={counts.get(label, 0)}" for label in ("easy", "medium", "hard"))
    )
    print(
        f"IRT thresholds: hard>{hard_threshold:.3f}, "
        f"medium>{medium_threshold:.3f}"
    )
    print(
        f"Overall accuracy: {model_a_label}={format_accuracy(model_a_accuracy)}, "
        f"{model_b_label}={format_accuracy(model_b_accuracy)}"
    )
    print(
        "Mean p(correct) delta by IRT difficulty: "
        + ", ".join(
            f"{label}={format_delta(bucket_stats[label]['mean_delta'])}"
            f" (n={bucket_stats[label]['count']})"
            for label in ("easy", "medium", "hard")
        )
    )
    print(
        "Accuracy by IRT difficulty: "
        + ", ".join(
            f"{label}="
            f"{format_accuracy(bucket_stats[label]['model_a_accuracy'])}"
            f"->{format_accuracy(bucket_stats[label]['model_b_accuracy'])}"
            f" (n={bucket_stats[label]['count']})"
            for label in ("easy", "medium", "hard")
        )
    )
    print(
        "IRT difficulty_b ranges: "
        + ", ".join(
            f"{label}=[{irt_summary[label]['min_difficulty_b']:.3f}, "
            f"{irt_summary[label]['max_difficulty_b']:.3f}]"
            for label in ("easy", "medium", "hard")
        )
    )
    print(f"Saved plot to {args.output}")
    print(f"Saved accuracy bar plot to {args.bar_output}")


if __name__ == "__main__":
    main()
