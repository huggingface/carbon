"""Plot paired SeqQA p(correct) scatter diagnostics for two models.

Usage:
    uv run --directory evaluation python scripts/plot_seqqa_pair.py \
        --base-dataset hf-carbon/details_Qwen__Qwen3-4B-Base_private \
        --mid-dataset hf-carbon/details_abl10-mix-papers-regex-lr2e5__step_20000_private \
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
            "for each question, and plot a paired scatter with base-model percentile difficulty buckets."
        )
    )
    parser.add_argument(
        "--base-dataset",
        default="hf-carbon/details_Qwen__Qwen3-4B-Base_private",
        help="HF dataset repo id for the base model.",
    )
    parser.add_argument(
        "--mid-dataset",
        required=True,
        help="HF dataset repo id for the comparison model.",
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
            "scratch/seqqa_pair/{base_org}/{base_model}__vs__{mid_org}__{mid_model}/scatter.png."
        ),
    )
    parser.add_argument(
        "--bar-output",
        type=Path,
        default=None,
        help=(
            "Where to save the grouped accuracy bar PNG. Defaults under "
            "scratch/seqqa_pair/{base_org}/{base_model}__vs__{mid_org}__{mid_model}/bar.png."
        ),
    )
    args = parser.parse_args()
    default_outputs = make_default_output_paths(args.base_dataset, args.mid_dataset)
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


def make_default_output_paths(base_dataset_repo_id: str, mid_dataset_repo_id: str) -> dict[str, Path]:
    base_org, base_model_name = dataset_model_parts(base_dataset_repo_id)
    mid_org, mid_model_name = dataset_model_parts(mid_dataset_repo_id)
    base_dir = (
        SCRATCH_ROOT
        / base_org
        / f"{base_model_name}__vs__{mid_org}__{mid_model_name}"
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
    base_rows: dict[tuple[str, tuple[str, ...], int], dict],
    mid_rows: dict[tuple[str, tuple[str, ...], int], dict],
) -> tuple[list[dict], int, int]:
    common_keys = sorted(base_rows.keys() & mid_rows.keys())
    paired_rows = [
        {
            "base_p_correct": base_rows[key]["p_correct"],
            "mid_p_correct": mid_rows[key]["p_correct"],
            "difficulty": base_rows[key]["percentile_difficulty"],
            "base_is_correct": base_rows[key]["is_correct"],
            "mid_is_correct": mid_rows[key]["is_correct"],
        }
        for key in common_keys
    ]
    return paired_rows, len(base_rows) - len(common_keys), len(mid_rows) - len(common_keys)


def compute_bucket_stats(
    paired_rows: list[dict],
) -> dict[str, dict[str, float | int | None]]:
    stats = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in paired_rows if row["difficulty"] == difficulty]
        base_correct = [row["base_is_correct"] for row in subset if row["base_is_correct"] is not None]
        mid_correct = [row["mid_is_correct"] for row in subset if row["mid_is_correct"] is not None]
        base_accuracy = float(sum(base_correct) / len(base_correct)) if base_correct else None
        mid_accuracy = float(sum(mid_correct) / len(mid_correct)) if mid_correct else None
        base_stderr = float(np.std(base_correct) / np.sqrt(len(base_correct))) if base_correct else None
        mid_stderr = float(np.std(mid_correct) / np.sqrt(len(mid_correct))) if mid_correct else None
        mean_delta = float(
            np.mean([row["mid_p_correct"] - row["base_p_correct"] for row in subset])
        ) if subset else None
        stats[difficulty] = {
            "count": len(subset),
            "base_accuracy": base_accuracy,
            "mid_accuracy": mid_accuracy,
            "base_stderr": base_stderr,
            "mid_stderr": mid_stderr,
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
    base_label: str,
    mid_label: str,
    bucket_stats: dict[str, dict[str, float | int | None]],
    base_accuracy: float | None,
    mid_accuracy: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in paired_rows if row["difficulty"] == difficulty]
        if not subset:
            continue
        ax.scatter(
            [row["base_p_correct"] for row in subset],
            [row["mid_p_correct"] for row in subset],
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
    ax.set_xlabel(f"{base_label} p(correct)")
    ax.set_ylabel(f"{mid_label} p(correct)")
    ax.set_title(fill(f"SeqQA Paired p(correct): {base_label} vs {mid_label}", width=64))
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="lower right")

    overall_delta = float(
        np.mean([row["mid_p_correct"] - row["base_p_correct"] for row in paired_rows])
    )
    summary = "\n".join(
        [
            f"paired n={len(paired_rows)}",
            f"acc={format_accuracy(base_accuracy)} -> {format_accuracy(mid_accuracy)}",
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
    base_label: str,
    mid_label: str,
    bucket_stats: dict[str, dict[str, float | int | None]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    difficulties = ("easy", "medium", "hard")
    x = np.arange(len(difficulties))
    width = 0.34
    base_values = [
        np.nan if bucket_stats[difficulty]["base_accuracy"] is None else bucket_stats[difficulty]["base_accuracy"]
        for difficulty in difficulties
    ]
    mid_values = [
        np.nan if bucket_stats[difficulty]["mid_accuracy"] is None else bucket_stats[difficulty]["mid_accuracy"]
        for difficulty in difficulties
    ]
    base_stderrs = [
        np.nan if bucket_stats[difficulty]["base_stderr"] is None else bucket_stats[difficulty]["base_stderr"]
        for difficulty in difficulties
    ]
    mid_stderrs = [
        np.nan if bucket_stats[difficulty]["mid_stderr"] is None else bucket_stats[difficulty]["mid_stderr"]
        for difficulty in difficulties
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    error_style = {
        "elinewidth": 1.2,
        "ecolor": "#333333",
        "capsize": 5,
        "capthick": 1.2,
    }
    base_bars = ax.bar(
        x - width / 2,
        base_values,
        width=width,
        color="#4c78a8",
        label=base_label,
        yerr=base_stderrs,
        error_kw=error_style,
    )
    mid_bars = ax.bar(
        x + width / 2,
        mid_values,
        width=width,
        color="#f58518",
        label=mid_label,
        yerr=mid_stderrs,
        error_kw=error_style,
    )

    for bars, values, errors in (
        (base_bars, base_values, base_stderrs),
        (mid_bars, mid_values, mid_stderrs),
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
    ax.set_title(fill(f"SeqQA Accuracy by Base Percentile Difficulty \n {base_label} vs {mid_label}", width=64))
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    base_org, base_model_name = dataset_model_parts(args.base_dataset)
    mid_org, mid_model_name = dataset_model_parts(args.mid_dataset)
    base_label = f"{base_org}/{base_model_name}"
    mid_label = f"{mid_org}/{mid_model_name}"

    base_dataset = load_dataset(args.base_dataset, args.config, split=args.split)
    mid_dataset = load_dataset(args.mid_dataset, args.config, split=args.split)

    base_rows, base_skipped = build_rows(base_dataset)
    mid_rows, mid_skipped = build_rows(mid_dataset)
    percentile_summary = assign_percentile_difficulties(base_rows)
    paired_rows, base_only, mid_only = pair_rows(base_rows, mid_rows)
    if not paired_rows:
        raise RuntimeError("No paired rows could be matched between the requested dataset splits.")

    bucket_stats = compute_bucket_stats(paired_rows)
    base_accuracy = compute_overall_accuracy(paired_rows, "base_is_correct")
    mid_accuracy = compute_overall_accuracy(paired_rows, "mid_is_correct")

    plot_rows(
        paired_rows,
        args.output,
        base_label,
        mid_label,
        bucket_stats,
        base_accuracy,
        mid_accuracy,
    )
    plot_accuracy_bars(
        args.bar_output,
        base_label,
        mid_label,
        bucket_stats,
    )

    counts = Counter(row["difficulty"] for row in paired_rows)
    print(f"Loaded {len(base_rows)} base rows from {args.base_dataset}/{args.config}/{args.split}")
    print(f"Loaded {len(mid_rows)} mid rows from {args.mid_dataset}/{args.config}/{args.split}")
    print(f"Skipped base rows: {base_skipped}")
    print(f"Skipped mid rows: {mid_skipped}")
    print(f"Paired rows: {len(paired_rows)}")
    print(f"Base-only rows after pairing: {base_only}")
    print(f"Mid-only rows after pairing: {mid_only}")
    print(
        "Fixed base percentile difficulty counts: "
        + ", ".join(f"{label}={counts.get(label, 0)}" for label in ("easy", "medium", "hard"))
    )
    print(
        f"Overall accuracy: {base_label}={format_accuracy(base_accuracy)}, "
        f"{mid_label}={format_accuracy(mid_accuracy)}"
    )
    print(
        "Mean p(correct) delta by base percentile difficulty: "
        + ", ".join(
            f"{label}={format_delta(bucket_stats[label]['mean_delta'])}"
            f" (n={bucket_stats[label]['count']})"
            for label in ("easy", "medium", "hard")
        )
    )
    print(
        "Accuracy by base percentile difficulty: "
        + ", ".join(
            f"{label}="
            f"{format_accuracy(bucket_stats[label]['base_accuracy'])}"
            f"->{format_accuracy(bucket_stats[label]['mid_accuracy'])}"
            f" (n={bucket_stats[label]['count']})"
            for label in ("easy", "medium", "hard")
        )
    )
    print(
        "Base percentile bucket p(correct) ranges: "
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
