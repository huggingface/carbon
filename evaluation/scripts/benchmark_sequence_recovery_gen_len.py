import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "evaluation" / "sequence_recovery_eval.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scratch" / "sequence_recovery_gen_len_benchmark"
DEFAULT_GEN_LENS = [5, 10, 20, 40, 80, 160, 320, 640]
GROUP_ORDER = [
    "fungi",
    "invertebrate",
    "plant",
    "protozoa",
    "vertebrate_mammalian",
    "vertebrate_other",
    "overall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sequence recovery serially over a gen_len sweep, using all selected GPUs "
            "for each eval, then aggregate and plot accuracy-vs-gen_len."
        )
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Optional output name override passed through to sequence_recovery_eval.py",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision/tag/commit",
    )
    parser.add_argument(
        "--data_type",
        default="eukaryote",
        choices=["eukaryote", "bacteria", "others"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--data_path",
        default="hf://datasets/GenerTeam/sequence-recovery",
        help="HF dataset parquet path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for run outputs and aggregate artifacts",
    )
    parser.add_argument(
        "--gen_lens",
        type=int,
        nargs="+",
        default=DEFAULT_GEN_LENS,
        help="gen_len values to benchmark serially",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of visible GPUs to expose to each eval subprocess",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=6144,
        help="Max input length in bp",
    )
    parser.add_argument(
        "--gen_len_bp",
        type=int,
        default=None,
        help=(
            "Base-pair generation length for Evo2 runs. Defaults to "
            "gen_len * bp_per_token for each sweep point."
        ),
    )
    parser.add_argument(
        "--bp_per_token",
        type=int,
        default=6,
        help="Base pairs represented by each HF generation token.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional test-only sample cap passed through to the eval script",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="Random seed used when --max_samples subsamples the dataset",
    )
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument(
        "--use_evo2",
        action="store_true",
        help="Use official Evo2 inference path",
    )
    parser.add_argument(
        "--use_dna_tags",
        action="store_true",
        help="Wrap DNA sequences with <dna>...</dna> tags",
    )
    parser.add_argument(
        "--no_prefix",
        action="store_true",
        help="Do not add a BOS or DNA prefix token",
    )
    parser.add_argument(
        "--use_species_tags",
        action="store_true",
        help="Prepend species tags before DNA sequences",
    )
    return parser.parse_args()


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return sanitized or "run"


def resolve_visible_gpu_ids(requested_count: int) -> list[str]:
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_value and env_value != "NoDevFiles":
        visible_ids = [item.strip() for item in env_value.split(",") if item.strip()]
    else:
        visible_ids = [str(index) for index in range(torch.cuda.device_count())]

    if len(visible_ids) < requested_count:
        raise ValueError(
            f"Requested {requested_count} GPUs but only found {len(visible_ids)} visible GPUs"
        )

    return visible_ids[:requested_count]


def build_eval_command(
    args: argparse.Namespace,
    run_dir: Path,
    gen_len: int,
) -> list[str]:
    gen_len_bp = args.gen_len_bp
    if gen_len_bp is None:
        gen_len_bp = gen_len * args.bp_per_token

    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--model",
        args.model,
        "--data_type",
        args.data_type,
        "--data_path",
        args.data_path,
        "--output_dir",
        str(run_dir),
        "--max_seq_len",
        str(args.max_seq_len),
        "--gen_len",
        str(gen_len),
        "--gen_len_bp",
        str(gen_len_bp),
        "--batch_size",
        str(args.batch_size),
        "--accuracy_mode",
        "prediction_length",
        "--bp_per_token",
        str(args.bp_per_token),
    ]
    if args.model_name:
        command.extend(["--model_name", args.model_name])
    if args.revision:
        command.extend(["--revision", args.revision])
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
        command.extend(["--sample_seed", str(args.sample_seed)])
    if args.bf16:
        command.append("--bf16")
    if args.use_evo2:
        command.append("--use_evo2")
    if args.use_dna_tags:
        command.append("--use_dna_tags")
    if args.no_prefix:
        command.append("--no_prefix")
    if args.use_species_tags:
        command.append("--use_species_tags")
    return command


def load_run_outputs(run_dir: Path) -> tuple[Path, Path, dict]:
    parquet_paths = sorted(run_dir.glob("*.parquet"))
    summary_paths = sorted(run_dir.glob("*.json"))
    if len(parquet_paths) != 1 or len(summary_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one parquet and one json in {run_dir}, "
            f"found {len(parquet_paths)} parquet and {len(summary_paths)} json files"
        )

    summary_path = summary_paths[0]
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return parquet_paths[0], summary_path, summary


def build_aggregate_rows(
    gen_len: int, generation_bp: int, run_df: pd.DataFrame
) -> list[dict]:
    rows = [
        {
            "gen_len": gen_len,
            "generation_bp": generation_bp,
            "group": "overall",
            "accuracy": float(run_df["accuracy"].mean()),
            "num_sequences": int(len(run_df)),
            "effective_scored_bp": float(run_df["scored_bp"].mean()),
        }
    ]

    if "type" in run_df.columns:
        grouped = (
            run_df.groupby("type", dropna=False)
            .agg(
                accuracy=("accuracy", "mean"),
                num_sequences=("accuracy", "size"),
                effective_scored_bp=("scored_bp", "mean"),
            )
            .reset_index()
        )
        for row in grouped.to_dict("records"):
            rows.append(
                {
                    "gen_len": gen_len,
                    "generation_bp": generation_bp,
                    "group": row["type"],
                    "accuracy": float(row["accuracy"]),
                    "num_sequences": int(row["num_sequences"]),
                    "effective_scored_bp": float(row["effective_scored_bp"]),
                }
            )

    return rows


def plot_accuracy_by_group(aggregate_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    for group in GROUP_ORDER:
        group_df = aggregate_df[aggregate_df["group"] == group].sort_values("gen_len")
        if group_df.empty:
            continue

        x_values = group_df["generation_bp"]
        style = {
            "marker": "o",
            "linewidth": 2.75 if group == "overall" else 1.8,
            "color": "black" if group == "overall" else None,
        }
        ax.plot(x_values, group_df["accuracy"], label=group, **style)

    bp_lengths = sorted(aggregate_df["generation_bp"].unique())
    ax.set_xticks(bp_lengths)
    ax.set_xticklabels([str(value) for value in bp_lengths])
    ax.set_xlabel("Base pair generation length")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Sequence Recovery Accuracy vs Base Pair Generation Length by Type")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    max_scored_bp = aggregate_df["effective_scored_bp"].max()
    fig.text(
        0.5,
        0.01,
        f"Mean scored bp is capped by the dataset label length. Observed max mean scored bp: {max_scored_bp:.1f}",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overall_accuracy(aggregate_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall_df = aggregate_df[aggregate_df["group"] == "overall"].sort_values("gen_len")
    x_values = overall_df["generation_bp"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        x_values,
        overall_df["accuracy"],
        color="black",
        marker="o",
        linewidth=2.75,
    )
    ax.set_xticks(x_values.tolist())
    ax.set_xticklabels([str(value) for value in x_values.tolist()])
    ax.set_xlabel("Base pair generation length")
    ax.set_ylabel("Overall accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Overall Sequence Recovery Accuracy vs Base Pair Generation Length")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    visible_gpu_ids = resolve_visible_gpu_ids(args.num_gpus)

    model_label = args.model_name or args.model.split("/")[-1]
    benchmark_root = (
        args.output_dir
        / sanitize_path_component(model_label)
        / sanitize_path_component(args.data_type)
    )
    benchmark_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    run_manifest = []

    for gen_len in args.gen_lens:
        run_dir = benchmark_root / f"gen_len_{gen_len}"
        run_dir.mkdir(parents=True, exist_ok=True)

        command = build_eval_command(args, run_dir, gen_len)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpu_ids)

        print(
            f"\nRunning gen_len={gen_len} on GPUs {env['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )
        print("Command:", " ".join(command), flush=True)

        start_time = time.time()
        subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)
        elapsed = time.time() - start_time

        parquet_path, summary_path, summary = load_run_outputs(run_dir)
        run_df = pd.read_parquet(parquet_path)
        generation_bp = int(
            summary.get("requested_rollout_bp") or gen_len * args.bp_per_token
        )
        aggregate_rows.extend(build_aggregate_rows(gen_len, generation_bp, run_df))

        run_manifest.append(
            {
                "gen_len": gen_len,
                "run_dir": str(run_dir),
                "parquet_path": str(parquet_path),
                "summary_path": str(summary_path),
                "elapsed_seconds": elapsed,
                "overall_accuracy": float(summary["overall_accuracy"]),
                "requested_rollout_bp": generation_bp,
                "mean_scored_bp": float(summary["mean_scored_bp"]),
                "visible_gpu_count": int(summary["visible_gpu_count"]),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df["group"] = pd.Categorical(
        aggregate_df["group"],
        categories=GROUP_ORDER,
        ordered=True,
    )
    aggregate_df = aggregate_df.sort_values(
        ["group", "generation_bp", "gen_len"]
    ).reset_index(drop=True)

    aggregate_csv_path = benchmark_root / "accuracy_vs_gen_len.csv"
    aggregate_json_path = benchmark_root / "benchmark_manifest.json"
    group_plot_path = benchmark_root / "accuracy_vs_gen_len_by_type.png"
    overall_plot_path = benchmark_root / "accuracy_vs_gen_len_overall.png"

    aggregate_df.to_csv(aggregate_csv_path, index=False)
    plot_accuracy_by_group(aggregate_df, group_plot_path)
    plot_overall_accuracy(aggregate_df, overall_plot_path)

    manifest = {
        "model": args.model,
        "model_name": model_label,
        "revision": args.revision,
        "data_type": args.data_type,
        "data_path": args.data_path,
        "gen_lens": args.gen_lens,
        "bp_per_token": args.bp_per_token,
        "accuracy_mode": "prediction_length",
        "requested_gpu_count": args.num_gpus,
        "cuda_visible_devices": visible_gpu_ids,
        "sample_seed": args.sample_seed if args.max_samples is not None else None,
        "output_root": str(benchmark_root),
        "runs": run_manifest,
    }
    with aggregate_json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nWrote aggregate CSV to {aggregate_csv_path}")
    print(f"Wrote benchmark manifest to {aggregate_json_path}")
    print(f"Wrote grouped plot to {group_plot_path}")
    print(f"Wrote overall plot to {overall_plot_path}")


if __name__ == "__main__":
    main()
