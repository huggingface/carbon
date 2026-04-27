"""Plot sequence-recovery accuracy vs generation length across models.

Reads summary JSONs written by `evaluation/sequence_recovery_eval.py` in the
directory layout produced by `evaluation/submit_sequence_recovery_gen_len_sweep.sh`:

  {base_dir}/{model_name}/{data_type}/gen_len_{gen_len}/*.json

For each model it plots overall_accuracy (and optionally per-type accuracy) as a
function of gen_len_bp = gen_len * bp_per_token (inferred from summary).

Usage:
  uv run --project evaluation python evaluation/scripts/plot_sequence_recovery_sweep.py \
    --base_dir ./eval_results/sequence_recovery_long_rollouts_pow2 \
    --data_type eukaryote \
    --model "3B hybrid=Carbon-3B-600B-dna-generv2-fp32-lmhead" \
    --model "8B hybrid=Carbon-8B-600B-dna-fp32-lmhead" \
    --out scratch/plots/sequence_recovery_sweep_overall.png \
    --type_panels scratch/plots/sequence_recovery_sweep_types.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Root dir containing {model_name}/{data_type}/gen_len_*/ subdirs.",
    )
    parser.add_argument(
        "--data_type",
        default="eukaryote",
        help="Data-type split name used as a subdirectory.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        dest="models",
        help="Repeatable 'LABEL=MODEL_NAME' or 'LABEL=BASE_DIR::MODEL_NAME' mapping. "
             "MODEL_NAME must match the directory name under the resolved base dir. "
             "When BASE_DIR is omitted, --base_dir is used.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path for the overall-accuracy plot.",
    )
    parser.add_argument(
        "--type_panels",
        default=None,
        help="Optional output PNG path for a per-type panel grid.",
    )
    parser.add_argument(
        "--random_baseline",
        type=float,
        default=0.25,
        help="Horizontal reference line (default 0.25 = 4-base uniform).",
    )
    return parser.parse_args()


def load_sweep(base_dir: str, data_type: str, model_name: str):
    """Return rows = list of dicts, one per gen_len, sorted by gen_len_bp."""
    pattern = os.path.join(base_dir, model_name, data_type, "gen_len_*", "*.json")
    paths = glob.glob(pattern)
    rows = []
    for p in paths:
        with open(p) as f:
            s = json.load(f)
        gen_len_dir = os.path.basename(os.path.dirname(p))
        gen_len = int(gen_len_dir.removeprefix("gen_len_"))
        requested_bp = int(s.get("requested_rollout_bp") or 0)
        bp_per_token = (
            requested_bp // gen_len
            if gen_len > 0 and requested_bp % gen_len == 0
            else 6
        )
        rows.append(
            {
                "gen_len": gen_len,
                "gen_len_bp": gen_len * bp_per_token,
                "overall": float(s["overall_accuracy"]),
                "label_source": s.get("label_source", "dataset"),
                "type_accuracy": s.get("type_accuracy", {}),
                "accuracy_mode": s.get("accuracy_mode"),
            }
        )
    rows.sort(key=lambda r: r["gen_len_bp"])
    return rows


def parse_model_specs(specs, default_base_dir):
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--model must be 'LABEL=MODEL_NAME' or 'LABEL=BASE_DIR::MODEL_NAME', got: {spec}"
            )
        label, rhs = spec.split("=", 1)
        if "::" in rhs:
            base, name = rhs.split("::", 1)
        else:
            base, name = default_base_dir, rhs
        parsed.append((label.strip(), base.strip(), name.strip()))
    return parsed


def plot_overall(models_data, out_path: str, random_baseline: float):
    fig, ax = plt.subplots(figsize=(11, 6.6), dpi=200)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (label, _), rows, color in zip(
        models_data.keys(),
        models_data.values(),
        colors,
    ):
        if not rows:
            continue
        xs = [r["gen_len_bp"] for r in rows]
        ys = [r["overall"] for r in rows]
        ax.plot(xs, ys, color=color, linewidth=2, label=label, zorder=2)
        for r in rows:
            marker = "o" if r["label_source"] == "dataset" else "s"
            ax.scatter(
                [r["gen_len_bp"]],
                [r["overall"]],
                color=color,
                marker=marker,
                s=60,
                zorder=3,
                edgecolors="white",
                linewidths=0.8,
            )

    ax.axhline(
        random_baseline,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        label="Random baseline",
    )
    ax.scatter([], [], color="#444444", marker="o", s=60, label="label_source=dataset")
    ax.scatter([], [], color="#444444", marker="s", s=60, label="label_source=sequence_tail")

    ax.set_xscale("log", base=2)
    all_x = sorted({r["gen_len_bp"] for rows in models_data.values() for r in rows})
    if all_x:
        ax.set_xticks(all_x)
        ax.set_xticklabels([str(x) for x in all_x])
    ax.set_xlabel("Generation length (base pairs)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Long-rollout sweep: Overall accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved overall plot to {out_path}")
    plt.close(fig)


def plot_type_panels(models_data, out_path: str, random_baseline: float):
    type_names = sorted(
        {
            t
            for rows in models_data.values()
            for r in rows
            for t in r["type_accuracy"].keys()
        }
    )
    if not type_names:
        print("No per-type accuracy available; skipping type panels")
        return

    n = len(type_names)
    cols = 3
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.5, rows_n * 3.2), dpi=200)
    axes = np.array(axes).reshape(-1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, tname in zip(axes, type_names):
        for (label, _), rows, color in zip(
            models_data.keys(),
            models_data.values(),
            colors,
        ):
            xs = [r["gen_len_bp"] for r in rows if tname in r["type_accuracy"]]
            ys = [r["type_accuracy"][tname] for r in rows if tname in r["type_accuracy"]]
            if not xs:
                continue
            ax.plot(xs, ys, color=color, linewidth=1.8, label=label)
            for r in rows:
                if tname not in r["type_accuracy"]:
                    continue
                marker = "o" if r["label_source"] == "dataset" else "s"
                ax.scatter(
                    [r["gen_len_bp"]],
                    [r["type_accuracy"][tname]],
                    color=color,
                    marker=marker,
                    s=40,
                    edgecolors="white",
                    linewidths=0.6,
                )

        ax.axhline(random_baseline, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_title(tname)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    for extra in axes[n:]:
        extra.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Long-rollout sweep: Per-type accuracy", y=1.00)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved type-panel plot to {out_path}")
    plt.close(fig)


def main():
    args = parse_args()
    model_specs = parse_model_specs(args.models, args.base_dir)

    models_data = {}
    for label, base, name in model_specs:
        rows = load_sweep(base, args.data_type, name)
        models_data[(label, name)] = rows
        print(f"  [{label}] ({base}/{name}): {len(rows)} gen_len points")

    if not any(models_data.values()):
        raise SystemExit(
            f"No summary JSONs found under {args.base_dir}. "
            f"Check --base_dir / --model names / --data_type."
        )

    plot_overall(models_data, args.out, args.random_baseline)
    if args.type_panels:
        plot_type_panels(models_data, args.type_panels, args.random_baseline)


if __name__ == "__main__":
    main()
