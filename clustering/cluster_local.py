#!/usr/bin/env python3
"""
Cluster a local dataset of .jsonl.gz shards and overlay biology benchmarks.

Sampling: reservoir-sample --n_per_subfolder texts from each subfolder of
--dataset_dir, embed jointly with LabBench and MMLU-Bio questions, fit one
UMAP on everything, run DBSCAN on training texts only, then produce per-
benchmark overlay plots.

Usage:
    python cluster_local.py \\
        --dataset_dir /path/to/processed_regex \\
        --dataset_name processed_regex \\
        --n_per_subfolder 50000 \\
        --output_root ./output

Paths to benchmark caches are set via environment variables (see CONFIGURATION
below) or fall back to the defaults shown.
"""

import argparse
import gzip
import json
import logging
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── configuration ─────────────────────────────────────────────────────────────

TEXT_CLUSTERING_SRC = os.environ.get(
    "TEXT_CLUSTERING_SRC",
    "/fsx/dana_aubakirova/carbon_project/text-clustering",
)

HF_CACHE = os.environ.get("HF_DATASETS_CACHE", "/fsx/dana_aubakirova/.cache")

LABBENCH_ROOT = os.path.join(
    HF_CACHE,
    "datasets--hf-carbon--lab-bench/snapshots/f6fc1dad5578aaeb0d39ead96f59ed19ef1c363e",
)
LABBENCH_SUBSETS = ["CloningScenarios", "SeqQA"]

MMLU_BIO_SOURCES = [
    # (label, path_or_dir, question_col, options_col)
    (
        "mmlu/college-biology",
        os.path.join(
            HF_CACHE,
            "cais___mmlu/college_biology/0.0.0"
            "/c30699e8356da336a370243923dbaf21066bb9fe/mmlu-test.arrow",
        ),
        "question", "choices",
    ),
    (
        "mmlu/high-school-biology",
        os.path.join(
            HF_CACHE,
            "cais___mmlu/high_school_biology/0.0.0"
            "/c30699e8356da336a370243923dbaf21066bb9fe/mmlu-test.arrow",
        ),
        "question", "choices",
    ),
    (
        "mmlu-pro-bio",
        os.path.join(
            HF_CACHE,
            "hf-carbon___mmlu-pro-biology/default/0.0.0"
            "/2f77fcaf8fdc517804582eb4dfe53931ac17ebbf/mmlu-pro-biology-test.arrow",
        ),
        "question", "options",
    ),
    (
        "mmlu-redux/college-biology",
        os.path.join(HF_CACHE, "hf-carbon___mmlu-redux-2.0-biology/college_biology/"),
        "question", "options",
    ),
    (
        "mmlu-redux/medical-genetics",
        os.path.join(HF_CACHE, "hf-carbon___mmlu-redux-2.0-biology/medical_genetics/"),
        "question", "options",
    ),
    (
        "mmlu-redux/high-school-biology",
        os.path.join(HF_CACHE, "hf-carbon___mmlu-redux-2.0-biology/high_school_biology/"),
        "question", "options",
    ),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── shard reading ─────────────────────────────────────────────────────────────

def _iter_shard(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_shard(path: str, min_tokens: int, max_tokens: int):
    """Return list of (text, is_eligible) for one .jsonl.gz shard."""
    results = []
    try:
        for doc in _iter_shard(path):
            tc = doc.get("metadata", {}).get("token_count", 0)
            text = doc.get("text", "")
            results.append((text, (min_tokens <= tc <= max_tokens) and bool(text.strip())))
    except Exception:
        pass
    return results


def reservoir_sample_subfolder(
    subfolder: str,
    n: int,
    min_tokens: int = 64,
    max_tokens: int = 4096,
    seed: int = 42,
    n_workers: int = 16,
) -> tuple[list[str], int, int]:
    """Reservoir-sample up to n texts from all .jsonl.gz shards in a subfolder.

    Reads shards in parallel (n_workers threads) to handle datasets with many
    small files efficiently. Returns (texts, total_docs, eligible_docs).
    """
    shards = sorted(Path(subfolder).glob("*.jsonl.gz"))
    if not shards:
        log.warning(f"No .jsonl.gz files found in {subfolder}")
        return [], 0, 0

    rng = random.Random(seed)
    reservoir: list[str] = []
    total = 0
    eligible = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_read_shard, str(s), min_tokens, max_tokens): s
            for s in shards
        }
        for fut in tqdm(as_completed(futures), total=len(shards),
                        desc=f"  {Path(subfolder).name}", leave=False):
            for text, is_eligible in fut.result():
                total += 1
                if not is_eligible:
                    continue
                eligible += 1
                if len(reservoir) < n:
                    reservoir.append(text)
                else:
                    j = rng.randint(0, eligible - 1)
                    if j < n:
                        reservoir[j] = text

    return reservoir, total, eligible


# ── benchmark loaders ─────────────────────────────────────────────────────────

def _fmt_question(question: str, options) -> str:
    if isinstance(options, (list, np.ndarray)):
        return f"{question}\nOptions: {' | '.join(str(o) for o in options)}"
    return question


def load_labbench_texts() -> tuple[list[str], list[str]]:
    """Load LabBench questions (all subsets). Returns (texts, subset_labels)."""
    import pandas as pd

    texts, labels = [], []
    for subset in LABBENCH_SUBSETS:
        path = os.path.join(LABBENCH_ROOT, subset, "train-00000-of-00001.parquet")
        if not os.path.exists(path):
            log.warning(f"LabBench subset not found: {path}")
            continue
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            texts.append(_fmt_question(
                str(row.get("question", "")), row.get("options", [])
            ))
            labels.append(f"labbench-{subset.lower()}")
        log.info(f"  LabBench {subset}: {len(df)} texts")

    return texts, labels


def load_mmlu_bio_texts() -> tuple[list[str], list[str]]:
    """Load MMLU-Bio questions (all subsets). Returns (texts, subset_labels)."""
    import glob as _glob
    import pyarrow as pa

    texts, labels = [], []
    for label, path, q_col, opts_col in MMLU_BIO_SOURCES:
        arrow_path = path
        if os.path.isdir(path):
            files = _glob.glob(f"{path}/**/*.arrow", recursive=True)
            if not files:
                log.warning(f"MMLU-Bio {label}: no arrow files in {path}")
                continue
            arrow_path = files[0]
        if not os.path.exists(arrow_path):
            log.warning(f"MMLU-Bio {label}: not found at {arrow_path}")
            continue
        tbl = pa.ipc.open_stream(arrow_path).read_all()
        for i in range(len(tbl)):
            texts.append(_fmt_question(tbl[q_col][i].as_py(), tbl[opts_col][i].as_py()))
            labels.append(label)
        log.info(f"  MMLU-Bio {label}: {len(tbl)} texts")

    return texts, labels


# ── plotting ──────────────────────────────────────────────────────────────────

def _draw_training_clusters(ax, projections, labels, summaries, centers):
    import matplotlib.cm as cm

    labels_arr = np.array(labels)
    unique = sorted(set(l for l in labels if l != -1))
    n = len(unique)

    if n <= 20:
        palette = [cm.tab20(i / 20) for i in range(n)]
    elif n <= 40:
        palette = [cm.tab20(i / 20) for i in range(20)] + [cm.tab20b(i / 20) for i in range(n - 20)]
    else:
        palette = [cm.hsv(i / n) for i in range(n)]

    noise = labels_arr == -1
    if noise.any():
        ax.scatter(projections[noise, 0], projections[noise, 1],
                   c="lightgray", s=1, alpha=0.3, zorder=1)

    for i, label in enumerate(unique):
        mask = labels_arr == label
        ax.scatter(projections[mask, 0], projections[mask, 1],
                   c=[palette[i]], s=2, alpha=0.5, zorder=2)
        if summaries and label in summaries:
            cx, cy = centers[label]
            ax.text(cx, cy, str(summaries[label]),
                    fontsize=5, ha="center", va="center", color="black", zorder=4)


def _save_overlay_plot(
    ax, title: str, out_path: str, legend_outside: bool = False
):
    import matplotlib.pyplot as plt
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8,
                  markerscale=1.2, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    else:
        ax.legend(loc="upper right", fontsize=8, markerscale=1.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Plot → {out_path}")


def plot_benchmark_overlays(
    train_projections, train_labels, cluster_summaries, cluster_centers,
    lb_projections, lb_labels,
    mmlu_projections, mmlu_labels,
    out_dir: str,
):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    MARKERS = ["*", "D", "^", "P", "X", "o", "s", "v", "<", ">", "h", "H", "p", "8", "1"]

    # LabBench overlay
    fig, ax = plt.subplots(figsize=(14, 10))
    _draw_training_clusters(ax, train_projections, train_labels, cluster_summaries, cluster_centers)
    lb_colors = ["red", "orange", "deepskyblue", "lime", "magenta"]
    for i, subset in enumerate(sorted(set(lb_labels))):
        idx = [j for j, s in enumerate(lb_labels) if s == subset]
        ax.scatter(lb_projections[idx, 0], lb_projections[idx, 1],
                   c=lb_colors[i % len(lb_colors)], marker=MARKERS[i % len(MARKERS)],
                   s=120, alpha=0.9, zorder=5, edgecolors="black", linewidths=0.5,
                   label=subset)
    _save_overlay_plot(ax, "Training clusters + LabBench overlay",
                       os.path.join(out_dir, "cluster_plot_labbench_overlay.png"))

    # MMLU-Bio overlay
    if mmlu_projections is not None and len(mmlu_projections):
        fig, ax = plt.subplots(figsize=(16, 11))
        _draw_training_clusters(ax, train_projections, train_labels, cluster_summaries, cluster_centers)
        mmlu_unique = sorted(set(mmlu_labels))
        mmlu_palette = [cm.tab20(i / 20) for i in range(min(len(mmlu_unique), 20))]
        for i, subset in enumerate(mmlu_unique):
            idx = [j for j, s in enumerate(mmlu_labels) if s == subset]
            ax.scatter(mmlu_projections[idx, 0], mmlu_projections[idx, 1],
                       c=[mmlu_palette[i % len(mmlu_palette)]],
                       marker=MARKERS[i % len(MARKERS)],
                       s=130, alpha=0.9, zorder=5, edgecolors="black", linewidths=0.4,
                       label=f"{subset} (n={len(idx)})")
        _save_overlay_plot(ax, "Training clusters + MMLU-Bio overlay",
                           os.path.join(out_dir, "cluster_plot_mmlu_bio_overlay.png"),
                           legend_outside=True)


# ── main ──────────────────────────────────────────────────────────────────────

def run_clustering(args):
    sys.path.insert(0, TEXT_CLUSTERING_SRC)
    from src.text_clustering import ClusterClassifier

    dataset_dir = Path(args.dataset_dir)
    subfolders = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    if not subfolders:
        log.error(f"No subfolders found in {dataset_dir}")
        sys.exit(1)

    log.info(f"Found {len(subfolders)} subfolders: {[s.name for s in subfolders]}")

    # Sample training texts
    train_texts = []
    for sf in subfolders:
        log.info(f"Sampling {args.n_per_subfolder:,} from {sf.name} ...")
        texts, total, eligible = reservoir_sample_subfolder(
            str(sf), n=args.n_per_subfolder,
            min_tokens=args.min_tokens, max_tokens=args.max_tokens, seed=args.seed,
        )
        log.info(f"  → {len(texts):,} sampled / {eligible:,} eligible / {total:,} total")
        train_texts.extend(texts)

    random.Random(args.seed).shuffle(train_texts)
    n_train = len(train_texts)
    log.info(f"Total training texts: {n_train:,}")

    # Load benchmarks
    log.info("Loading LabBench ...")
    lb_texts, lb_labels = load_labbench_texts()
    log.info(f"  → {len(lb_texts)} texts")

    log.info("Loading MMLU-Bio ...")
    mmlu_texts, mmlu_labels = load_mmlu_bio_texts()
    log.info(f"  → {len(mmlu_texts)} texts")

    # Output dir
    out_dir = os.path.join(args.output_root, args.dataset_name,
                           f"clusters_eps{args.dbscan_eps:.2f}")
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"Output → {out_dir}")

    # Joint embedding: training + benchmarks in one pass
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Embedding device: {device}")

    cc = ClusterClassifier(
        embed_model_name="all-MiniLM-L6-v2",
        embed_device=device,
        embed_batch_size=256,
        embed_max_seq_length=512,
        summary_create=args.summary_create,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )

    all_texts = train_texts + lb_texts + mmlu_texts
    log.info(f"Embedding {len(all_texts):,} texts (training + LabBench + MMLU-Bio) ...")
    all_embeddings = cc.embed(all_texts)

    n_lb, n_mmlu = len(lb_texts), len(mmlu_texts)
    train_embeddings = all_embeddings[:n_train]

    # Fit UMAP on full joint corpus so benchmarks land in the training space
    log.info("Fitting UMAP ...")
    all_projections, _ = cc.project(all_embeddings)
    train_projections = all_projections[:n_train]
    lb_projections    = all_projections[n_train:n_train + n_lb]
    mmlu_projections  = all_projections[n_train + n_lb:]

    # DBSCAN on training texts only
    log.info("Running DBSCAN ...")
    cc.embeddings    = train_embeddings
    cc.projections   = train_projections
    cc.texts         = train_texts
    cc.faiss_index   = cc.build_faiss_index(train_embeddings)
    train_labels     = cc.cluster(train_projections)
    cc.cluster_labels = train_labels

    cc.id2cluster = {i: l for i, l in enumerate(train_labels)}
    cc.label2docs = defaultdict(list)
    for i, l in enumerate(train_labels):
        cc.label2docs[l].append(i)
    cc.cluster_centers = {
        l: (float(np.mean([train_projections[d, 0] for d in docs])),
            float(np.mean([train_projections[d, 1] for d in docs])))
        for l, docs in cc.label2docs.items()
    }
    cc.cluster_summaries = None
    cc.save(out_dir)

    # Save benchmark projections
    np.save(os.path.join(out_dir, "labbench_projections.npy"), lb_projections)
    with open(os.path.join(out_dir, "labbench_subset_labels.json"), "w") as f:
        json.dump(lb_labels, f)
    np.save(os.path.join(out_dir, "mmlu_bio_projections.npy"), mmlu_projections)
    with open(os.path.join(out_dir, "mmlu_bio_subset_labels.json"), "w") as f:
        json.dump(mmlu_labels, f)

    # Cluster stats
    labels_arr = np.array(train_labels)
    n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    n_noise    = int(np.sum(labels_arr == -1))
    with open(os.path.join(out_dir, "cluster_stats.json"), "w") as f:
        json.dump({
            "dataset":             args.dataset_name,
            "dataset_dir":         str(dataset_dir),
            "n_subfolders":        len(subfolders),
            "n_per_subfolder":     args.n_per_subfolder,
            "token_range":         [args.min_tokens, args.max_tokens],
            "n_train_texts":       n_train,
            "n_labbench_texts":    n_lb,
            "n_mmlu_bio_texts":    n_mmlu,
            "n_clusters":          n_clusters,
            "n_noise":             n_noise,
            "noise_pct":           round(100 * n_noise / max(n_train, 1), 2),
            "dbscan_eps":          args.dbscan_eps,
            "dbscan_min_samples":  args.dbscan_min_samples,
            "cluster_sizes": {
                str(k): len(v)
                for k, v in sorted(cc.label2docs.items(), key=lambda x: -len(x[1]))
                if k != -1
            },
        }, f, indent=2)

    # Plots
    plot_benchmark_overlays(
        train_projections, train_labels, cc.cluster_summaries or {}, cc.cluster_centers,
        lb_projections, lb_labels,
        mmlu_projections, mmlu_labels,
        out_dir,
    )

    log.info(f"Done: {n_clusters} clusters, noise={n_noise} ({100*n_noise//max(n_train,1)}%)")
    log.info(f"Results → {out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir",   required=True,
                        help="Directory with subfolders of .jsonl.gz shards")
    parser.add_argument("--dataset_name",  required=True,
                        help="Name used for the output subdirectory")
    parser.add_argument("--n_per_subfolder", type=int, default=50_000,
                        help="Texts to reservoir-sample per subfolder (default: 50 000)")
    parser.add_argument("--output_root",   default="./output",
                        help="Root directory for outputs (default: ./output)")
    parser.add_argument("--dbscan_eps",    type=float, default=0.15)
    parser.add_argument("--dbscan_min_samples", type=int, default=50)
    parser.add_argument("--min_tokens",    type=int, default=64)
    parser.add_argument("--max_tokens",    type=int, default=4096)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--summary_create", action="store_true",
                        help="Generate LLM cluster-topic labels (slow)")
    return parser.parse_args()


if __name__ == "__main__":
    run_clustering(get_args())
