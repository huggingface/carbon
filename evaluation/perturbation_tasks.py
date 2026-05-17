"""
Sequence-level perturbation tasks: TATA perturbation and synonymous codon substitution.

Both apply a structural perturbation to a real biological sequence (motif
disruption or codon swap) and ask whether the model assigns higher
log-likelihood to the unperturbed version. Distinct from VEP, which scores
single-nucleotide variants and reports AUROC of the LL delta.

  tata_perturbation:
    Disrupt TATA-box motifs in promoters with random substitutions. The model
    should score the intact promoter higher than the perturbed one.
    Dataset: HuggingFaceBio/carbon-perturbation-bench  cols: original_sequence (real), sequence (perturbed)

  synonymous_codon_substitution:
    Replace codons in a CDS with synonyms encoding the same amino acid. The
    real codon usage should be preferred over the synonymous variant.
    Dataset: HuggingFaceBio/carbon-perturbation-bench  cols: original_sequence (real), sequence (synonymous)

Metric: pairwise discrimination accuracy = mean(LL(real) > LL(perturbed)).

Backends and tag flags work the same way as the other Carbon evals:
  --backend hf       Carbon, GENERator, any HF causal LM
  --backend evo2     official Evo2 inference library

Example:
  python perturbation_tasks.py \
      --task tata_perturbation \
      --model HuggingFaceBio/Carbon-3B \
      --bf16

  python perturbation_tasks.py \
      --task tata_perturbation \
      --model GenerTeam/GENERator-v2-eukaryote-3b-base \
      --bf16

  python perturbation_tasks.py \
      --task synonymous_codon_substitution \
      --model evo2_7b --backend evo2 --bf16
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# pos_col (real, unperturbed) and neg_col (perturbed) for each task,
# plus the matching subset name in HuggingFaceBio/carbon-perturbation-bench.
TASKS = {
    "motif_human":             {"pos": "original_sequence", "neg": "sequence", "subset": "motif_human"},
    "syn_human":               {"pos": "original_sequence", "neg": "sequence", "subset": "syn_human"},
    "syn_mouse":               {"pos": "original_sequence", "neg": "sequence", "subset": "syn_mouse"},
    "promoter_revcomp":        {"pos": "original_sequence", "neg": "sequence", "subset": "promoter_revcomp"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--model", required=True, help="HF repo / local path / evo2 model name")
    p.add_argument("--revision", default=None)
    p.add_argument("--backend", choices=["hf", "evo2"], default="hf")
    p.add_argument("--dataset", default="HuggingFaceBio/carbon-perturbation-bench")
    p.add_argument("--subset", default=None,
                   help="HF dataset config. Defaults to the per-task subset "
                        "(`tata` / `synonymous_codons`).")
    p.add_argument("--split", default="test")
    p.add_argument("--output_dir", default="./results/perturbation_tasks")
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def load_df(args) -> pd.DataFrame:
    if args.dataset.endswith(".parquet") or args.dataset.startswith("hf://"):
        return pd.read_parquet(args.dataset)
    return load_dataset(args.dataset, args.subset, split=args.split).to_pandas()


def _shard_worker(args):
    """Score one GPU shard using bp-level score_sequence. Returns [(uid, log_likelihood), ...]."""
    shard_id, items, model, revision, dtype, batch_size, max_length = args
    torch.cuda.set_device(shard_id)
    device = f"cuda:{shard_id}"

    from transformers_compat import patch_generator_sample, patch_legacy_tokenizer_base

    patch_legacy_tokenizer_base()
    tok = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        model, revision=revision, trust_remote_code=True, dtype=getattr(torch, dtype)
    ).to(device).eval()
    patch_generator_sample(m)

    out = []
    with tqdm(total=len(items), desc=f"gpu{shard_id}", unit="seq") as pbar:
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            seqs = [b["seq"] for b in batch]
            # Truncate sequences if needed
            seqs = [s[:max_length] if len(s) > max_length else s for s in seqs]

            # Use score_sequence for bp-level scoring
            with torch.no_grad():
                if len(seqs) == 1:
                    _, actual_probs = m.score_sequence(seqs[0])
                    actual_probs_list = [actual_probs]
                else:
                    _, actual_probs_list = m.score_sequence(seqs)

            # Compute mean log-prob per base
            for j, b in enumerate(batch):
                mean_logp = torch.log(actual_probs_list[j]).mean().item()
                out.append((b["uid"], float(mean_logp)))

            pbar.update(len(batch))
    return out


@torch.no_grad()
def score_hf(seqs, model, revision, dtype_str, batch_size: int, max_length: int):
    """Multi-GPU bp-level scoring across all sequences."""
    items = [{"uid": str(i), "seq": s} for i, s in enumerate(seqs)]

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPU available")
    print(f"  sharding {len(items)} sequences across {n_gpus} GPUs")

    shard_size = (len(items) + n_gpus - 1) // n_gpus
    work = [
        (g, items[g * shard_size : (g + 1) * shard_size], model, revision, dtype_str, batch_size, max_length)
        for g in range(n_gpus)
        if g * shard_size < len(items)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_gpus) as pool:
        results = list(pool.imap(_shard_worker, work))

    uid_to_logp = {uid: lp for shard in results for uid, lp in shard}
    return [uid_to_logp[str(i)] for i in range(len(seqs))]


def score_evo2(seqs, batch_size: int, model_name: str):
    from evo2_runtime import preload_cudnn_libraries

    preload_cudnn_libraries()
    from evo2 import Evo2

    torch.cuda.set_device(0)
    model = Evo2(model_name)
    return model.score_sequences(
        seqs, batch_size=batch_size, reduce_method="mean", average_reverse_complement=False
    )


def main():
    args = parse_args()
    dtype_str = "bfloat16" if args.bf16 else "float32"

    print("=" * 70)
    print(f"Perturbation task · task={args.task} · model={args.model} · backend={args.backend}")
    print("=" * 70)

    task_cfg = TASKS[args.task]
    if args.subset is None:
        args.subset = task_cfg["subset"]
    df = load_df(args)
    pos_col, neg_col = task_cfg["pos"], task_cfg["neg"]
    pos = df[pos_col].astype(str).tolist()
    neg = df[neg_col].astype(str).tolist()
    print(f"Loaded {len(df)} pairs ({pos_col} vs {neg_col})")

    # Combine pos and neg sequences to score in one pass
    all_seqs = pos + neg
    n_pairs = len(pos)

    t0 = time.time()
    if args.backend == "evo2":
        model_name = args.model.split("/")[-1]
        all_scores = score_evo2(all_seqs, args.batch_size, model_name)
    else:
        dtype_str = "bfloat16" if args.bf16 else "float32"
        all_scores = score_hf(all_seqs, args.model, args.revision, dtype_str, args.batch_size, args.max_length)

    # Split back into pos and neg
    pos_scores = all_scores[:n_pairs]
    neg_scores = all_scores[n_pairs:]
    print(f"Scoring took {time.time() - t0:.1f}s")

    # Non-strict comparator (ties count as correct) — matches the production
    # cds_half_shuffle_eval.py recipe used to generate Carbon's reported numbers.
    correct = [int(p > n) for p, n in zip(pos_scores, neg_scores)]
    acc = sum(correct) / max(len(correct), 1)
    print(f"\nDiscrimination accuracy: {acc:.4f}  (n={len(correct)})")

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = args.model.split("/")[-1]
    base = f"{model_tag}_{args.task}_{dtype_str}"
    out_df = pd.DataFrame(
        {
            "pos_hash": [_hash(s) for s in pos],
            "neg_hash": [_hash(s) for s in neg],
            "pos_score": pos_scores,
            "neg_score": neg_scores,
            "correct": correct,
        }
    )
    parquet = os.path.join(args.output_dir, f"{base}.parquet")
    summary = os.path.join(args.output_dir, f"{base}.json")
    out_df.to_parquet(parquet)
    with open(summary, "w") as f:
        json.dump(
            {
                "model": args.model,
                "revision": args.revision,
                "task": args.task,
                "dataset": args.dataset,
                "backend": args.backend,
                "accuracy": acc,
                "num_examples": len(correct),
                "dtype": dtype_str,
            },
            f,
            indent=2,
        )
    print(f"Saved {parquet}\n      {summary}")


if __name__ == "__main__":
    main()
