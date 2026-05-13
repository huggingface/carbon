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
  --add_dna_tag      prepend <dna> (Carbon hybrid models)
  (default)          no prefix — GENERator and Evo2

Example:
  python perturbation_tasks.py \
      --task tata_perturbation \
      --model HuggingFaceBio/Carbon-3B \
      --add_dna_tag --bf16

  python perturbation_tasks.py \
      --task synonymous_codon_substitution \
      --model evo2_7b_base --backend evo2 --bf16
"""

import argparse
import hashlib
import json
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
    "tata_perturbation":             {"pos": "original_sequence", "neg": "sequence", "subset": "tata"},
    "synonymous_codon_substitution": {"pos": "original_sequence", "neg": "sequence", "subset": "synonymous_codons"},
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
    p.add_argument("--split", default="train")
    p.add_argument("--output_dir", default="./results/perturbation_tasks")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--add_dna_tag", action="store_true", help="Wrap with <dna>...</dna> (Carbon hybrid)")
    return p.parse_args()


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def load_df(args) -> pd.DataFrame:
    if args.dataset.endswith(".parquet") or args.dataset.startswith("hf://"):
        return pd.read_parquet(args.dataset)
    return load_dataset(args.dataset, args.subset, split=args.split).to_pandas()


def wrap(seqs, add_dna_tag: bool):
    return [f"<dna>{s}</dna>" if add_dna_tag else s for s in seqs]


@torch.no_grad()
def score_hf(model, tok, seqs, max_length: int, batch_size: int):
    """Mean log-prob per token."""
    out = []
    for i in tqdm(range(0, len(seqs), batch_size), desc="scoring"):
        batch = seqs[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, add_special_tokens=False)
        ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)
        logits = model(ids).logits[:, :-1, :]
        targets = ids[:, 1:]
        mask = attn[:, 1:]
        logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        out.extend(((logp * mask).sum(dim=1) / denom).tolist())
    return out


def score_evo2(seqs, batch_size: int, model_name: str):
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
    pos = wrap(df[pos_col].astype(str).tolist(), args.add_dna_tag)
    neg = wrap(df[neg_col].astype(str).tolist(), args.add_dna_tag)
    print(f"Loaded {len(df)} pairs ({pos_col} vs {neg_col})")

    t0 = time.time()
    if args.backend == "evo2":
        model_name = args.model.split("/")[-1]
        pos_scores = score_evo2(pos, args.batch_size, model_name)
        neg_scores = score_evo2(neg, args.batch_size, model_name)
    else:
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            device_map="auto",
        )
        pos_scores = score_hf(model, tok, pos, args.max_length, args.batch_size)
        neg_scores = score_hf(model, tok, neg, args.max_length, args.batch_size)
    print(f"Scoring took {time.time() - t0:.1f}s")

    # Non-strict comparator (ties count as correct) — matches the production
    # cds_half_shuffle_eval.py recipe used to generate Carbon's reported numbers.
    correct = [int(p >= n) for p, n in zip(pos_scores, neg_scores)]
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
