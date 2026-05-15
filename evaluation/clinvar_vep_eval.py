"""
ClinVar zero-shot variant-effect prediction (Carbon production recipe).

Scoring is **next-token at the right end** of a left-context window — the
recipe Carbon's reported ClinVar numbers come from, originally from the
GENERator paper. This is *different* from the BRCA / TraitGym recipe in
`vep_eval.py`, which uses a centered 8 kb window and a full-sequence LL delta:

  For each variant:
    1. Build a left-context window: hg38[chrom][pos - ctx : pos]  (variant base
       is the FIRST base of the *next* 6-mer, sitting at the right end).
    2. Strip leading N's, trim to a multiple of 6 so the tokenizer boundary
       lands exactly at the variant position.
    3. Forward pass; take softmax of the last token's logits.
    4. Marginalise over 6-mer tokens by first base: P(ref), P(alt) = sum of
       probabilities over all tokens starting with the ref / alt nucleotide.
    5. Score = log(P(ref) / P(alt)) — higher when the alt is more surprising
       (i.e. the variant is pathogenic). AUROC of score vs label==1.

Dataset: `HuggingFaceBio/clinvar-vep-final` (GenerTeam coding ClinVar + Carbon's
extra noncoding curation, since GenerTeam's release is ~99% coding).

Backends & flags follow the rest of the evaluation suite:
  --backend hf       Carbon, GENERator, any HF causal LM
  --backend evo2     official Evo2 inference library
  --add_dna_tag      prepend <dna> (Carbon hybrid models)
  --add_bos          prepend <s> (GENERator pure-DNA)
  (default)          no prefix — Evo2

Example:
  # Carbon 3B hybrid (flagship, 8 GPUs, 24 kb context)
  python clinvar_vep_eval.py \
      --model HuggingFaceBio/Carbon-3B \
      --add_dna_tag --bf16 --context_length 24000 \
      --output_dir ./results/clinvar
  
  # GENERator
  python clinvar_vep_eval.py \
      --model GenerTeam/GENERator-v2-eukaryote-3b-base \
      --add_bos --bf16 --context_length 24000 \
      --output_dir ./results/clinvar

  # Evo2 7B
  python clinvar_vep_eval.py \
      --model evo2_7b --backend evo2 --bf16 \
      --context_length 24000 --output_dir ./results/clinvar_evo2
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_HG38 = "hf://datasets/GenerTeam/variant-effect-prediction/hg38.parquet"
DEFAULT_CLINVAR = "hf://datasets/HuggingFaceBio/clinvar-vep-final/clinvar_vep_final.parquet"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF repo / local path / evo2 model name")
    p.add_argument("--revision", default=None)
    p.add_argument("--backend", choices=["hf", "evo2"], default="hf")
    p.add_argument("--hg38_path", default=DEFAULT_HG38, help="hg38 reference parquet (chrom -> sequence)")
    p.add_argument("--clinvar_path", default=DEFAULT_CLINVAR)
    p.add_argument("--context_length", type=int, default=24000,
                   help="Left-context length in bp (variant sits just past the right end).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_processes", type=int, default=16,
                   help="CPU procs for the per-variant probability marginalisation.")
    p.add_argument("--output_dir", default="./results/clinvar")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--add_dna_tag", action="store_true", help="Prepend <dna> (Carbon hybrid)")
    p.add_argument("--add_bos", action="store_true", help="Prepend <s> (GENERator pure-DNA)")
    return p.parse_args()

def _prefix(args) -> str:
    if args.add_dna_tag:
        return "<dna>"
    if args.add_bos:
        return "<s>"
    return ""

def load_and_extract(hg38_path: str, clinvar_path: str, context_length: int) -> pd.DataFrame:
    """Load ClinVar + hg38 and build the left-context sequence for each variant."""
    print(f"Loading hg38 reference from {hg38_path}")
    seq_df = pd.read_parquet(hg38_path)
    chrom_to_seq = dict(zip(seq_df["ID"].tolist(), seq_df["Sequence"].tolist()))
    del seq_df

    print(f"Loading ClinVar from {clinvar_path}")
    df = pd.read_parquet(clinvar_path)
    print(f"  {len(df):,} variants")
    if "region" in df.columns:
        print(f"  region: {df['region'].value_counts().to_dict()}")

    chroms = df["chrom"].tolist()
    positions = df["pos"].tolist()
    sequences = []
    for chrom, pos in tqdm(zip(chroms, positions), total=len(df), desc="extract"):
        full = chrom_to_seq["chr" + str(chrom)]
        seq = full[max(0, pos - 1 - context_length) : pos - 1]  # 1-indexed → 0-indexed; variant is the NEXT base
        seq = seq.lstrip("N")
        # Trim from the left to a multiple of 6 so the tokenizer boundary lands
        # exactly at the variant position (for 6-mer tokenizers).
        extra = len(seq) % 6
        if extra:
            seq = seq[extra:]
        sequences.append(seq)

    df = df.copy()
    df["sequence"] = sequences
    df["hash"] = [hashlib.md5(f"{s}_{i}".encode()).hexdigest()[:16]
                  for i, s in enumerate(sequences)]
    print(f"  avg context length: {np.mean([len(s) for s in sequences]):.0f} bp")
    return df


def _hf_shard(args):
    """One-GPU worker: forward pass, return last-token softmax probabilities per variant."""
    shard_id, records, model, revision, dtype = args
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
    with tqdm(total=len(records), desc=f"gpu{shard_id}", unit="seq") as pbar:
        for i in range(0, len(records), args_batch_size):
            batch = records[i : i + args_batch_size]
            seqs = [r["sequence"] for r in batch]
            enc = tok(seqs, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = m(**enc).logits
            for j, r in enumerate(batch):
                # Index of the last real (non-pad) token, skipping a trailing EOS if present
                ids = tok(r["sequence"]).get("input_ids", [])
                offset = 2 if ids and ids[-1] == tok.eos_token_id else 1
                last = logits[j, len(ids) - offset, :]
                probs = F.softmax(last, dim=0).cpu().float().numpy().tolist()
                out.append({"hash": r["hash"], "probs": probs})
            pbar.update(len(batch))
    return out


# Set at runtime so subprocesses pick up the batch size — multiprocessing
# doesn't share argparse state.
args_batch_size = 4


def compute_probs_hf(df: pd.DataFrame, model: str, revision: str, dtype: str, batch_size: int):
    global args_batch_size
    args_batch_size = batch_size

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPU available")
    print(f"Sharding across {n_gpus} GPUs")

    records = df[["sequence", "hash"]].to_dict("records")
    shard_size = (len(records) + n_gpus - 1) // n_gpus
    work = [
        (g, records[g * shard_size : (g + 1) * shard_size], model, revision, dtype)
        for g in range(n_gpus)
        if g * shard_size < len(records)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_gpus) as pool:
        results = list(pool.imap(_hf_shard, work))
    hash_to_probs = {item["hash"]: item["probs"] for shard in results for item in shard}
    return [hash_to_probs[h] for h in df["hash"]]


def get_first_base_indices(tokenizer) -> dict:
    """Map each DNA base ∈ {A,C,G,T} to the token ids whose first character is that base.

    Marginalising P(next base = X) = sum of P(token) over all tokens starting with X.
    Works for both 6-mer (Carbon) and BPE (GENERator) tokenizers — anything where
    a DNA token is a string of A/C/G/T.
    """
    vocab = tokenizer.get_vocab()
    bases = set("ACGT")
    out = {b: [] for b in bases}
    for token, tid in vocab.items():
        if isinstance(token, str) and token and all(c in bases for c in token):
            out[token[0]].append(tid)
    return out


def _marginalise_one(args):
    ref, alt, probs, first_base = args
    p_ref = sum(probs[i] for i in first_base.get(ref, []) if i < len(probs))
    p_alt = sum(probs[i] for i in first_base.get(alt, []) if i < len(probs))
    return p_ref, p_alt


def marginalise_probs(df: pd.DataFrame, probs_list, tokenizer, num_processes: int):
    first_base = get_first_base_indices(tokenizer)
    args_list = [(df["ref"].iloc[i], df["alt"].iloc[i], probs_list[i], first_base)
                 for i in range(len(df))]
    chunksize = max(1, len(args_list) // (num_processes * 4))
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(_marginalise_one, args_list, chunksize=chunksize),
                            total=len(args_list), desc="marginalise"))
    p_ref, p_alt = zip(*results)
    return np.array(p_ref), np.array(p_alt)


def _evo2_shard(args):
    """One-GPU Evo2 worker: last-token softmax, then read P(ref) / P(alt) directly."""
    shard_id, sequences, refs, alts, model_name, batch_size = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(shard_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from evo2_runtime import preload_cudnn_libraries

    preload_cudnn_libraries()
    from evo2 import Evo2
    from evo2.scoring import prepare_batch

    torch.cuda.set_device(0)
    model = Evo2(model_name)

    base_ids = {}
    for b in "ACGT":
        ids = model.tokenizer.tokenize(b)
        base_ids[b] = ids[0] if ids else None

    p_ref, p_alt = [], []
    for i in tqdm(range(0, len(sequences), batch_size), desc=f"evo2 gpu{shard_id}"):
        batch_seqs = sequences[i : i + batch_size]
        batch_refs = refs[i : i + batch_size]
        batch_alts = alts[i : i + batch_size]
        input_ids, seq_lengths = prepare_batch(batch_seqs, model.tokenizer, device="cuda:0")
        with torch.inference_mode():
            output, _ = model(input_ids)
            logits = output[0] if isinstance(output, tuple) else output
        for j in range(len(batch_seqs)):
            last = logits[j, seq_lengths[j] - 1, :]
            probs = torch.softmax(last, dim=0)
            rid, aid = base_ids.get(batch_refs[j]), base_ids.get(batch_alts[j])
            p_ref.append(float(probs[rid]) if rid is not None else 0.0)
            p_alt.append(float(probs[aid]) if aid is not None else 0.0)
    return p_ref, p_alt


def compute_probs_evo2(df: pd.DataFrame, model_name: str, batch_size: int):
    sequences = df["sequence"].tolist()
    refs = df["ref"].tolist()
    alts = df["alt"].tolist()
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPU available")
    shard_size = (len(sequences) + n_gpus - 1) // n_gpus
    work = [
        (g, sequences[g * shard_size : (g + 1) * shard_size],
         refs[g * shard_size : (g + 1) * shard_size],
         alts[g * shard_size : (g + 1) * shard_size],
         model_name.split("/")[-1], batch_size)
        for g in range(n_gpus)
        if g * shard_size < len(sequences)
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(work)) as pool:
        results = list(pool.imap(_evo2_shard, work))
    p_ref, p_alt = [], []
    for r, a in results:
        p_ref.extend(r)
        p_alt.extend(a)
    return np.array(p_ref), np.array(p_alt)


def main():
    args = parse_args()
    dtype_str = "bfloat16" if args.bf16 else "float32"

    print("=" * 70)
    print(f"ClinVar VEP (right-end / next-token) · model={args.model} · backend={args.backend}")
    print("=" * 70)

    # Sanity-check context length for HF models (6 bp per token for 6-mer)
    if args.backend == "hf":
        cfg = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
        max_bp = cfg.max_position_embeddings * 6
        if args.context_length > max_bp:
            raise ValueError(
                f"context_length={args.context_length} bp > model max "
                f"{cfg.max_position_embeddings} tokens × 6 = {max_bp} bp"
            )

    df = load_and_extract(args.hg38_path, args.clinvar_path, args.context_length)

    prefix = _prefix(args)
    df["sequence"] = df["sequence"].apply(lambda s: prefix + s)

    t0 = time.time()
    if args.backend == "evo2":
        p_ref, p_alt = compute_probs_evo2(df, args.model, args.batch_size)
    else:
        probs = compute_probs_hf(df, args.model, args.revision, dtype_str, args.batch_size)
        from transformers_compat import patch_generator_sample, patch_legacy_tokenizer_base

        patch_legacy_tokenizer_base()
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
        p_ref, p_alt = marginalise_probs(df, probs, tok, args.num_processes)
    print(f"Scoring took {time.time() - t0:.1f}s")

    df["p_ref"] = p_ref
    df["p_alt"] = p_alt
    df["score"] = np.log(p_ref / (p_alt + 1e-10))

    labels = df["label"].astype(int).values
    auroc = roc_auc_score(labels, df["score"].values)
    precision, recall, _ = precision_recall_curve(labels, df["score"].values)
    auprc = auc(recall, precision)
    print("=" * 70)
    print(f"AUROC: {auroc:.4f}   AUPRC: {auprc:.4f}   (n={len(df)})")

    # Optional per-region / per-variant_type breakdowns (if present in dataset)
    breakdowns = {}
    for col in ("region", "variant_type"):
        if col in df.columns:
            for v in sorted(df[col].dropna().unique()):
                mask = (df[col] == v).values
                y, s = labels[mask], df["score"].values[mask]
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                pr, rc, _ = precision_recall_curve(y, s)
                breakdowns[f"{col}={v}"] = {
                    "n": int(mask.sum()),
                    "AUROC": float(roc_auc_score(y, s)),
                    "AUPRC": float(auc(rc, pr)),
                }
                print(f"  [{col}={v:>11}]  n={mask.sum():>6d}   "
                      f"AUROC={breakdowns[f'{col}={v}']['AUROC']:.4f}   "
                      f"AUPRC={breakdowns[f'{col}={v}']['AUPRC']:.4f}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = args.model.split("/")[-1]
    base = f"{model_tag}_{dtype_str}"
    parquet = os.path.join(args.output_dir, f"{base}.parquet")
    summary = os.path.join(args.output_dir, f"{base}.json")
    df.drop(columns=["sequence"]).to_parquet(parquet)
    with open(summary, "w") as f:
        json.dump(
            {
                "model": args.model,
                "revision": args.revision,
                "backend": args.backend,
                "context_length": args.context_length,
                "num_variants": int(len(df)),
                "AUROC": float(auroc),
                "AUPRC": float(auprc),
                "breakdowns": breakdowns,
                "dtype": dtype_str,
            },
            f,
            indent=2,
        )
    print(f"Saved {parquet}\n      {summary}")


if __name__ == "__main__":
    main()
