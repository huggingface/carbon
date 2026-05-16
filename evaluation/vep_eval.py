"""
Zero-shot variant-effect prediction (VEP) for BRCA2 and TraitGym.

For each SNV: score the full 8,192 bp window log-likelihood for the reference
and variant sequences. Use delta = LL(var) - LL(ref) as the variant score.
Report AUROC of -delta against (class == "LOF") + Spearman ρ vs the
continuous functional score where available.

The eval is dataset-agnostic — any parquet with the schema
  chrom, pos, ref, alt, score, class, ref_seq, var_seq
works. We ship two prep scripts that produce this schema:
  - prep_brca2.py     → HuggingFaceBio/brca2-vep     (6,836 SNVs)
  - prep_traitgym.py  → HuggingFaceBio/traitgym      (Mendelian: 3,380 / Complex: 11,400)

For ClinVar, see clinvar_vep_eval.py — it uses a different scoring recipe
(next-token at the right end of a left-context window, instead of centered
+ full-LL delta).

References:
  BRCA2:    Huang et al. 2025, Nature s41586-024-08388-8    (Evo2 §A.3.15)
  TraitGym: Benegas, Eraslan & Song 2025, bioRxiv 2025.02.11.637758

Example:
  # Carbon 3B hybrid on BRCA2 (8 GPUs)
  python vep_eval.py \
      --model HuggingFaceBio/Carbon-3B \
      --data_path hf://datasets/HuggingFaceBio/brca2-vep/brca2_vep.parquet \
      --bf16 --output_dir ./results/brca2_vep
    
  # GENERator on BRCA2 (8 GPUs)
  python vep_eval.py \
      --model GenerTeam/GENERator-v2-eukaryote-3b-base \
      --data_path hf://datasets/HuggingFaceBio/brca2-vep/brca2_vep.parquet \
      --bf16 --output_dir ./results/brca2_vep

  # Evo2 on BRCA2 (1 GPU)
  python vep_eval.py \
      --model evo2_7b --backend evo2 \
      --data_path hf://datasets/HuggingFaceBio/brca2-vep/brca2_vep.parquet \
      --bf16 --output_dir ./results/brca2_vep
    
  # TraitGym Mendelian
  python vep_eval.py \
      --model HuggingFaceBio/Carbon-3B \
      --data_path hf://datasets/HuggingFaceBio/traitgym/mendelian_traits_vep.parquet \
      --bf16 --rev_comp_avg \
      --output_dir ./results/traitgym_mendelian
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True,
                   help="Parquet with columns: chrom,pos,ref,alt,score,class,ref_seq,var_seq")
    p.add_argument("--model", required=True, help="HF repo / local path / evo2 model name")
    p.add_argument("--revision", default=None)
    p.add_argument("--backend", choices=["hf", "evo2"], default="hf")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--rev_comp_avg", action="store_true",
                   help="Strand-symmetric scoring: score each variant on the forward "
                        "window AND its reverse-complement, then average the two deltas. "
                        "~2x compute. Recommended for TraitGym/ClinVar where variants "
                        "can sit on either strand; not needed for single-gene DMS (BRCA).")
    return p.parse_args()


_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _shard_worker(args):
    """Score one GPU shard using bp-level score_sequence. Returns [(uid, log_likelihood), ...]."""
    shard_id, items, model, revision, dtype, batch_size = args
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

            # Use score_sequence for bp-level scoring
            with torch.no_grad():
                if len(seqs) == 1:
                    _, actual_probs = m.score_sequence(seqs[0])
                    actual_probs_list = [actual_probs]
                else:
                    _, actual_probs_list = m.score_sequence(seqs)

            # Compute log-likelihood for each sequence
            for j, b in enumerate(batch):
                log_likelihood = torch.log(actual_probs_list[j]).mean().item()
                out.append((b["uid"], float(log_likelihood)))

            pbar.update(len(batch))
    return out


def score_hf(df: pd.DataFrame, args) -> tuple[np.ndarray, np.ndarray]:
    """Multi-GPU full-sequence LL scoring across ref+var sequences."""
    items = []
    for i, row in df.iterrows():
        items.append({"uid": f"{i}_ref", "seq": row["ref_seq"]})
        items.append({"uid": f"{i}_var", "seq": row["var_seq"]})

    # Dedup identical sequences (many variants share the same ref window)
    seen, uniq = {}, []
    for it in items:
        if it["seq"] not in seen:
            seen[it["seq"]] = it["uid"]
            uniq.append(it)
    print(f"  {len(items)} scoring requests, {len(uniq)} unique sequences")

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPU available")
    print(f"  sharding across {n_gpus} GPUs")

    shard_size = (len(uniq) + n_gpus - 1) // n_gpus
    dtype = "bfloat16" if args.bf16 else "float32"
    work = [
        (g, uniq[g * shard_size : (g + 1) * shard_size], args.model, args.revision, dtype, args.batch_size)
        for g in range(n_gpus)
        if g * shard_size < len(uniq)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_gpus) as pool:
        results = list(pool.imap(_shard_worker, work))

    uid_to_logp = {uid: lp for shard in results for uid, lp in shard}
    seq_to_logp = {it["seq"]: uid_to_logp[it["uid"]] for it in uniq if it["uid"] in uid_to_logp}

    ref = np.array([seq_to_logp[r["ref_seq"]] for _, r in df.iterrows()])
    var = np.array([seq_to_logp[r["var_seq"]] for _, r in df.iterrows()])
    return ref, var


def score_evo2(df: pd.DataFrame, args) -> tuple[np.ndarray, np.ndarray]:
    """Single-GPU full-sequence LL via the official Evo2 library."""
    from evo2_runtime import preload_cudnn_libraries

    preload_cudnn_libraries()
    from evo2 import Evo2

    model = Evo2(args.model.split("/")[-1])

    ref_seqs, ref_idx = [], {}
    var_seqs = []
    for _, row in df.iterrows():
        if row["ref_seq"] not in ref_idx:
            ref_idx[row["ref_seq"]] = len(ref_seqs)
            ref_seqs.append(row["ref_seq"])
        var_seqs.append(row["var_seq"])
    ref_lookup = np.array([ref_idx[r] for r in df["ref_seq"]])

    print(f"  scoring {len(ref_seqs)} unique reference sequences")
    ref_scores = np.array(model.score_sequences(ref_seqs))
    print(f"  scoring {len(var_seqs)} variant sequences")
    var_scores = np.array(model.score_sequences(var_seqs))
    return ref_scores[ref_lookup], var_scores


def main():
    args = parse_args()
    dtype_str = "bfloat16" if args.bf16 else "float32"

    print("=" * 70)
    print(f"Zero-shot VEP · model={args.model} · backend={args.backend}")
    print("=" * 70)

    df = pd.read_parquet(args.data_path).copy()
    print(f"Loaded {len(df)} variants from {args.data_path}")
    print(f"  class distribution: {df['class'].value_counts().to_dict()}")

    score_fn = score_evo2 if args.backend == "evo2" else score_hf

    t0 = time.time()
    ref_logp, var_logp = score_fn(df, args)
    delta = var_logp - ref_logp

    if args.rev_comp_avg:
        # Strand-symmetric scoring: also score the reverse-complement of each
        # (ref, var) pair and average the two deltas. Carbon is trained on both
        # strands so reverse-complement is in-distribution.
        print("\n--- reverse-complement pass ---")
        df_rev = df.copy()
        df_rev["ref_seq"] = df["ref_seq"].apply(_revcomp)
        df_rev["var_seq"] = df["var_seq"].apply(_revcomp)
        ref_logp_rev, var_logp_rev = score_fn(df_rev, args)
        delta = (delta + (var_logp_rev - ref_logp_rev)) / 2
    print(f"Scoring took {time.time() - t0:.1f}s")

    df["ref_logp"] = ref_logp
    df["var_logp"] = var_logp
    df["delta"] = delta

    # AUROC + AUPRC on rows with a binary class label (LOF vs FUNC/INT)
    cls = df[df["class"].isin(["LOF", "FUNC/INT"])]
    y = (cls["class"] == "LOF").astype(int).values
    scores = -cls["delta"].values  # lower delta -> more LOF
    # Global AUPRC uses auc(recall, precision) — the convention used for the
    # BRCA2 number in Carbon's production reports.
    auroc, auprc = float("nan"), float("nan")
    if y.sum() and y.sum() < len(y):
        auroc = roc_auc_score(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        auprc = auc(recall, precision)

    # By-chromosome weighted AUROC + AUPRC — TraitGym leaderboard convention.
    # Per-chrom AUPRC uses `average_precision_score` (TraitGym uses sklearn's
    # canonical AP, not the auc(rc, pr) shortcut). Weighted by chromosome size.
    auroc_by_chrom, auprc_by_chrom = float("nan"), float("nan")
    if len(cls) and "chrom" in cls.columns:
        per_chrom = []
        for _, sub in cls.groupby("chrom"):
            yc = (sub["class"] == "LOF").astype(int).values
            if yc.sum() == 0 or yc.sum() == len(yc):
                continue
            sc = -sub["delta"].values
            per_chrom.append((len(sub), float(roc_auc_score(yc, sc)), float(average_precision_score(yc, sc))))
        if per_chrom:
            ws = np.array([p[0] for p in per_chrom], dtype=float)
            ws /= ws.sum()
            auroc_by_chrom = float(np.sum(ws * np.array([p[1] for p in per_chrom])))
            auprc_by_chrom = float(np.sum(ws * np.array([p[2] for p in per_chrom])))

    # Spearman ρ on rows with a continuous functional score (DMS / fine-mapping PIP)
    sp_df = df[df["score"].notna()]
    spearman = (
        float(sp_df[["score", "delta"]].corr(method="spearman").iloc[0, 1])
        if len(sp_df) > 1
        else float("nan")
    )

    # Multi-chromosome datasets (TraitGym, ClinVar genome-wide) report
    # by-chrom-weighted AUROC/AUPRC as the headline — it's the TraitGym
    # leaderboard convention and what every recent paper compares against.
    # Single-locus datasets (BRCA2) have one chromosome, so by-chrom
    # collapses to global; we just print global in that case.
    multi_chrom = not np.isnan(auroc_by_chrom)

    print("=" * 70)
    if multi_chrom:
        print(f"AUROC by-chrom weighted: {auroc_by_chrom:.4f}   AUPRC by-chrom weighted: {auprc_by_chrom:.4f}   (n_class={len(cls)})   ← headline (leaderboard convention)")
        print(f"AUROC global:            {auroc:.4f}   AUPRC global:            {auprc:.4f}")
    else:
        print(f"AUROC: {auroc:.4f}   AUPRC: {auprc:.4f}   (n_class={len(cls)})")
    if not np.isnan(spearman):
        print(f"Spearman ρ (delta vs functional score): {spearman:.4f}   (n_score={len(sp_df)})")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = args.model.split("/")[-1]
    base = f"{model_tag}_{dtype_str}"
    parquet = os.path.join(args.output_dir, f"{base}.parquet")
    summary = os.path.join(args.output_dir, f"{base}.json")
    df.drop(columns=["ref_seq", "var_seq"]).to_parquet(parquet)
    with open(summary, "w") as f:
        json.dump(
            {
                "model": args.model,
                "revision": args.revision,
                "backend": args.backend,
                "num_variants": int(len(df)),
                "num_class_labeled": int(len(cls)),
                "AUROC": None if np.isnan(auroc) else auroc,
                "AUPRC": None if np.isnan(auprc) else auprc,
                "AUROC_by_chrom_weighted": None if np.isnan(auroc_by_chrom) else auroc_by_chrom,
                "AUPRC_by_chrom_weighted": None if np.isnan(auprc_by_chrom) else auprc_by_chrom,
                "Spearman_score_vs_delta": None if np.isnan(spearman) else spearman,
                "dtype": dtype_str,
            },
            f,
            indent=2,
        )
    print(f"Saved {parquet}\n      {summary}")


if __name__ == "__main__":
    main()
