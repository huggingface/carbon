"""
Build the TraitGym zero-shot VEP dataset (Benegas, Eraslan & Song, bioRxiv 2025).

Source: songlab/TraitGym on HuggingFace. Two configs:
  - mendelian_traits: 3,380 variants (338 causal + 3,042 matched controls).
    113 monogenic OMIM diseases, non-coding regulatory variants.
  - complex_traits:   11,400 variants (1,140 causal + 10,260 matched controls).
    83 polygenic traits from UK BioBank fine-mapping (PIP > 0.9 = causal).

Methodology mirrors TraitGym's run_vep_evo2.py: 8,192 bp window centered on
each variant, hg38 reference. For genome-wide benchmarks like TraitGym where
variants can sit on either strand, pass --rev_comp_avg at eval time.

Output schema (matches vep_eval.py):
  chrom, pos, ref, alt, label (0/1), class ("LOF"/"FUNC/INT"), consequence,
  score (PIP for complex traits; None for mendelian), ref_seq, var_seq

Hub: HuggingFaceBio/traitgym  (one repo, two parquets: mendelian_traits_vep, complex_traits_vep)

Usage:
  python prep_traitgym.py --config mendelian_traits
  python prep_traitgym.py --config complex_traits --push_to_hub
"""

import argparse
import gzip
import os
import shutil
import urllib.request

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

WINDOW_SIZE = 8192
UCSC_HG38_TPL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr{}.fa.gz"
TRAITGYM_REPO = "songlab/TraitGym"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="mendelian_traits",
                   choices=["mendelian_traits", "complex_traits"])
    p.add_argument("--split", default="test")
    p.add_argument("--cache_dir", default="./traitgym_cache")
    p.add_argument("--output_path", default=None,
                   help="Defaults to {cache_dir}/{config}_vep.parquet")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo_id", default="HuggingFaceBio/traitgym")
    return p.parse_args()


def download(url: str, dst: str) -> None:
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"  downloading {url} -> {dst}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def load_fa_gz(path: str) -> str:
    parts = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                parts.append(line.strip().upper())
    return "".join(parts)


def window(chrom_seq: str, pos: int, ref: str, alt: str):
    p = pos - 1
    start = max(0, p - WINDOW_SIZE // 2)
    end = min(len(chrom_seq), p + WINDOW_SIZE // 2)
    ref_seq = chrom_seq[start:end]
    snv_pos = min(WINDOW_SIZE // 2, p)
    if snv_pos >= len(ref_seq) or ref_seq[snv_pos] != ref or len(ref_seq) != WINDOW_SIZE:
        return None, None
    var_seq = ref_seq[:snv_pos] + alt + ref_seq[snv_pos + 1:]
    return ref_seq, var_seq


def main():
    args = parse_args()
    output_path = args.output_path or os.path.join(args.cache_dir, f"{args.config}_vep.parquet")
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Loading {TRAITGYM_REPO}/{args.config}, split={args.split}")
    df = load_dataset(TRAITGYM_REPO, args.config, split=args.split).to_pandas()
    print(f"  total variants: {len(df)}")
    print(f"  label counts: {df['label'].value_counts().to_dict()}")

    # SNVs only (single-base ref/alt) — TraitGym already filters but be defensive
    snv_mask = (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)
    df = df[snv_mask].copy()
    print(f"  SNV-only filter: {len(df)}")

    chroms = sorted(df["chrom"].unique(), key=lambda c: (len(str(c)), str(c)))
    rows, n_skip = [], 0
    for chrom in chroms:
        sub = df[df["chrom"] == chrom]
        fa_path = os.path.join(args.cache_dir, f"chr{chrom}_hg38.fa.gz")
        download(UCSC_HG38_TPL.format(chrom), fa_path)
        chr_seq = load_fa_gz(fa_path)
        print(f"  chr{chrom}: {len(sub)} variants, reference {len(chr_seq):,} bp")

        for _, r in tqdm(sub.iterrows(), total=len(sub), desc=f"chr{chrom}", leave=False):
            rs, vs = window(chr_seq, int(r["pos"]), r["ref"].upper(), r["alt"].upper())
            if rs is None:
                n_skip += 1
                continue
            rows.append({
                "chrom": str(r["chrom"]),
                "pos": int(r["pos"]),
                "ref": r["ref"].upper(),
                "alt": r["alt"].upper(),
                "label": int(r["label"]),
                # class column expected by vep_eval.py
                "class": "LOF" if r["label"] else "FUNC/INT",
                "consequence": r.get("consequence"),
                "score": float(r["pip"]) if "pip" in df.columns and r["pip"] is not None else None,
                "ref_seq": rs,
                "var_seq": vs,
            })

    out = pd.DataFrame(rows)
    print(f"\nKept {len(out)} / {len(df)} variants  ({n_skip} dropped on ref mismatch / out-of-bounds)")
    print(f"  class distribution: {out['class'].value_counts().to_dict()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_parquet(output_path)
    print(f"Saved {output_path}  ({os.path.getsize(output_path) / 1e6:.1f} MB)")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hub_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=output_path,
            path_in_repo=f"{args.config}_vep.parquet",
            repo_id=args.hub_repo_id,
            repo_type="dataset",
        )
        print(f"  pushed to {args.hub_repo_id}/{args.config}_vep.parquet")


if __name__ == "__main__":
    main()
