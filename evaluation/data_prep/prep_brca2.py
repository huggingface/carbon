"""
Build the BRCA2 zero-shot VEP dataset (Huang et al. 2025).

Pipeline:
  1. Download the Nature 2025 supplementary Excel — ~7K BRCA2 DBD SNVs with
     ACMG-style functional classes (P/B strong/moderate/supporting + VUS).
  2. Download chr13 hg19 from UCSC goldenPath.
  3. Filter to P*/B* classes (drop VUS / NaN); collapse to {LOF, FUNC/INT}
     following Evo2 §A.3.15.
  4. Slice an 8,192 bp window centered on the variant; produce (ref_seq, var_seq).

Output schema (what vep_eval.py reads):
  chrom, pos, ref, alt, score, class, ref_seq, var_seq

Usage:
  python prep_brca2.py --output_path /tmp/brca2_vep.parquet
  python prep_brca2.py --push_to_hub --hub_repo_id HuggingFaceBio/brca2-vep
"""

import argparse
import gzip
import os
import shutil
import urllib.request

import pandas as pd
from tqdm import tqdm

WINDOW_SIZE = 8192
HUANG_XLSX_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-08388-8"
    "/MediaObjects/41586_2024_8388_MOESM3_ESM.xlsx"
)
CHR13_HG19_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr13.fa.gz"

LOF_CLASSES = {"P strong", "P moderate", "P supporting"}
FUNC_CLASSES = {"B strong", "B moderate", "B supporting"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="./brca2_cache")
    p.add_argument("--output_path", default="./brca2_cache/brca2_vep.parquet")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo_id", default="HuggingFaceBio/brca2-vep")
    return p.parse_args()


def download(url: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"  cached: {dst}")
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
    var_seq = ref_seq[:snv_pos] + alt + ref_seq[snv_pos + 1:]
    if ref_seq[snv_pos] != ref:
        return None, None
    return ref_seq, var_seq


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    xlsx_path = os.path.join(args.cache_dir, "huang2025_brca2.xlsx")
    chr13_path = os.path.join(args.cache_dir, "chr13_hg19.fa.gz")
    download(HUANG_XLSX_URL, xlsx_path)
    download(CHR13_HG19_URL, chr13_path)

    print("Reading Huang 2025 Table S3...")
    df = pd.read_excel(xlsx_path, sheet_name="Table S3", header=2)
    df.columns = [
        "exon", "chrom", "pos_hg19", "id_hg19", "pos_hg38", "id_hg38",
        "region", "ref", "alt", "hgvs",
        "ca", "cb", "cc", "cd", "ce", "cf",
        "func_score_a", "func_score_b", "func_score_c",
        "cg", "ch", "class",
    ]
    print(f"  loaded {len(df)} SNVs · class distribution: {df['class'].value_counts(dropna=False).to_dict()}")

    keep = df["class"].isin(LOF_CLASSES | FUNC_CLASSES)
    before = len(df)
    df = df[keep].copy()
    print(f"After filtering to P/B classes: {len(df)} (dropped {before - len(df)} VUS / NaN)")

    df["class"] = df["class"].apply(lambda c: "LOF" if c in LOF_CLASSES else "FUNC/INT")
    df = df.rename(columns={"pos_hg19": "pos"})
    df["score"] = df[["func_score_a", "func_score_b", "func_score_c"]].mean(axis=1)
    print(f"  binary class distribution: {df['class'].value_counts().to_dict()}")

    print("Loading chr13 hg19 reference...")
    seq_chr13 = load_fa_gz(chr13_path)
    print(f"  chr13 length: {len(seq_chr13):,} bp")

    print(f"Building {WINDOW_SIZE} bp windows around each variant...")
    refs, vars_, kept = [], [], []
    n_skip = 0
    for i in tqdm(range(len(df))):
        r, v = window(seq_chr13, int(df["pos"].iloc[i]), df["ref"].iloc[i], df["alt"].iloc[i])
        if r is None:
            n_skip += 1
            continue
        refs.append(r)
        vars_.append(v)
        kept.append(i)
    df = df.iloc[kept].reset_index(drop=True)
    df["ref_seq"] = refs
    df["var_seq"] = vars_
    print(f"  kept {len(df)} variants ({n_skip} dropped on reference mismatch)")

    out = df[["chrom", "pos", "ref", "alt", "score", "class", "ref_seq", "var_seq"]]
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out.to_parquet(args.output_path)
    print(f"Saved {len(out)} rows -> {args.output_path}  ({os.path.getsize(args.output_path)/1e6:.1f} MB)")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hub_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=args.output_path,
            path_in_repo="brca2_vep.parquet",
            repo_id=args.hub_repo_id,
            repo_type="dataset",
        )
        print(f"  pushed to {args.hub_repo_id}")


if __name__ == "__main__":
    main()
