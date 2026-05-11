"""
Build the BRCA1 zero-shot VEP dataset (Findlay et al. 2018).

Pipeline:
  1. Download the Nature 2018 supplementary Excel — 3,893 BRCA1 SNVs with
     saturation-genome-editing functional scores and a {LOF, INT, FUNC} label.
  2. Download chr17 hg19 from UCSC goldenPath.
  3. For each SNV, slice an 8,192 bp window centered on the variant and build
     a paired (ref_seq, var_seq).

Output schema (parquet):
  chrom, pos, ref, alt, score, class, ref_seq, var_seq

Class is collapsed to {LOF, FUNC/INT} to match Evo2 §4.3.15.

Usage:
  python prep_brca1.py --output_path /tmp/brca1_vep.parquet
  python prep_brca1.py --push_to_hub --hub_repo_id hf-carbon/brca1-vep
"""

import argparse
import gzip
import os
import shutil
import urllib.request

import pandas as pd
from tqdm import tqdm

WINDOW_SIZE = 8192
FINDLAY_XLSX_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0461-z"
    "/MediaObjects/41586_2018_461_MOESM3_ESM.xlsx"
)
CHR17_HG19_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr17.fa.gz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="./brca1_cache")
    p.add_argument("--output_path", default="./brca1_cache/brca1_vep.parquet")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo_id", default="hf-carbon/brca1-vep")
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
    assert ref_seq[snv_pos] == ref, f"ref mismatch at pos {pos}: {ref_seq[snv_pos]} vs {ref}"
    assert len(var_seq) == len(ref_seq)
    return ref_seq, var_seq


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    xlsx_path = os.path.join(args.cache_dir, "findlay2018_brca1.xlsx")
    chr17_path = os.path.join(args.cache_dir, "chr17_hg19.fa.gz")
    download(FINDLAY_XLSX_URL, xlsx_path)
    download(CHR17_HG19_URL, chr17_path)

    print("Reading Findlay 2018 Table S1...")
    df = pd.read_excel(xlsx_path, header=2)
    df = df[["chromosome", "position (hg19)", "reference", "alt",
             "function.score.mean", "func.class"]]
    df = df.rename(columns={
        "chromosome": "chrom",
        "position (hg19)": "pos",
        "reference": "ref",
        "function.score.mean": "score",
        "func.class": "class",
    })
    # Collapse FUNC + INT into one functional class (Evo2 §4.3.15)
    df["class"] = df["class"].replace(["FUNC", "INT"], "FUNC/INT")
    print(f"  loaded {len(df)} SNVs · class distribution: {df['class'].value_counts().to_dict()}")

    print("Loading chr17 hg19 reference...")
    seq_chr17 = load_fa_gz(chr17_path)
    print(f"  chr17 length: {len(seq_chr17):,} bp")

    print(f"Building {WINDOW_SIZE} bp windows around each variant...")
    refs, vars_ = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        r, v = window(seq_chr17, int(row["pos"]), row["ref"], row["alt"])
        refs.append(r)
        vars_.append(v)
    df["ref_seq"] = refs
    df["var_seq"] = vars_

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_parquet(args.output_path)
    print(f"Saved {len(df)} rows -> {args.output_path}  ({os.path.getsize(args.output_path)/1e6:.1f} MB)")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hub_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=args.output_path,
            path_in_repo="brca1_vep.parquet",
            repo_id=args.hub_repo_id,
            repo_type="dataset",
        )
        print(f"  pushed to {args.hub_repo_id}")


if __name__ == "__main__":
    main()
