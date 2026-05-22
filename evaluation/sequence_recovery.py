"""
Sequence recovery: training-free generative DNA eval.

Given a fixed-length DNA context, generate the next N tokens and score the
exact-base recovery accuracy against the held-out continuation.

Three model families are supported via a single flag:

  --backend hf   (default)  Carbon, GENERator, or any HF causal LM
  --backend evo2            official `evo2` inference library

Tag flags (mutually exclusive):
  --add_dna_tag   prepend `<dna>` to each sequence (Carbon hybrid models)
  --add_bos       prepend `<s>` (GENERator pure-DNA)
  (default)       no prefix — Evo2

Eukaryote / bacteria / others come from the GenerTeam/sequence-recovery dataset.

Example:

  # Carbon 3B hybrid (flagship)
  python sequence_recovery.py \
      --model HuggingFaceBio/Carbon-3B \
      --data_type eukaryote --add_dna_tag --bf16

  # GENERator
  python sequence_recovery.py \
      --model GenerTeam/GENERator-v2-eukaryote-3b-base \
      --data_type eukaryote --add_bos --bf16

  # Evo2 7B (1 GPU)
  python sequence_recovery.py \
      --model evo2_7b --backend evo2 \
      --data_type eukaryote --gen_len_bp 30
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequence recovery eval")
    p.add_argument("--model", required=True, help="HF repo / local path / evo2 model name")
    p.add_argument("--revision", default=None)
    p.add_argument("--backend", choices=["hf", "evo2"], default="hf")
    p.add_argument("--data_path", default="hf://datasets/GenerTeam/sequence-recovery")
    p.add_argument("--data_type", choices=["eukaryote", "bacteria", "others"], default="eukaryote")
    p.add_argument("--output_dir", default="./results/sequence_recovery")
    p.add_argument("--max_seq_len", type=int, default=6144, help="Context length in bp")
    p.add_argument("--gen_len", type=int, default=5, help="HF: number of tokens to generate")
    p.add_argument("--gen_len_bp", type=int, default=30, help="Evo2: number of bases to generate")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--add_dna_tag", action="store_true", help="Prepend <dna> (Carbon hybrid)")
    p.add_argument("--add_bos", action="store_true", help="Prepend <s> (GENERator pure-DNA)")
    p.add_argument("--max_samples", type=int, default=None, help="For quick testing")
    p.add_argument("--shard_idx", type=int, default=0, help="0-based row shard index")
    p.add_argument("--n_shards", type=int, default=1, help="Total number of row shards")
    return p.parse_args()


# FIX 1: inherit from LogitsProcessor so transformers validates it correctly
class SuppressSpecialTokens(LogitsProcessor):
    def __init__(self, ids):
        self.ids = ids

    def __call__(self, input_ids, scores):
        for tid in self.ids:
            scores[:, tid] = -float("inf")
        return scores


def _prefix(args) -> str:
    if args.add_dna_tag:
        return "<dna>"
    if args.add_bos:
        return "<s>"
    return ""


def _truncate(seq: str, max_seq_len: int) -> str:
    """Keep rightmost bases; round down to a multiple of 6 for 6-mer tokenizers."""
    n = (min(len(seq), max_seq_len) // 6) * 6
    return seq[-n:]


def calculate_accuracy(preds, labels, seq_len: int = 30):
    # FIX 3: seq_len is now always passed explicitly from main(); the default
    # of 30 is kept only as a fallback so call-sites that don't yet pass it
    # still work, but main() derives the value from args instead of relying
    # on the hard-coded default.
    out = []
    for label, pred in zip(labels, preds):
        match = sum(1 for i in range(min(len(label), len(pred), seq_len)) if label[i] == pred[i])
        out.append(match / seq_len)
    return out


def hf_shard(shard_id, records, args, dtype_str):
    """Generate continuations on one GPU for a list of sequences."""
    torch.cuda.set_device(shard_id)
    device = f"cuda:{shard_id}"
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # FIX 2: use torch_dtype (documented kwarg); plain dtype= is silently
    # ignored by from_pretrained and leaves the model in float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()

    special_ids = getattr(tok, "all_special_ids", []) or []
    logits_processor = LogitsProcessorList([SuppressSpecialTokens(special_ids)])

    prefix = _prefix(args)
    preds = []

    with tqdm(total=len(records), desc=f"gpu{shard_id}", unit="seq") as pbar:
        for i in range(0, len(records), args.batch_size):
            batch = records[i : i + args.batch_size]
            seqs = [prefix + _truncate(r["sequence"], args.max_seq_len) for r in batch]
            enc = tok(seqs, add_special_tokens=False, return_tensors="pt", padding=True, truncation=False)
            enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}

            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.gen_len,
                    pad_token_id=tok.pad_token_id,
                    do_sample=False,
                    logits_processor=logits_processor,
                )

            # decode last gen_len tokens; tag-mode needs per-row decode to keep DNA bases
            new_ids = out[:, -args.gen_len:]
            if args.add_dna_tag:
                decoded = [tok.decode(new_ids[j].tolist()) for j in range(new_ids.shape[0])]
            else:
                decoded = tok.batch_decode(new_ids, skip_special_tokens=True)

            for r, txt in zip(batch, decoded):
                preds.append({"hash": r["hash"], "pred": txt})
            pbar.update(len(batch))

    return preds


def run_hf(df: pd.DataFrame, args, dtype_str: str):
    records = df[["sequence", "hash"]].to_dict("records")
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPU available")

    shard_size = (len(records) + n_gpus - 1) // n_gpus
    all_preds = []

    # FIX 1: use spawn context — fork + CUDA is not safe on Linux
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx) as ex:
        futs = {
            ex.submit(hf_shard, g, records[g * shard_size : (g + 1) * shard_size], args, dtype_str): g
            for g in range(n_gpus)
            if g * shard_size < len(records)
        }
        for f in as_completed(futs):
            all_preds.extend(f.result())

    return all_preds


def run_evo2(df: pd.DataFrame, args):
    from evo2_runtime import preload_cudnn_libraries

    preload_cudnn_libraries()
    from evo2 import Evo2

    model_name = args.model.split("/")[-1]
    model = Evo2(model_name)

    preds = []
    seqs = df["sequence"].tolist()
    hashes = df["hash"].tolist()

    with tqdm(total=len(seqs), desc="evo2", unit="seq") as pbar:
        for i in range(0, len(seqs), args.batch_size):
            batch = seqs[i : i + args.batch_size]
            prompts = [s[-min(len(s), args.max_seq_len) :] for s in batch]
            try:
                out = model.generate(prompt_seqs=prompts, n_tokens=args.gen_len_bp, temperature=0.0, top_k=1)
                for h, gen in zip(hashes[i : i + args.batch_size], out.sequences):
                    preds.append({"hash": h, "pred": gen})
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                for s, h in zip(batch, hashes[i : i + args.batch_size]):
                    out = model.generate(
                        prompt_seqs=[s[-args.max_seq_len :]],
                        n_tokens=args.gen_len_bp,
                        temperature=0.0,
                        top_k=1,
                    )
                    preds.append({"hash": h, "pred": out.sequences[0]})
            pbar.update(len(batch))

    return preds


def main():
    args = parse_args()
    assert not (args.add_dna_tag and args.add_bos), "--add_dna_tag and --add_bos are mutually exclusive"

    if args.n_shards < 1:
        raise ValueError("--n_shards must be >= 1")
    if not 0 <= args.shard_idx < args.n_shards:
        raise ValueError("--shard_idx must satisfy 0 <= shard_idx < n_shards")

    dtype_str = "bfloat16" if args.bf16 else "float32"

    print("=" * 70)
    print(f"Sequence recovery · model={args.model} · backend={args.backend} · data={args.data_type}")
    print("=" * 70)

    df = pd.read_parquet(f"{args.data_path}/{args.data_type}/test.parquet")

    if args.max_samples:
        df = df.sample(min(args.max_samples, len(df)), random_state=0).reset_index(drop=True)

    df["hash"] = df.apply(
        lambda row: hashlib.md5(f"{row['sequence']}_{row.name}".encode()).hexdigest()[:16],
        axis=1,
    )

    if args.n_shards > 1:
        total_before_shard = len(df)
        df = df.iloc[args.shard_idx :: args.n_shards].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(
                f"Shard {args.shard_idx}/{args.n_shards} has no rows from {total_before_shard} sequences"
            )
        print(
            f"Shard {args.shard_idx}/{args.n_shards}: selected {len(df)} "
            f"of {total_before_shard} sequences"
        )

    print(f"Loaded {len(df)} sequences ({df['type'].value_counts().to_dict() if 'type' in df else 'n/a'})")

    t0 = time.time()
    if args.backend == "evo2":
        preds = run_evo2(df, args)
    else:
        preds = run_hf(df, args, dtype_str)
    print(f"Generation took {time.time() - t0:.1f}s")

    pred_df = pd.DataFrame(preds)
    out = df.merge(pred_df, on="hash", how="left")
    out["pred"] = out["pred"].fillna("")

    # FIX 3: derive seq_len from args rather than relying on the hardcoded
    # default of 30.  For the HF backend one token = one 6-mer = 6 bp, so
    # gen_len * 6 gives the expected number of bases.  For Evo2 the user
    # supplies gen_len_bp directly.
    if args.backend == "evo2":
        seq_len = args.gen_len_bp
    else:
        seq_len = args.gen_len * 6

    out["accuracy"] = calculate_accuracy(out["pred"].tolist(), out["label"].tolist(), seq_len=seq_len)

    overall = out["accuracy"].mean()
    by_type = out.groupby("type")["accuracy"].mean() if "type" in out.columns else None

    print(f"\nOverall accuracy: {overall:.4f}")
    if by_type is not None:
        print("Per-type:")
        for t, v in by_type.items():
            print(f"  {t}: {v:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = args.model.split("/")[-1]
    shard_tag = f"_shard{args.shard_idx}of{args.n_shards}" if args.n_shards > 1 else ""
    base = f"{model_tag}_{args.data_type}_{dtype_str}{shard_tag}"
    parquet = os.path.join(args.output_dir, f"{base}.parquet")
    summary = os.path.join(args.output_dir, f"{base}.json")

    cols = [c for c in ["hash", "type", "label", "pred", "accuracy"] if c in out.columns]
    out[cols].to_parquet(parquet)

    with open(summary, "w") as f:
        json.dump(
            {
                "model": args.model,
                "revision": args.revision,
                "backend": args.backend,
                "data_type": args.data_type,
                "num_sequences": int(len(out)),
                "overall_accuracy": float(overall),
                "type_accuracy": {k: float(v) for k, v in (by_type.items() if by_type is not None else [])},
                "dtype": dtype_str,
                "shard_idx": int(args.shard_idx),
                "n_shards": int(args.n_shards),
            },
            f,
            indent=2,
        )

    print(f"\nSaved {parquet}\n      {summary}")


if __name__ == "__main__":
    main()
