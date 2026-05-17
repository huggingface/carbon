"""
Genome-NIAH long-context retrieval evaluation.

Long-context needle-in-a-haystack for DNA. A 24 bp (key, value) pair is
inserted at a controlled depth in a real-genome haystack; the model must
greedy-decode the value when prompted with `haystack + key`.

Four tasks of increasing difficulty (`--task`):
  niah          no distractors
  neardup_d4    8 distractors at 83% key identity (Δ=4 bp)
  neardup_d2    8 distractors at 92% key identity (Δ=2 bp)
  neardup_d1    8 distractors at 96% key identity (Δ=1 bp) — discriminator

Six context lengths in Carbon-tokens (6 bp/token), set via `--ctx`:
  4096, 8192, 16384, 32768, 65536, 131072

Backends (`--backend`):
  hf     transformers AutoModelForCausalLM. Carbon, GENERator-v2, any HF
         causal LM. Auto-shards across visible GPUs.
  evo2   official `evo2` library. Required for Arc Institute checkpoints.
         Auto-distributes layers across visible GPUs via vortex.

Tag flag (HF backend only):
  --add_dna_tag    prepend `<dna>` (Carbon hybrid models)
  --add_bos        prepend `<s>` (GENERator pure-DNA)
  (default)        no prefix — Evo2
    
Sharding (`--shard_idx --n_shards`) splits a (task, ctx) cell across multiple
SLURM jobs. Especially useful for Evo2 at ctx ≥ 64k where a row takes
~4–14 hours.

Metrics reported:
  gen_exact_match   greedy-decoded value == label (headline metric)
  gen_base_accuracy per-base accuracy of decoded value
  ll_correct        LL(positive_sequence) > LL(negative_sequence)   (HF backend only)
  ll_margin         LL(positive) − LL(negative)

Examples:
  # Carbon-3B, 32k native context
  python genome_niah_eval.py \\
      --model HuggingFaceBio/Carbon-3B \\
      --task niah --ctx 32768 --add_dna_tag --bf16

  # GENERator-v2 3B at 16k
  python genome_niah_eval.py \\
      --model GenerTeam/GENERator-v2-eukaryote-3b-base \\
      --task niah --ctx 16384 --bf16

  # Evo2-7B at 32k, shard 0 of 6 (run each shard as its own 8-GPU SLURM job)
  python genome_niah_eval.py \\
      --model evo2_7b --backend evo2 \\
      --task niah --ctx 32768 \\
      --shard_idx 0 --n_shards 6 --prefill_chars 4096
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    p.add_argument("--model", required=True,
                   help="HF repo / local path (hf backend); evo2 model name like 'evo2_7b' (evo2 backend)")
    p.add_argument("--revision", default=None, help="HF revision (branch / tag / commit)")
    p.add_argument("--backend", choices=["hf", "evo2"], default="hf")
    # Dataset
    p.add_argument("--data_path", default="HuggingFaceBio/genome-niah",
                   help="HF dataset id (default: HuggingFaceBio/genome-niah) or local parquet path")
    p.add_argument("--task", required=True,
                   choices=["niah", "neardup_d4", "neardup_d2", "neardup_d1"])
    p.add_argument("--ctx", type=int, required=True,
                   choices=[4096, 8192, 16384, 32768, 65536, 131072],
                   help="context length in Carbon-tokens (6 bp/token)")
    # HF backend
    p.add_argument("--bf16", action="store_true", help="HF: load model in bf16")
    p.add_argument("--with_yarn", action="store_true",
                   help="HF: load with YaRN rope scaling for 64k-token context")
    p.add_argument("--add_dna_tag", action="store_true",
                   help="HF: prepend `<dna>` (Carbon hybrid models)")
    p.add_argument("--add_bos", action="store_true", help="HF: prepend <s> (GENERator pure-DNA)")
    p.add_argument("--restrict_to_dna_tokens", action="store_true", default=True,
                   help="HF: mask non-DNA tokens during generation (Carbon hybrid). "
                   "Reads DNA token range from the model's dna_config.json. "
                   "Auto-detected and applied for Carbon hybrid; safely no-ops for others.")
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--batch_size", type=int, default=1)
    # Evo2 backend
    p.add_argument("--prefill_chars", type=int, default=4096,
                   help="Evo2: chunked-prefill size; rest decoded 1-by-1")
    # Sharding / sampling
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap rows after deterministic shuffle. Default: all (500/cell).")
    p.add_argument("--sample_offset", type=int, default=0,
                   help="Skip first N rows of deterministic shuffle (incremental runs)")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--output_dir", default="./results/genome_niah")
    p.add_argument("--model_short_name", default=None,
                   help="Short name for result filenames (default: derived from --model)")
    return p.parse_args()


# ============================================================
# Dataset loading
# ============================================================

CTX_LABEL = {4096: "4k", 8192: "8k", 16384: "16k",
             32768: "32k", 65536: "64k", 131072: "128k"}


def _hf_config_for(task: str, ctx: int) -> str:
    """Map (task, ctx) to the Hub dataset config name.

    Hub naming convention: `niah_<ctx>` for the bare task, and
    `niah_<distractor_variant>_<ctx>` for near-duplicate variants.
    """
    if task == "niah":
        return f"niah_{CTX_LABEL[ctx]}"
    # neardup_d1, neardup_d2, neardup_d4 → niah_neardup_dX_<ctx>
    return f"niah_{task}_{CTX_LABEL[ctx]}"


def load_dataset(data_path: str, task: str, ctx: int) -> pd.DataFrame:
    """Load + filter to one (task, ctx) cell. Accepts HF dataset id or local parquet path."""
    if Path(data_path).exists():
        df = pd.read_parquet(data_path)
    else:
        from datasets import load_dataset as hf_load
        config = _hf_config_for(task, ctx)
        try:
            ds = hf_load(data_path, config, split="test")
        except Exception:
            ds = hf_load(data_path, config)
            ds = ds[list(ds.keys())[0]]
        df = ds.to_pandas()
    df = df[df["context_length_tokens"] == ctx].copy().reset_index(drop=True)
    return df


def apply_sharding(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.max_samples is not None:
        df = df.sample(min(args.max_samples, len(df)),
                       random_state=args.seed).reset_index(drop=True)
    if args.sample_offset > 0:
        if args.sample_offset >= len(df):
            raise ValueError(f"sample_offset={args.sample_offset} >= pool {len(df)}")
        df = df.iloc[args.sample_offset:].reset_index(drop=True)
    if args.n_shards > 1:
        df = df.iloc[args.shard_idx::args.n_shards].reset_index(drop=True)
    return df


# ============================================================
# HF backend (Carbon, GENERator, any HF causal LM)
# ============================================================

def _prefix(args) -> str:
    if args.add_dna_tag:
        return "<dna>"
    if args.add_bos:
        return "<s>"
    return ""


def _load_hf(args):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers_compat import patch_generator_sample, patch_legacy_tokenizer_base

    patch_legacy_tokenizer_base()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "revision": args.revision,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float32,
        "attn_implementation": args.attn_implementation,
        "device_map": "auto",
    }
    if args.with_yarn:
        config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
        config.max_position_embeddings = 65536
        rope_parameters = dict(getattr(config, "rope_parameters", {}) or {})
        rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", None))
        yarn_rope_parameters = {
            "rope_type": "yarn",
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
        if rope_theta is not None:
            yarn_rope_parameters["rope_theta"] = rope_theta
        config.rope_scaling = yarn_rope_parameters
        model_kwargs["config"] = config
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )
    model.eval()
    patch_generator_sample(model)
    return model, tokenizer


def _resolve_dna_token_range(model_path: str, revision=None):
    """Return (dna_start, dna_count) for the 6-mer block from dna_config.json.
    Returns None if the model is not a Carbon hybrid (no dna_config.json).
    """
    cfg_path = Path(model_path) / "dna_config.json"
    if not cfg_path.exists():
        try:
            from huggingface_hub import hf_hub_download
            cfg_path = Path(hf_hub_download(repo_id=model_path, filename="dna_config.json", revision=revision))
        except Exception:
            return None  # not a Carbon hybrid
    cfg = json.loads(cfg_path.read_text())
    n_specials = len(cfg.get("dna_special_tokens", []))
    return cfg["dna_start_id"] + n_specials, 4096  # 4096 6-mer tokens


def hf_generate(model, tokenizer, df, args) -> List[dict]:
    """Greedy-decode the value for each row, suppressing special tokens.
    For Carbon hybrid (dna_config.json present), also restrict generation to
    the 6-mer DNA token block."""
    from transformers import LogitsProcessor, LogitsProcessorList

    class SuppressSpecial(LogitsProcessor):
        def __init__(self, ids):
            self.ids = ids
        def __call__(self, input_ids, scores):
            for i in self.ids: scores[:, i] = float("-inf")
            return scores

    class RestrictToDNA(LogitsProcessor):
        def __init__(self, dna_start, dna_count):
            self.dna_start = dna_start
            self.dna_end = dna_start + dna_count
        def __call__(self, input_ids, scores):
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.dna_start:self.dna_end] = 0.0
            return scores + mask

    device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    sp_ids = list(getattr(tokenizer, "all_special_ids", []) or [])
    procs = [SuppressSpecial(sp_ids)]
    if args.restrict_to_dna_tokens:
        dna_range = _resolve_dna_token_range(args.model, args.revision)
        if dna_range is not None:
            procs.append(RestrictToDNA(*dna_range))
            print(f"  DNA token restriction: ids [{dna_range[0]}, {dna_range[0]+dna_range[1]})")
        else:
            print(f"  (no dna_config.json — DNA token restriction skipped; assumed non-hybrid)")
    proc = LogitsProcessorList(procs)

    value_lens = df["value_len_bp"].astype(int).tolist()
    n_gen_tokens_set = sorted(set(v // 6 for v in value_lens))
    if len(n_gen_tokens_set) != 1:
        raise ValueError("All rows in a (task, ctx) cell must share value_len_bp")
    max_new = n_gen_tokens_set[0]

    out = []
    prefix = _prefix(args)
    for start in tqdm(range(0, len(df), args.batch_size), desc="hf-gen"):
        end = min(start + args.batch_size, len(df))
        prompts = [prefix + s for s in df["prompt"].iloc[start:end]]
        enc = tokenizer(prompts, add_special_tokens=False, return_tensors="pt",
                        padding=True, truncation=False).to(device)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id, logits_processor=proc)
        new_ids = gen[:, -max_new:]
        for offset in range(end - start):
            pred = tokenizer.decode(new_ids[offset].tolist())  # NO skip_special_tokens — hybrid drops 6-mers
            idx = start + offset
            vlen = value_lens[idx]
            pred = (pred or "")[:vlen]
            label = df["value"].iloc[idx]
            same = sum(1 for j in range(vlen) if j < len(pred) and pred[j] == label[j])
            out.append({"uid": df["uid"].iloc[idx], "pred": pred,
                        "gen_exact_match": int(pred == label),
                        "gen_base_accuracy": same / vlen})
    return out


def hf_likelihood(model, tokenizer, df, args) -> List[dict]:
    """LL(positive) vs LL(negative) over the value tokens. Slices logits to save memory."""
    device = next(model.parameters()).device
    tokenizer.padding_side = "right"
    body = getattr(model, "model", None) or getattr(model, "transformer", None)
    if body is None:
        raise RuntimeError("Model body not found at .model or .transformer")

    def ll(prompts, full_seqs):
        prefix = _prefix(args)
        full_enc = tokenizer([prefix + s for s in full_seqs],
                             add_special_tokens=False, return_tensors="pt", padding=True).to(device)
        prompt_enc = tokenizer([prefix + s for s in prompts],
                               add_special_tokens=False, return_tensors="pt", padding=True).to(device)
        full_ids = full_enc["input_ids"]
        full_len = int(full_enc["attention_mask"].sum(1)[0].item())
        prompt_len = int(prompt_enc["attention_mask"].sum(1)[0].item())
        with torch.inference_mode():
            h = body(input_ids=full_ids, attention_mask=full_enc["attention_mask"], use_cache=False).last_hidden_state
            slice_h = h[:, prompt_len-1:full_len-1, :]                      # (B, V_tok, H)
            slice_logits = model.lm_head(slice_h)                            # (B, V_tok, V)
            log_probs = torch.log_softmax(slice_logits.float(), dim=-1)
            # device_map="auto" can leave log_probs on a different GPU than full_ids; align
            targets = full_ids[:, prompt_len:full_len].to(log_probs.device)
            token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_lp.mean(dim=1).cpu().tolist()

    out = []
    for start in tqdm(range(0, len(df), args.batch_size), desc="hf-ll"):
        end = min(start + args.batch_size, len(df))
        prompts = df["prompt"].iloc[start:end].tolist()
        pos = df["positive_sequence"].iloc[start:end].tolist()
        neg = df["negative_sequence"].iloc[start:end].tolist()
        pos_ll = ll(prompts, pos)
        neg_ll = ll(prompts, neg)
        for o, (p, n) in enumerate(zip(pos_ll, neg_ll)):
            out.append({"uid": df["uid"].iloc[start + o],
                        "target_ll": p, "negative_ll": n,
                        "ll_margin": p - n, "ll_correct": int(p > n)})
    return out


def hf_eval(args, df) -> pd.DataFrame:
    model, tokenizer = _load_hf(args)
    gen_rows = hf_generate(model, tokenizer, df, args)
    ll_rows = hf_likelihood(model, tokenizer, df, args)
    gen = pd.DataFrame(gen_rows).set_index("uid")
    ll = pd.DataFrame(ll_rows).set_index("uid")
    return gen.join(ll, how="outer").reset_index()


# ============================================================
# Evo2 backend — chunked prefill + single-token decode (vortex multi-GPU)
# ============================================================

ACGT_IDS = (ord("A"), ord("C"), ord("G"), ord("T"))


def _move_ip(ip, device):
    for k, v in list(ip["mha"].key_value_memory_dict.items()):
        ip["mha"].key_value_memory_dict[k] = v.to(device)
    for ck in ("hcl", "hcm", "hcs"):
        for d in (getattr(ip[ck], "fir_state_dict", {}),
                  getattr(ip[ck], "fir_inner_state_dict", {}),
                  getattr(ip[ck], "state_dict", {})):
            for k, v in list(d.items()):
                if hasattr(v, "to"): d[k] = v.to(device)


def _set_offset(ip, pos):
    for k in ("mha", "hcl", "hcm", "hcs"): ip[k].seqlen_offset = pos


def _unwrap(out):
    while isinstance(out, tuple): out = out[0]
    return out


def evo2_eval(args, df) -> pd.DataFrame:
    """Generation only — Evo2 LL path is too expensive at long ctx (no separate body call)."""
    from evo2_runtime import preload_cudnn_libraries

    preload_cudnn_libraries()
    from evo2 import Evo2
    model_name = args.model.split("/")[-1]
    print(f"Loading Evo2 model: {model_name}")
    t0 = time.time()
    model = Evo2(model_name)
    print(f"loaded in {time.time()-t0:.1f}s")
    tokenizer = model.tokenizer
    device = "cuda:0"
    acgt_t = torch.tensor(ACGT_IDS, dtype=torch.long, device=device)

    def generate_value(prompt: str, n_gen_chars: int) -> str:
        input_ids = torch.tensor(tokenizer.tokenize(prompt), dtype=torch.long, device=device).unsqueeze(0)
        L = int(input_ids.shape[1])
        ip = model.model.initialize_inference_params(max_seqlen=L + n_gen_chars)
        ip["mha"].max_batch_size = 1
        _move_ip(ip, device)
        prefill_len = min(args.prefill_chars, L)
        with torch.inference_mode():
            _set_offset(ip, 0)
            out = model.model(input_ids[:, :prefill_len], inference_params_dict=ip)
            last_logits = _unwrap(out)[:, -1:, :].clone()
            for pos in range(prefill_len, L):
                _set_offset(ip, pos)
                out = model.model(input_ids[:, pos:pos+1], inference_params_dict=ip)
                last_logits = _unwrap(out)
            # Generate ACGT chars
            generated = []
            for step in range(n_gen_chars):
                logits_acgt = last_logits[0, -1, :].float().index_select(0, acgt_t)
                choice = ACGT_IDS[int(logits_acgt.argmax().item())]
                generated.append(choice)
                _set_offset(ip, L + step)
                out = model.model(torch.tensor([[choice]], device=device), inference_params_dict=ip)
                last_logits = _unwrap(out)
        return "".join(chr(t) for t in generated)

    rows = []
    for i in tqdm(range(len(df)), desc="evo2-eval"):
        row = df.iloc[i]
        vlen = int(row["value_len_bp"])
        try:
            pred = generate_value(row["prompt"], vlen)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  err row {i}: {str(e)[:200]}")
            pred = ""
        label = row["value"]
        same = sum(1 for j in range(vlen) if j < len(pred) and pred[j] == label[j])
        rows.append({"uid": row["uid"], "pred": pred,
                     "gen_exact_match": int(pred == label),
                     "gen_base_accuracy": same / vlen})
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    print(f"Genome-NIAH eval · model={args.model} · backend={args.backend} · "
          f"task={args.task} · ctx={args.ctx}")

    df = load_dataset(args.data_path, args.task, args.ctx)
    df = apply_sharding(df, args)
    if df.empty:
        raise ValueError(f"No rows for task={args.task} ctx={args.ctx} after sharding")
    print(f"  → {len(df)} rows (shard {args.shard_idx}/{args.n_shards}, "
          f"offset {args.sample_offset}, max {args.max_samples or 'all'})")

    t0 = time.time()
    if args.backend == "hf":
        res = hf_eval(args, df)
    elif args.backend == "evo2":
        res = evo2_eval(args, df)
    else:
        raise ValueError(args.backend)
    elapsed = time.time() - t0

    # Aggregate
    summary = {
        "model": args.model, "revision": args.revision, "backend": args.backend,
        "task": args.task, "ctx": args.ctx, "n_samples": int(len(res)),
        "shard_idx": args.shard_idx, "n_shards": args.n_shards,
        "with_yarn": bool(args.with_yarn),
        "gen_exact_match": float(res["gen_exact_match"].mean()),
        "gen_base_accuracy": float(res["gen_base_accuracy"].mean()),
        "elapsed_sec": elapsed,
    }
    if "ll_correct" in res.columns:
        summary["ll_correct"] = float(res["ll_correct"].mean())
        ll_margin = res["ll_margin"].dropna()
        summary["ll_margin"] = float(ll_margin.mean()) if len(ll_margin) else float("nan")

    # Save
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    short = args.model_short_name or args.model.split("/")[-1].replace("_", "-")
    shard_tag = f"_shard{args.shard_idx}of{args.n_shards}" if args.n_shards > 1 else ""
    base = f"{short}_{args.task}_ctx{args.ctx}{shard_tag}"
    res.to_parquet(out_dir / f"{base}.parquet", index=False)
    (out_dir / f"{base}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Summary ({args.task} @ ctx {args.ctx}, n={len(res)}) ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {out_dir / base}.{{json,parquet}}")


if __name__ == "__main__":
    main()
