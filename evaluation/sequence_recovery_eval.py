"""
Usage:
  Legacy fixed-window scoring (preserves the original --gen_len 5 metric):
    uv run --project evaluation python evaluation/sequence_recovery_eval.py \
      --model hf-carbon/carbon-3B-600B-dna-generv2-fp32-lmhead \
      --model_name 3B-WITH-TAGS \
      --data_type eukaryote \
      --data_path hf://datasets/GenerTeam/sequence-recovery \
      --output_dir scratch/sequence_recovery/fixed_bp \
      --max_seq_len 6144 \
      --gen_len 5 \
      --gen_len_bp 30 \
      --batch_size 64 \
      --bf16 \
      --use_dna_tags

  Long-rollout scoring (scores the generated window and derives a tail label when needed):
    uv run --project evaluation python evaluation/sequence_recovery_eval.py \
      --model hf-carbon/carbon-3B-600B-dna-generv2-fp32-lmhead \
      --model_name 3B-WITH-TAGS \
      --data_type eukaryote \
      --data_path hf://datasets/GenerTeam/sequence-recovery \
      --output_dir scratch/sequence_recovery/prediction_length \
      --max_seq_len 6144 \
      --gen_len 512 \
      --batch_size 64 \
      --bf16 \
      --use_dna_tags \
      --accuracy_mode prediction_length
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
)
from transformers.generation import ContinuousBatchingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequence recovery eval (post-training)"
    )
    parser.add_argument(
        "--data_type",
        default="eukaryote",
        choices=["eukaryote", "bacteria", "others"],
        help="Data type split to evaluate",
    )
    parser.add_argument(
        "--data_path",
        default="hf://datasets/GenerTeam/sequence-recovery",
        help="HF dataset path (parquet via hf://datasets/...)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or path (HF hub repo or local)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision/tag/commit",
    )
    parser.add_argument(
        "--output_dir",
        default="./eval_results/sequence_recovery",
        help="Directory to save eval outputs",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=6144,
        help=(
            "Legacy max input length in bp (truncate left, keep rightmost). "
            "Use --prompt_len_bp for prompt-length sweeps."
        ),
    )
    parser.add_argument(
        "--prompt_len_bp",
        type=int,
        default=None,
        help=(
            "Prompt/input length in bp before generation. Overrides --max_seq_len "
            "when provided."
        ),
    )
    parser.add_argument(
        "--gen_len",
        type=int,
        default=5,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--gen_len_bp",
        type=int,
        default=30,
        help="Number of base pairs to generate when using Evo2",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16",
    )
    parser.add_argument(
        "--use_evo2",
        action="store_true",
        help="Use Evo2 inference (official evo2 library) instead of HF AutoModel",
    )
    parser.add_argument(
        "--evo2_force_prompt_threshold",
        type=int,
        default=None,
        help=(
            "Optional force_prompt_threshold passed to Evo2.generate. "
            "Set higher than max_seq_len to explicitly disable prompt forcing."
        ),
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for generation (mutually exclusive with --use_evo2).",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory fraction (passed to LLM(gpu_memory_utilization=...)).",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Override vLLM max_model_len. Defaults to max_seq_len/bp_per_token + gen_len + 8.",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size. Defaults to 1 (set explicitly to shard "
        "the model across GPUs). TP can fail for models that fall back to "
        "vLLM's Transformers backend with dimensions not divisible by TP.",
    )
    parser.add_argument(
        "--vllm_data_parallel_size",
        type=int,
        default=1,
        help="vLLM data_parallel_size. Defaults to 1. Set to torch.cuda.device_count() "
        "to replicate the model across all GPUs and load-balance requests "
        "(recommended for throughput on multi-GPU nodes when the model fits "
        "on one GPU).",
    )
    parser.add_argument(
        "--generation_backend",
        default="static",
        choices=["static", "continuous"],
        help="HF generation mode: 'static' uses model.generate() per batch; "
        "'continuous' uses model.generate_batch() with Transformers' "
        "ContinuousBatchingConfig. Ignored when --use_vllm or --use_evo2 is set.",
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional attention backend passed to AutoModelForCausalLM.from_pretrained "
        "(e.g. 'paged|sdpa', 'paged|flash_attention_2'). "
        "Required for continuous batching with a paged backend.",
    )
    parser.add_argument(
        "--cb_max_batch_tokens",
        type=int,
        default=None,
        help="ContinuousBatchingConfig.max_batch_tokens. None uses the library default.",
    )
    parser.add_argument(
        "--cb_max_memory_percent",
        type=float,
        default=0.8,
        help="ContinuousBatchingConfig.max_memory_percent.",
    )
    parser.add_argument(
        "--cb_scheduler_type",
        default="fifo",
        choices=["fifo", "prefill_first"],
        help="ContinuousBatchingConfig.scheduler_type.",
    )
    parser.add_argument(
        "--cb_use_cuda_graph",
        action="store_true",
        help="Enable CUDA graphs for continuous batching.",
    )
    parser.add_argument(
        "--cb_use_async_batching",
        action="store_true",
        help="Enable async batching for continuous batching.",
    )
    parser.add_argument(
        "--cb_use_default_compile_configs",
        action="store_true",
        help="Use the library's default torch.compile configs for continuous batching.",
    )
    parser.add_argument(
        "--cb_q_padding_interval_size",
        type=int,
        default=0,
        help="q_padding_interval_size for continuous batching. 0 keeps library default.",
    )
    parser.add_argument(
        "--cb_kv_padding_interval_size",
        type=int,
        default=0,
        help="kv_padding_interval_size for continuous batching. 0 keeps library default.",
    )
    parser.add_argument(
        "--cb_max_cached_graphs",
        type=int,
        default=0,
        help="max_cached_graphs for continuous batching. 0 keeps library default.",
    )
    parser.add_argument(
        "--cb_max_blocks_per_request",
        type=int,
        default=0,
        help="max_blocks_per_request for the continuous-batching fast decode path.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload outputs to the Hub",
    )
    parser.add_argument(
        "--hub_repo_id",
        default=None,
        help="HF repo to upload results (e.g., hf-carbon/seq-recovery-results)",
    )
    parser.add_argument(
        "--hub_repo_type",
        default="dataset",
        choices=["dataset", "model"],
        help="HF repo type",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of sequences to evaluate (for testing). If None, evaluates all sequences.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="Random seed used when subsampling with --max_samples.",
    )
    parser.add_argument(
        "--use_dna_tags",
        action="store_true",
        help="Wrap DNA sequences with <dna>...</dna> tags for hybrid tokenizer models",
    )
    parser.add_argument(
        "--no_prefix",
        action="store_true",
        help="Don't add any prefix token (no <s> or <dna>). Use for models without BOS token.",
    )
    parser.add_argument(
        "--use_species_tags",
        action="store_true",
        help="Prepend species metadata tag before <dna> tag (requires --use_dna_tags). "
        "Maps dataset 'type' column to tags: vertebrate_mammalian-><mammalian_species>, "
        "vertebrate_other-><vertebrate_non_mammalian_species>, fungi-><fungi_species>, "
        "plant-><plant_species>, protozoa-><protozoan_species>, invertebrate-><invertebrate_species>.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Override model name for output file naming. If not provided, uses the last component of --model path.",
    )
    parser.add_argument(
        "--upcast_lm_head",
        action="store_true",
        help="Upcast lm_head to float32 while keeping rest in bf16. "
        "Fixes SR degradation from bf16 rounding in the logit projection.",
    )
    parser.add_argument(
        "--accuracy_mode",
        default="fixed_bp",
        choices=["fixed_bp", "prediction_length"],
        help="How to score generated rollouts. "
        "'fixed_bp' preserves the legacy 30 bp denominator, "
        "while 'prediction_length' scores the full generated window up to the label length.",
    )
    parser.add_argument(
        "--score_len_bp",
        type=int,
        default=30,
        help="Base-pair denominator used by --accuracy_mode fixed_bp.",
    )
    parser.add_argument(
        "--label_source",
        default="auto",
        choices=["auto", "dataset", "sequence_tail"],
        help="Where evaluation labels come from. "
        "'dataset' uses the shipped label column, "
        "'sequence_tail' derives a held-out suffix from the input sequence, "
        "and 'auto' keeps the dataset label unless the requested scored window exceeds it.",
    )
    parser.add_argument(
        "--bp_per_token",
        type=int,
        default=6,
        help="Expected bp represented by each token when inferring sequence-tail labels. "
        "For example, a k-mer tokenizer has k base pair per token.",
    )
    args = parser.parse_args()
    if args.use_vllm and args.use_evo2:
        parser.error("--use_vllm and --use_evo2 are mutually exclusive")
    if args.use_vllm and args.upcast_lm_head:
        parser.error(
            "--upcast_lm_head is HF-only: vLLM's Transformers backend builds "
            "its own ParallelLMHead and never calls the HF-side monkey-patch."
        )
    if args.generation_backend == "continuous" and (args.use_vllm or args.use_evo2):
        parser.error(
            "--generation_backend continuous is an HF-only option; "
            "it cannot be combined with --use_vllm or --use_evo2."
        )
    return args


class SuppressSpecialTokensLogitsProcessor:
    """Suppress all special tokens during generation by setting logits to -inf."""

    def __init__(self, special_token_ids: list):
        self.special_token_ids = special_token_ids

    def __call__(self, input_ids, scores):
        for token_id in self.special_token_ids:
            scores[:, token_id] = -float("inf")
        return scores


def calculate_accuracy(
    predictions: List[str],
    labels: List[str],
    accuracy_mode: str = "fixed_bp",
    score_len_bp: int = 30,
) -> List[Dict[str, float]]:
    if accuracy_mode == "fixed_bp" and score_len_bp <= 0:
        raise ValueError("score_len_bp must be positive when accuracy_mode='fixed_bp'")

    metrics = []
    for label, pred in zip(labels, predictions):
        if accuracy_mode == "fixed_bp":
            scored_bp = score_len_bp
        elif accuracy_mode == "prediction_length":
            scored_bp = min(len(label), len(pred))
        else:
            raise ValueError(f"Unsupported accuracy_mode: {accuracy_mode}")

        same_count = sum(
            1
            for i in range(min(len(label), len(pred), scored_bp))
            if label[i] == pred[i]
        )

        accuracy = same_count / scored_bp if scored_bp > 0 else 0.0
        metrics.append({"accuracy": accuracy, "scored_bp": scored_bp})

    return metrics


def load_parquet_hf(data_path: str, data_type: str) -> pd.DataFrame:
    parquet_path = f"{data_path}/{data_type}/test.parquet"
    return pd.read_parquet(parquet_path)


def _load_model_and_tokenizer(
    model: str,
    revision: Optional[str],
    dtype: torch.dtype,
    attn_implementation: Optional[str] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "revision": revision,
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model_obj = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    return model_obj, tokenizer


SPECIES_TAG_MAP = {
    "vertebrate_mammalian": "<mammalian_species>",
    "vertebrate_other": "<vertebrate_non_mammalian_species>",
    "fungi": "<fungi_species>",
    "plant": "<plant_species>",
    "protozoa": "<protozoan_species>",
    "invertebrate": "<invertebrate_species>",
}


def _build_prompt(
    seq: str, species_type: Optional[str], args: argparse.Namespace
) -> str:
    if args.use_dna_tags:
        prefix = "<dna>"
    elif args.no_prefix:
        prefix = ""
    else:
        prefix = "<s>"
    sp_tag = (
        SPECIES_TAG_MAP.get(species_type, "")
        if args.use_species_tags and species_type is not None
        else ""
    )
    truncated = truncate_prompt_sequence(
        seq,
        args,
        align_multiple=max(1, int(getattr(args, "bp_per_token", 6))),
    )
    return sp_tag + prefix + truncated


def resolve_prompt_len_bp(args: argparse.Namespace) -> int:
    prompt_len_bp = (
        args.prompt_len_bp
        if getattr(args, "prompt_len_bp", None) is not None
        else args.max_seq_len
    )
    if prompt_len_bp <= 0:
        raise ValueError("Prompt length must be positive")
    return int(prompt_len_bp)


def truncate_prompt_sequence(
    seq: str,
    args: argparse.Namespace,
    align_multiple: Optional[int] = None,
) -> str:
    keep_len = min(len(seq), resolve_prompt_len_bp(args))
    if align_multiple is not None and align_multiple > 1:
        keep_len = (keep_len // align_multiple) * align_multiple
    if keep_len <= 0:
        return ""
    return seq[-keep_len:]


def effective_prompt_lengths_bp(
    prompt_sequences: pd.Series,
    args: argparse.Namespace,
    align_multiple: Optional[int] = None,
) -> pd.Series:
    requested_prompt_len_bp = resolve_prompt_len_bp(args)
    lengths = prompt_sequences.str.len().clip(upper=requested_prompt_len_bp)
    if align_multiple is not None and align_multiple > 1:
        lengths = (lengths // align_multiple) * align_multiple
    return lengths.astype(int)


def infer_requested_rollout_bp(args: argparse.Namespace) -> int:
    if args.accuracy_mode == "fixed_bp":
        return args.score_len_bp
    if args.use_evo2:
        return args.gen_len_bp
    return args.gen_len * args.bp_per_token


def prepare_eval_dataframe(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.use_species_tags and "type" not in df.columns:
        raise ValueError("--use_species_tags requires a dataset with a 'type' column")

    prepared_df = df.copy()
    prepared_df["dataset_label"] = prepared_df["label"]
    prepared_df["dataset_label_len_bp"] = prepared_df["dataset_label"].str.len()
    requested_rollout_bp = infer_requested_rollout_bp(args)

    if requested_rollout_bp <= 0:
        raise ValueError("Requested rollout length must be positive")

    min_dataset_label_len_bp = int(prepared_df["dataset_label_len_bp"].min())

    if args.label_source == "dataset":
        use_sequence_tail = False
    elif args.label_source == "sequence_tail":
        use_sequence_tail = True
    else:
        use_sequence_tail = requested_rollout_bp > min_dataset_label_len_bp

    if use_sequence_tail:
        too_short = prepared_df["sequence"].str.len() <= requested_rollout_bp
        if too_short.any():
            shortest_sequence_bp = int(prepared_df["sequence"].str.len().min())
            raise ValueError(
                "Cannot derive sequence-tail labels because at least one input sequence "
                f"has length {shortest_sequence_bp} bp, which is not longer than the "
                f"requested tail label length of {requested_rollout_bp} bp"
            )

        prepared_df["prompt_sequence"] = prepared_df["sequence"].str.slice(
            stop=-requested_rollout_bp
        )
        prepared_df["label"] = prepared_df["sequence"].str[-requested_rollout_bp:]
        prepared_df["label_source"] = "sequence_tail"
        prepared_df["label_len_bp"] = requested_rollout_bp
    else:
        prepared_df["prompt_sequence"] = prepared_df["sequence"]
        prepared_df["label_source"] = "dataset"
        prepared_df["label_len_bp"] = prepared_df["dataset_label_len_bp"]

    return prepared_df


def _decode_gen_tokens(tokenizer, token_lists, args):
    # Hybrid tokenizer's batch_decode mangles DNA tokens, so decode per sample.
    if args.use_dna_tags:
        return [tokenizer.decode(ids) for ids in token_lists]
    return tokenizer.batch_decode(token_lists, skip_special_tokens=True)


def _run_static_generation(
    shard_id,
    model,
    tokenizer,
    truncated_prompts,
    indices,
    device,
    args,
    logits_processor,
):
    predictions = []
    total = len(truncated_prompts)
    with tqdm(total=total, desc=f"Shard {shard_id}", unit="seq") as pbar:
        for i in range(0, total, args.batch_size):
            batch_prompts = truncated_prompts[i : i + args.batch_size]
            batch_indices = indices[i : i + args.batch_size]

            inputs = tokenizer(
                batch_prompts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            else:
                inputs = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.gen_len,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    logits_processor=logits_processor,
                )

            batch_token_lists = [
                outputs[row, -args.gen_len :].tolist()
                for row in range(outputs.shape[0])
            ]
            batch_preds = _decode_gen_tokens(tokenizer, batch_token_lists, args)

            for pred, hash_index in zip(batch_preds, batch_indices):
                predictions.append({"hash_index": hash_index, "pred": pred})
            pbar.update(len(batch_prompts))
    return predictions


def _run_continuous_generation(
    model, tokenizer, truncated_prompts, indices, args, logits_processor
):
    input_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in truncated_prompts
    ]
    generation_config = GenerationConfig(
        max_new_tokens=args.gen_len,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=-1,
        do_sample=False,
    )
    cb_config = ContinuousBatchingConfig(
        max_batch_tokens=args.cb_max_batch_tokens,
        max_memory_percent=args.cb_max_memory_percent,
        scheduler_type=args.cb_scheduler_type,
        use_cuda_graph=args.cb_use_cuda_graph if args.cb_use_cuda_graph else None,
        use_async_batching=(
            args.cb_use_async_batching if args.cb_use_async_batching else None
        ),
        use_default_compile_configs=args.cb_use_default_compile_configs,
        q_padding_interval_size=args.cb_q_padding_interval_size,
        kv_padding_interval_size=args.cb_kv_padding_interval_size,
        max_cached_graphs=args.cb_max_cached_graphs,
        max_blocks_per_request=args.cb_max_blocks_per_request,
    )

    outputs = model.generate_batch(
        inputs=input_ids,
        generation_config=generation_config,
        continuous_batching_config=cb_config,
        progress_bar=True,
        logits_processor=logits_processor,
    )
    if not outputs:
        raise RuntimeError(
            "Continuous batching returned no results. Try a different "
            "--attn_implementation or disable --cb_use_cuda_graph / "
            "--cb_use_async_batching."
        )

    ordered = [
        out
        for _, out in sorted(outputs.items(), key=lambda kv: int(kv[0].split("_")[-1]))
    ]
    token_lists = [o.generated_tokens for o in ordered]
    preds = _decode_gen_tokens(tokenizer, token_lists, args)
    return [{"hash_index": h, "pred": p} for h, p in zip(indices, preds)]


def process_data_shard(shard_id, sequences_data, args, dtype):
    torch.cuda.set_device(shard_id)
    device = f"cuda:{shard_id}"
    dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    print(f"Shard {shard_id}: Loading model on GPU {shard_id}...")
    model, tokenizer = _load_model_and_tokenizer(
        args.model,
        args.revision,
        dtype,
        attn_implementation=getattr(args, "attn_implementation", None),
    )
    model = model.to(device)

    if getattr(args, "upcast_lm_head", False) and hasattr(model, "lm_head"):
        # Wrap lm_head forward to compute in fp32 (weights stay bf16, cast on the fly)
        import torch.nn.functional as F

        _original_lm_head = model.lm_head

        def _fp32_lm_head_forward(input):
            return F.linear(
                input.float(),
                _original_lm_head.weight.float(),
                (
                    _original_lm_head.bias.float()
                    if _original_lm_head.bias is not None
                    else None
                ),
            )

        model.lm_head.forward = _fp32_lm_head_forward
        print(f"Shard {shard_id}: Wrapped lm_head forward to compute in fp32")

    tokenizer.padding_side = "left"

    # Get special token IDs - handle different tokenizer implementations
    if hasattr(tokenizer, "special_tokens"):
        special_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens)
    elif hasattr(tokenizer, "all_special_ids"):
        special_token_ids = tokenizer.all_special_ids
    else:
        special_token_ids = []
    logits_processor = LogitsProcessorList(
        [SuppressSpecialTokensLogitsProcessor(special_token_ids)]
    )

    sequences_shard = [item["prompt_sequence"] for item in sequences_data]
    indices_shard = [item["hash_index"] for item in sequences_data]
    species_types = (
        [item.get("type") for item in sequences_data]
        if args.use_species_tags
        else [None] * len(sequences_shard)
    )
    truncated_prompts = [
        _build_prompt(seq, sp_type, args)
        for seq, sp_type in zip(sequences_shard, species_types)
    ]

    if args.generation_backend == "continuous":
        predictions = _run_continuous_generation(
            model, tokenizer, truncated_prompts, indices_shard, args, logits_processor
        )
    else:
        predictions = _run_static_generation(
            shard_id,
            model,
            tokenizer,
            truncated_prompts,
            indices_shard,
            device,
            args,
            logits_processor,
        )

    del model
    torch.cuda.empty_cache()

    return predictions


def _evo2_model_name(model_arg: str) -> str:
    return model_arg.split("/")[-1]


def process_data_evo2(sequences_data, args):
    try:
        from evo2 import Evo2
    except Exception as e:
        raise RuntimeError(
            "Evo2 library not available; install evo2 to use --use_evo2"
        ) from e

    # Do NOT set_device(0) — Evo-2's inference pipeline automatically shards large models across all visible GPUs. See https://github.com/ArcInstitute/evo2/tree/main?tab=readme-ov-file#setup
    # For multi-GPU models (20B, 40B), set CUDA_VISIBLE_DEVICES in the SLURM script.
    model_name = _evo2_model_name(args.model)
    model = Evo2(model_name)

    sequences = [item["prompt_sequence"] for item in sequences_data]
    indices = [item["hash_index"] for item in sequences_data]
    total_sequences = len(sequences)

    predictions = []

    # Use requested batch size - try it for each batch, fallback to individual if OOM
    evo2_batch_size = getattr(args, "batch_size", 8)
    print(
        f"Using batch_size={evo2_batch_size} for evo2 (will fallback to individual if OOM)"
    )

    with tqdm(total=total_sequences, desc="Evo2", unit="seq") as pbar:
        for i in range(0, total_sequences, evo2_batch_size):
            batch_seqs = sequences[i : i + evo2_batch_size]
            batch_indices = indices[i : i + evo2_batch_size]

            # Prepare prompts for batch
            batch_prompts = [truncate_prompt_sequence(seq, args) for seq in batch_seqs]

            try:
                # Try batch generation with original batch size
                if evo2_batch_size > 1:
                    output = model.generate(
                        prompt_seqs=batch_prompts,
                        n_tokens=args.gen_len_bp,
                        temperature=1.0,
                        top_k=1,
                        top_p=0.0,
                        verbose=0,
                        force_prompt_threshold=args.evo2_force_prompt_threshold,
                    )
                    # Extract predictions for each sequence in batch
                    for j, (pred, hash_index) in enumerate(
                        zip(output.sequences, batch_indices)
                    ):
                        predictions.append({"hash_index": hash_index, "pred": pred})
                else:
                    # Process individually
                    for seq, hash_index in zip(batch_seqs, batch_indices):
                        prompt = truncate_prompt_sequence(seq, args)
                        output = model.generate(
                            prompt_seqs=[prompt],
                            n_tokens=args.gen_len_bp,
                            temperature=1.0,
                            top_k=1,
                            top_p=0.0,
                            verbose=0,
                            force_prompt_threshold=args.evo2_force_prompt_threshold,
                        )
                        predictions.append(
                            {"hash_index": hash_index, "pred": output.sequences[0]}
                        )

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # Handle OOM or 32-bit indexing errors - fallback to individual for this batch only
                is_oom = isinstance(
                    e, torch.cuda.OutOfMemoryError
                ) or "OutOfMemory" in str(e)
                is_32bit = "32BitIndexMath" in str(e)

                if (is_oom or is_32bit) and evo2_batch_size > 1:
                    # Fallback to individual processing for this batch
                    print(
                        f"\n{'OOM' if is_oom else '32-bit indexing'} error with batch_size={evo2_batch_size}, "
                        f"falling back to individual processing for this batch"
                    )
                    torch.cuda.empty_cache()
                    for seq, hash_index in zip(batch_seqs, batch_indices):
                        prompt = truncate_prompt_sequence(seq, args)
                        output = model.generate(
                            prompt_seqs=[prompt],
                            n_tokens=args.gen_len_bp,
                            temperature=1.0,
                            top_k=1,
                            top_p=0.0,
                            verbose=0,
                            force_prompt_threshold=args.evo2_force_prompt_threshold,
                        )
                        predictions.append(
                            {"hash_index": hash_index, "pred": output.sequences[0]}
                        )
                else:
                    # Some other error or already individual - re-raise
                    print(f"\nError during processing: {e}")
                    raise

            pbar.update(len(batch_seqs))

            # Clear cache periodically to help with memory fragmentation
            if (i + evo2_batch_size) % (evo2_batch_size * 20) == 0:
                torch.cuda.empty_cache()

    return predictions


def _run_vllm_engine(sequences_data, args, tp_size):
    """Run a single vLLM engine against a list of sequence records and return
    predictions. Expects the current process to have the desired GPUs visible."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    dtype = "bfloat16" if args.bf16 else "float32"
    max_model_len = args.vllm_max_model_len or (
        resolve_prompt_len_bp(args) // args.bp_per_token + args.gen_len + 8
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype=dtype,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    if hasattr(tokenizer, "special_tokens"):
        special_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens)
    elif hasattr(tokenizer, "all_special_ids"):
        special_token_ids = tokenizer.all_special_ids
    else:
        special_token_ids = []

    # vLLM 0.19 removed per-request logits_processors from SamplingParams.
    # Use logit_bias to push special-token logits far below any other logit so
    # they never win argmax under greedy decoding (matches HF path's -inf).
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.gen_len,
        n=1,
        logit_bias={int(sid): -1e9 for sid in special_token_ids},
    )

    species_types = (
        [item.get("type") for item in sequences_data]
        if args.use_species_tags
        else [None] * len(sequences_data)
    )
    prompts_text = [
        _build_prompt(item["prompt_sequence"], sp, args)
        for item, sp in zip(sequences_data, species_types)
    ]

    # Pre-tokenize with add_special_tokens=False to match the HF path exactly
    # and preserve the hybrid <dna> tokenizer's 6-mer segmentation (which vLLM's
    # default tokenize step would otherwise disturb).
    token_prompts = [
        TokensPrompt(
            prompt_token_ids=list(
                tokenizer(text, add_special_tokens=False)["input_ids"]
            )
        )
        for text in prompts_text
    ]

    outputs = llm.generate(token_prompts, sampling, use_tqdm=True)

    predictions = []
    for item, out in zip(sequences_data, outputs):
        gen_ids = list(out.outputs[0].token_ids)
        # Per-sample decode mirrors the HF use_dna_tags branch and is cheap.
        pred = tokenizer.decode(gen_ids)
        predictions.append({"hash_index": item["hash_index"], "pred": pred})

    return predictions


def _vllm_dp_worker(rank, sequences_data, args):
    """DP worker: pin to a single GPU, then run a standalone vLLM engine on
    this rank's shard. Must be picklable (module-level function)."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    return _run_vllm_engine(sequences_data, args, tp_size=1)


def process_data_vllm(sequences_data, args):
    try:
        import vllm  # noqa: F401  ensure vLLM is installed before dispatching
    except Exception as e:
        raise RuntimeError("vLLM not available; install vllm to use --use_vllm") from e

    tp_size = args.vllm_tensor_parallel_size
    dp_size = args.vllm_data_parallel_size

    print(
        f"vLLM: model={args.model} revision={args.revision} "
        f"dtype={'bfloat16' if args.bf16 else 'float32'} tp={tp_size} dp={dp_size}"
    )

    if dp_size <= 1:
        return _run_vllm_engine(sequences_data, args, tp_size=tp_size)

    # Data-parallel: N independent vLLM engines, each pinned to one GPU.
    # Equal static partitioning of prompts across ranks.
    n = len(sequences_data)
    shard_size = (n + dp_size - 1) // dp_size
    shards = []
    for rank in range(dp_size):
        start = rank * shard_size
        end = min(start + shard_size, n)
        if start < end:
            shards.append((rank, sequences_data[start:end]))
    print(
        f"vLLM DP: sharding {n} prompts across {len(shards)} workers "
        f"(~{shard_size} per worker)"
    )

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    all_predictions = []
    with ProcessPoolExecutor(max_workers=len(shards), mp_context=ctx) as executor:
        future_to_rank = {
            executor.submit(_vllm_dp_worker, rank, shard, args): rank
            for rank, shard in shards
        }
        for future in as_completed(future_to_rank):
            rank = future_to_rank[future]
            preds = future.result()
            all_predictions.extend(preds)
            print(f"vLLM DP rank {rank} completed: {len(preds)} predictions")

    return all_predictions


def process_checkpoint(args: argparse.Namespace, dtype: str) -> Dict:
    print("\n" + "=" * 80)
    print("🧬  SEQUENCE RECOVERY EVAL  🧬")
    print("=" * 80 + "\n")
    print(f"Model: {args.model}")
    if args.revision:
        print(f"Revision: {args.revision}")
    print(f"Data: {args.data_path}/{args.data_type}/test.parquet")

    df = load_parquet_hf(args.data_path, args.data_type)
    df = prepare_eval_dataframe(df, args)

    # Show dataset info
    if "type" in df.columns:
        type_counts = df["type"].value_counts()
        print(f"Dataset contains {len(df)} sequences with {len(type_counts)} types:")
        for type_name, count in type_counts.items():
            print(f"  - {type_name}: {count} sequences")

    sequence_lengths = df["sequence"].str.len()
    dataset_label_lengths = df["dataset_label_len_bp"]
    print(
        "Sequence lengths (bp): "
        f"min={int(sequence_lengths.min())}, "
        f"median={int(sequence_lengths.median())}, "
        f"max={int(sequence_lengths.max())}"
    )
    print(
        "Dataset label lengths (bp): "
        f"min={int(dataset_label_lengths.min())}, "
        f"max={int(dataset_label_lengths.max())}"
    )
    active_label_source = df["label_source"].iloc[0]
    active_label_len_bp = (
        int(df["label_len_bp"].iloc[0])
        if active_label_source == "sequence_tail"
        else int(dataset_label_lengths.min())
    )
    print(
        f"Evaluation label source: {active_label_source} "
        f"(requested_rollout_bp={infer_requested_rollout_bp(args)}, "
        f"label_len_bp={active_label_len_bp})"
    )

    # Limit number of samples for testing if requested
    if args.max_samples is not None and args.max_samples > 0:
        original_len = len(df)
        original_df = df.copy()

        # Use random sampling to get a representative subset across types
        if "type" in df.columns and len(df["type"].unique()) > 1:
            # Sample proportionally across types to get better representation
            samples_per_type = max(1, args.max_samples // len(df["type"].unique()))
            sampled_dfs = []
            sampled_indices = set()

            for type_name, group in df.groupby("type"):
                sample_size = min(len(group), samples_per_type)
                sampled = group.sample(sample_size, random_state=args.sample_seed)
                sampled_dfs.append(sampled)
                sampled_indices.update(sampled.index)

            sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)

            # If we got fewer samples than requested, randomly sample more from remaining data
            if len(sampled_df) < args.max_samples:
                remaining = args.max_samples - len(sampled_df)
                # Get remaining rows from original dataframe (exclude already sampled)
                remaining_df = original_df[~original_df.index.isin(sampled_indices)]
                if len(remaining_df) > 0:
                    additional = remaining_df.sample(
                        min(remaining, len(remaining_df)),
                        random_state=args.sample_seed,
                    )
                    sampled_df = pd.concat([sampled_df, additional]).reset_index(
                        drop=True
                    )
            # Ensure we don't exceed max_samples
            df = sampled_df.head(args.max_samples).copy()
        else:
            # No type column or only one type, use random sampling
            df = df.sample(
                min(args.max_samples, len(df)),
                random_state=args.sample_seed,
            ).reset_index(drop=True)

        print(
            f"⚠️  TEST MODE: Limited to {len(df)} samples (from {original_len} total)"
        )
        if "type" in df.columns:
            test_type_counts = df["type"].value_counts()
            print(f"Test subset contains {len(test_type_counts)} types:")
            for type_name, count in test_type_counts.items():
                print(f"  - {type_name}: {count} sequences")

    total_sequences = len(df)
    prompt_align_multiple = None if args.use_evo2 else max(1, int(args.bp_per_token))
    prompt_lengths_bp = effective_prompt_lengths_bp(
        df["prompt_sequence"], args, align_multiple=prompt_align_multiple
    )
    requested_prompt_len_bp = resolve_prompt_len_bp(args)
    print(
        "Prompt lengths after label holdout/truncation (bp): "
        f"requested={requested_prompt_len_bp}, "
        f"min={int(prompt_lengths_bp.min())}, "
        f"mean={float(prompt_lengths_bp.mean()):.1f}, "
        f"max={int(prompt_lengths_bp.max())}"
    )

    print("Generating hash indices for sequences...")
    df["hash_index"] = df.apply(
        lambda row: hashlib.md5(f"{row['sequence']}_{row.name}".encode()).hexdigest()[
            :16
        ],
        axis=1,
    )

    num_gpus = torch.cuda.device_count()

    if args.use_evo2:
        print(f"Using evo2 backend ({num_gpus} visible GPUs, sharded internally)")
        all_predictions = process_data_evo2(
            df[["prompt_sequence", "hash_index"]].to_dict("records"), args
        )
    elif args.use_vllm:
        print(f"Using vLLM backend ({num_gpus} visible GPUs)")
        vllm_cols = ["prompt_sequence", "hash_index"]
        if args.use_species_tags and "type" in df.columns:
            vllm_cols.append("type")
        all_predictions = process_data_vllm(df[vllm_cols].to_dict("records"), args)
    else:
        print(f"Using {num_gpus} GPUs with data sharding")

        shard_size = (total_sequences + num_gpus - 1) // num_gpus
        shards = []

        for i in range(num_gpus):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, total_sequences)
            if start_idx < total_sequences:
                shard_df = df.iloc[start_idx:end_idx].copy()
                shard_cols = ["prompt_sequence", "hash_index"]
                if args.use_species_tags and "type" in shard_df.columns:
                    shard_cols.append("type")
                shards.append(
                    {
                        "shard_id": i,
                        "data": shard_df[shard_cols].to_dict("records"),
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                    }
                )

        print(f"Data divided into {len(shards)} shards")
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            future_to_shard = {}
            for shard in shards:
                future = executor.submit(
                    process_data_shard,
                    shard["shard_id"],
                    shard["data"],
                    args,
                    dtype,
                )
                future_to_shard[future] = shard["shard_id"]

            all_predictions = []
            for future in as_completed(future_to_shard):
                shard_id = future_to_shard[future]
                try:
                    shard_predictions = future.result()
                    all_predictions.extend(shard_predictions)
                    print(
                        f"Shard {shard_id} completed, collected {len(shard_predictions)} predictions"
                    )
                except Exception as e:
                    print(f"Shard {shard_id} generated an exception: {e}")

        elapsed_time = time.time() - start_time
        print(f"All shards completed in {elapsed_time:.2f} seconds")

    pred_df = pd.DataFrame(all_predictions)
    results_df = df.merge(pred_df, on="hash_index", how="left", suffixes=("", "_pred"))

    missing_count = results_df["pred"].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} sequences missing predictions")
        results_df["pred"] = results_df["pred"].fillna("")

    final_predictions = results_df["pred"].tolist()
    final_labels = results_df["label"].tolist()

    metrics = calculate_accuracy(
        final_predictions,
        final_labels,
        accuracy_mode=args.accuracy_mode,
        score_len_bp=args.score_len_bp,
    )
    results_df["accuracy"] = [item["accuracy"] for item in metrics]
    results_df["scored_bp"] = [item["scored_bp"] for item in metrics]

    if "type" in results_df.columns:
        type_means = results_df.groupby("type")["accuracy"].mean()
    else:
        type_means = pd.Series(dtype=float)
    overall_mean = results_df["accuracy"].mean()
    mean_scored_bp = results_df["scored_bp"].mean()

    os.makedirs(args.output_dir, exist_ok=True)
    # Use --model_name if provided, otherwise fall back to last component of path
    model_name = args.model_name if args.model_name else args.model.split("/")[-1]
    revision_tag = args.revision or "main"
    upcast_suffix = "_upcast-lm-head" if getattr(args, "upcast_lm_head", False) else ""
    test_suffix = f"_test{args.max_samples}" if args.max_samples is not None else ""
    # If model_name is already provided (e.g., "hybrid_50B_gener_24000"), use simpler naming
    if args.model_name:
        output_basename = f"{model_name}_{dtype}{upcast_suffix}{test_suffix}"
    else:
        output_basename = f"{model_name}_{revision_tag}_{args.data_type}_{dtype}{upcast_suffix}{test_suffix}"

    output_path = os.path.join(args.output_dir, f"{output_basename}.parquet")
    output_columns = [
        "hash_index",
        "pred",
        "label",
        "dataset_label",
        "label_source",
        "label_len_bp",
        "accuracy",
        "scored_bp",
    ]
    if "type" in results_df.columns:
        output_columns.append("type")
    results_df[output_columns].to_parquet(output_path)

    summary = {
        "model": args.model,
        "revision": args.revision,
        "data_type": args.data_type,
        "overall_accuracy": float(overall_mean),
        "type_accuracy": {k: float(v) for k, v in type_means.items()},
        "num_sequences": int(total_sequences),
        "dtype": dtype,
        "accuracy_mode": args.accuracy_mode,
        "score_len_bp": args.score_len_bp,
        "label_source": active_label_source,
        "requested_rollout_bp": int(infer_requested_rollout_bp(args)),
        "requested_prompt_len_bp": int(requested_prompt_len_bp),
        "effective_prompt_len_bp_min": int(prompt_lengths_bp.min()),
        "effective_prompt_len_bp_mean": float(prompt_lengths_bp.mean()),
        "effective_prompt_len_bp_max": int(prompt_lengths_bp.max()),
        "bp_per_token": int(args.bp_per_token),
        "mean_scored_bp": float(mean_scored_bp),
        "visible_gpu_count": int(num_gpus),
        "generation_backend": (
            "vllm"
            if args.use_vllm
            else ("evo2" if args.use_evo2 else f"hf-{args.generation_backend}")
        ),
        "attn_implementation": getattr(args, "attn_implementation", None),
        "timestamp": time.time(),
    }

    summary_path = os.path.join(args.output_dir, f"{output_basename}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Overall accuracy: {overall_mean:.4f}")
    print(f"Type-wise accuracy:\n{type_means}")
    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")

    return {
        "parquet": output_path,
        "summary": summary_path,
        "summary_dict": summary,
    }


def maybe_push_to_hub(args, parquet_path: str, summary_path: str):
    if not args.push_to_hub:
        return
    if not args.hub_repo_id:
        raise ValueError("--hub_repo_id is required when --push_to_hub is set")

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Uploading results to {args.hub_repo_id} ({args.hub_repo_type})")
    api.upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo=os.path.basename(parquet_path),
        repo_id=args.hub_repo_id,
        repo_type=args.hub_repo_type,
    )
    api.upload_file(
        path_or_fileobj=summary_path,
        path_in_repo=os.path.basename(summary_path),
        repo_id=args.hub_repo_id,
        repo_type=args.hub_repo_type,
    )


def main():
    args = parse_args()
    dtype = "bfloat16" if args.bf16 else "float32"

    results = process_checkpoint(args, dtype)
    maybe_push_to_hub(args, results["parquet"], results["summary"])


if __name__ == "__main__":
    main()
