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
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList


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
        help="Max input length in bp (truncate left, keep rightmost)",
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
    return parser.parse_args()


class SuppressSpecialTokensLogitsProcessor:
    """Suppress all special tokens during generation by setting logits to -inf."""

    def __init__(self, special_token_ids: list):
        self.special_token_ids = special_token_ids

    def __call__(self, input_ids, scores):
        for token_id in self.special_token_ids:
            scores[:, token_id] = -float("inf")
        return scores


def calculate_accuracy(
    predictions: List[str], labels: List[str], seq_length: int = 30
) -> List[float]:
    accuracies = []
    for label, pred in zip(labels, predictions):
        same_count = sum(
            1
            for i in range(min(len(label), len(pred), seq_length))
            if label[i] == pred[i]
        )
        accuracies.append(same_count / seq_length)
    return accuracies


def load_parquet_hf(data_path: str, data_type: str) -> pd.DataFrame:
    parquet_path = f"{data_path}/{data_type}/test.parquet"
    return pd.read_parquet(parquet_path)


def _load_model_and_tokenizer(model: str, revision: Optional[str], dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_obj = AutoModelForCausalLM.from_pretrained(
        model, revision=revision, trust_remote_code=True, dtype=dtype
    )
    return model_obj, tokenizer


SPECIES_TAG_MAP = {
    "vertebrate_mammalian": "<mammalian_species>",
    "vertebrate_other": "<vertebrate_non_mammalian_species>",
    "fungi": "<fungi_species>",
    "plant": "<plant_species>",
    "protozoa": "<protozoan_species>",
    "invertebrate": "<invertebrate_species>",
}


def process_data_shard(shard_id, sequences_data, args, dtype):
    torch.cuda.set_device(shard_id)
    device = f"cuda:{shard_id}"
    dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    print(f"Shard {shard_id}: Loading model on GPU {shard_id}...")
    model, tokenizer = _load_model_and_tokenizer(args.model, args.revision, dtype)
    model = model.to(device)

    if getattr(args, 'upcast_lm_head', False) and hasattr(model, 'lm_head'):
        # Wrap lm_head forward to compute in fp32 (weights stay bf16, cast on the fly)
        import torch.nn.functional as F
        _original_lm_head = model.lm_head
        def _fp32_lm_head_forward(input):
            return F.linear(input.float(), _original_lm_head.weight.float(),
                          _original_lm_head.bias.float() if _original_lm_head.bias is not None else None)
        model.lm_head.forward = _fp32_lm_head_forward
        print(f"Shard {shard_id}: Wrapped lm_head forward to compute in fp32")

    tokenizer.padding_side = "left"

    # Get special token IDs - handle different tokenizer implementations
    if hasattr(tokenizer, 'special_tokens'):
        special_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens)
    elif hasattr(tokenizer, 'all_special_ids'):
        special_token_ids = tokenizer.all_special_ids
    else:
        special_token_ids = []
    logits_processor = LogitsProcessorList(
        [SuppressSpecialTokensLogitsProcessor(special_token_ids)]
    )

    sequences_shard = [item["sequence"] for item in sequences_data]
    indices_shard = [item["hash_index"] for item in sequences_data]
    species_types = [item.get("type") for item in sequences_data] if args.use_species_tags else None
    total_sequences = len(sequences_shard)

    predictions = []

    with tqdm(total=total_sequences, desc=f"Shard {shard_id}", unit="seq") as pbar:
        for i in range(0, total_sequences, args.batch_size):
            batch_seqs = sequences_shard[i : i + args.batch_size]
            batch_indices = indices_shard[i : i + args.batch_size]

            if args.use_dna_tags:
                # For hybrid tokenizer: wrap with <dna> tag to trigger 6-mer tokenization
                prefix = "<dna>"
            elif args.no_prefix:
                # No prefix token
                prefix = ""
            else:
                # Default: use <s> as BOS token for pure 6-mer models
                prefix = "<s>"

            if args.use_species_tags and species_types is not None:
                # Per-sequence species tag prefix: <species_tag><dna>SEQUENCE
                batch_species = species_types[i : i + args.batch_size]
                truncated_seqs = []
                for seq, sp_type in zip(batch_seqs, batch_species):
                    sp_tag = SPECIES_TAG_MAP.get(sp_type, "")
                    truncated_seq = seq[-((min(len(seq), args.max_seq_len) // 6) * 6) :]
                    truncated_seqs.append(sp_tag + prefix + truncated_seq)
            else:
                truncated_seqs = [
                    prefix + seq[-((min(len(seq), args.max_seq_len) // 6) * 6) :]
                    for seq in batch_seqs
                ]

            inputs = tokenizer(
                truncated_seqs,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            # Handle both BatchEncoding objects and plain dicts (e.g., HybridDNATokenizer)
            if hasattr(inputs, 'to'):
                inputs = inputs.to(device)
            else:
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.gen_len,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    logits_processor=logits_processor,
                )

            # For hybrid tokenizer, batch_decode doesn't work correctly with DNA tokens,
            # so we use decode in a loop instead
            if args.use_dna_tags:
                batch_preds = [
                    tokenizer.decode(outputs[i, -args.gen_len :].tolist())
                    for i in range(outputs.shape[0])
                ]
            else:
                batch_preds = tokenizer.batch_decode(
                    outputs[:, -args.gen_len :], skip_special_tokens=True
                )

            for pred, hash_index in zip(batch_preds, batch_indices):
                predictions.append({"hash_index": hash_index, "pred": pred})

            pbar.update(len(batch_seqs))

    del model
    torch.cuda.empty_cache()

    return predictions


def _evo2_model_name(model_arg: str) -> str:
    return model_arg.split("/")[-1]


def _patch_evo2_config_no_flash(model_name: str) -> None:
    try:
        from evo2.utils import CONFIG_MAP
    except Exception:
        return
    config_path = CONFIG_MAP.get(model_name)
    if not config_path or not os.path.exists(config_path):
        return
    if config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    config["use_flash_attn"] = False
    tmp_path = os.path.join("/tmp", f"{model_name}_no_flash.yml")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    CONFIG_MAP[model_name] = tmp_path


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
    _patch_evo2_config_no_flash(model_name)
    model = Evo2(model_name)

    sequences = [item["sequence"] for item in sequences_data]
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
            batch_prompts = [
                seq[-min(len(seq), args.max_seq_len) :] for seq in batch_seqs
            ]

            try:
                # Try batch generation with original batch size
                if evo2_batch_size > 1:
                    output = model.generate(
                        prompt_seqs=batch_prompts,
                        n_tokens=args.gen_len_bp,
                        temperature=0.0,
                        do_sample=False,
                    )
                    # Extract predictions for each sequence in batch
                    for j, (pred, hash_index) in enumerate(
                        zip(output.sequences, batch_indices)
                    ):
                        predictions.append({"hash_index": hash_index, "pred": pred})
                else:
                    # Process individually
                    for seq, hash_index in zip(batch_seqs, batch_indices):
                        prompt = seq[-min(len(seq), args.max_seq_len) :]
                        output = model.generate(
                            prompt_seqs=[prompt],
                            n_tokens=args.gen_len_bp,
                            temperature=0.0,
                            do_sample=False,
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
                        prompt = seq[-min(len(seq), args.max_seq_len) :]
                        output = model.generate(
                            prompt_seqs=[prompt],
                            n_tokens=args.gen_len_bp,
                            temperature=0.0,
                            do_sample=False,
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


def process_checkpoint(args: argparse.Namespace, dtype: str) -> Dict:
    print("\n" + "=" * 80)
    print("🧬  SEQUENCE RECOVERY EVAL  🧬")
    print("=" * 80 + "\n")
    print(f"Model: {args.model}")
    if args.revision:
        print(f"Revision: {args.revision}")
    print(f"Data: {args.data_path}/{args.data_type}/test.parquet")

    df = load_parquet_hf(args.data_path, args.data_type)

    # Show dataset info
    if "type" in df.columns:
        type_counts = df["type"].value_counts()
        print(f"Dataset contains {len(df)} sequences with {len(type_counts)} types:")
        for type_name, count in type_counts.items():
            print(f"  - {type_name}: {count} sequences")

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
                sampled = group.sample(sample_size)
                sampled_dfs.append(sampled)
                sampled_indices.update(sampled.index)

            sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)

            # If we got fewer samples than requested, randomly sample more from remaining data
            if len(sampled_df) < args.max_samples:
                remaining = args.max_samples - len(sampled_df)
                # Get remaining rows from original dataframe (exclude already sampled)
                remaining_df = original_df[~original_df.index.isin(sampled_indices)]
                if len(remaining_df) > 0:
                    additional = remaining_df.sample(min(remaining, len(remaining_df)))
                    sampled_df = pd.concat([sampled_df, additional]).reset_index(
                        drop=True
                    )
            # Ensure we don't exceed max_samples
            df = sampled_df.head(args.max_samples).copy()
        else:
            # No type column or only one type, use random sampling
            df = df.sample(min(args.max_samples, len(df))).reset_index(drop=True)

        print(f"⚠️  TEST MODE: Limited to {len(df)} samples (from {original_len} total)")
        if "type" in df.columns:
            test_type_counts = df["type"].value_counts()
            print(f"Test subset contains {len(test_type_counts)} types:")
            for type_name, count in test_type_counts.items():
                print(f"  - {type_name}: {count} sequences")

    total_sequences = len(df)

    print("Generating hash indices for sequences...")
    df["hash_index"] = df.apply(
        lambda row: hashlib.md5(f"{row['sequence']}_{row.name}".encode()).hexdigest()[
            :16
        ],
        axis=1,
    )

    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs with data sharding")

    shard_size = (total_sequences + num_gpus - 1) // num_gpus
    shards = []

    for i in range(num_gpus):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, total_sequences)
        if start_idx < total_sequences:
            shard_df = df.iloc[start_idx:end_idx].copy()
            shard_cols = ["sequence", "hash_index"]
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
    if args.use_evo2:
        all_predictions = process_data_evo2(
            df[["sequence", "hash_index"]].to_dict("records"), args
        )
    else:
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

    accuracies = calculate_accuracy(final_predictions, final_labels)
    results_df["accuracy"] = accuracies

    type_means = results_df.groupby("type")["accuracy"].mean()
    overall_mean = results_df["accuracy"].mean()

    os.makedirs(args.output_dir, exist_ok=True)
    # Use --model_name if provided, otherwise fall back to last component of path
    model_name = args.model_name if args.model_name else args.model.split("/")[-1]
    revision_tag = args.revision or "main"
    upcast_suffix = "_upcast-lm-head" if getattr(args, 'upcast_lm_head', False) else ""
    test_suffix = f"_test{args.max_samples}" if args.max_samples is not None else ""
    # If model_name is already provided (e.g., "hybrid_50B_gener_24000"), use simpler naming
    if args.model_name:
        output_basename = f"{model_name}_{dtype}{upcast_suffix}{test_suffix}"
    else:
        output_basename = (
            f"{model_name}_{revision_tag}_{args.data_type}_{dtype}{upcast_suffix}{test_suffix}"
        )

    output_path = os.path.join(args.output_dir, f"{output_basename}.parquet")
    results_df[["hash_index", "pred", "label", "type", "accuracy"]].to_parquet(
        output_path
    )

    summary = {
        "model": args.model,
        "revision": args.revision,
        "data_type": args.data_type,
        "overall_accuracy": float(overall_mean),
        "type_accuracy": {k: float(v) for k, v in type_means.items()},
        "num_sequences": int(total_sequences),
        "dtype": dtype,
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
