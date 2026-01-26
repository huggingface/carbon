import argparse
import hashlib
import json
import os
import time
from typing import Dict, List, Tuple
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClinVar VEP eval (post-training)")
    parser.add_argument(
        "--hg38_path",
        default="hf://datasets/GenerTeam/variant-effect-prediction/hg38.parquet",
        help="Reference genome parquet path",
    )
    parser.add_argument(
        "--clinvar_path",
        default="hf://datasets/GenerTeam/variant-effect-prediction/ClinVar_VEP_results.parquet",
        help="ClinVar parquet path",
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
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=32,
        help="Processes for probability aggregation",
    )
    parser.add_argument(
        "--output_dir",
        default="./eval_results/clinvar_vep",
        help="Output directory",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=96000,
        help="Context length in bp",
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
        help="HF repo to upload results (e.g., hf-carbon/clinvar-vep-results)",
    )
    parser.add_argument(
        "--hub_repo_type",
        default="dataset",
        choices=["dataset", "model"],
        help="HF repo type",
    )
    return parser.parse_args()


def load_and_prepare_data(
    hg38_path: str, clinvar_path: str, context_length: int
) -> pd.DataFrame:
    print("🧬 Loading genomic data...")
    start_time = time.time()
    seq_df = pd.read_parquet(hg38_path)
    clinvar_df = pd.read_parquet(clinvar_path)

    print(f"📊 Loaded {len(clinvar_df)} ClinVar variants")
    print(f"⚡ Data loading completed in {time.time() - start_time:.2f} seconds")

    print("🧪 Extracting sequences for each variant...")
    sequence_start_time = time.time()
    sequences = []
    for i in tqdm(range(len(clinvar_df)), desc="Sequence Extraction"):
        chrom_id = clinvar_df["chrom"][i]
        location = clinvar_df["pos"][i] - 1
        sequence = seq_df.loc[seq_df["ID"] == "chr" + chrom_id]["Sequence"].values[0][
            max(0, location - context_length) : location
        ]
        sequence = sequence.lstrip("N")
        truncate_length = len(sequence) % 6
        if truncate_length > 0:
            sequence = sequence[truncate_length:]
        sequences.append(sequence)

    clinvar_df["sequence"] = sequences

    print("Generating hash indices for sequences...")
    clinvar_df["hash_index"] = clinvar_df.apply(
        lambda row: hashlib.md5(f"{row['sequence']}_{row.name}".encode()).hexdigest()[
            :16
        ],
        axis=1,
    )

    print(
        f"✅ Sequence extraction completed in {time.time() - sequence_start_time:.2f} seconds"
    )
    print(f"📏 Average sequence length: {np.mean([len(s) for s in sequences]):.1f} bp")

    return clinvar_df


def _load_model_and_tokenizer(model: str, revision: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, trust_remote_code=True
    )
    model_obj = AutoModelForCausalLM.from_pretrained(
        model, revision=revision, trust_remote_code=True, dtype=dtype
    )
    return model_obj, tokenizer


def compute_logits_shard(args):
    shard_id, sequences_data, model, revision, dtype, batch_size = args

    torch.cuda.set_device(shard_id)
    device = f"cuda:{shard_id}"

    model_obj, tokenizer = _load_model_and_tokenizer(
        model, revision, getattr(torch, dtype)
    )
    model_obj = model_obj.to(device)
    model_obj.eval()

    sequences_shard = [item["sequence"] for item in sequences_data]
    indices_shard = [item["hash_index"] for item in sequences_data]
    total_sequences = len(sequences_shard)

    logits_shard = []

    with tqdm(total=total_sequences, desc=f"Shard {shard_id}", unit="seq") as pbar:
        for i in range(0, total_sequences, batch_size):
            batch_sequences = sequences_shard[i : i + batch_size]
            batch_indices = indices_shard[i : i + batch_size]

            inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model_obj(**inputs)

            for j, seq in enumerate(batch_sequences):
                seq_len = len(tokenizer(seq).input_ids)
                last_token_logits = outputs.logits[j, seq_len - 2, :]
                probs = (
                    F.softmax(last_token_logits, dim=0).cpu().float().numpy().tolist()
                )
                logits_shard.append({"hash_index": batch_indices[j], "logits": probs})

            pbar.update(len(batch_sequences))

    del model_obj
    torch.cuda.empty_cache()

    return logits_shard


def compute_logits_parallel(
    clinvar_df: pd.DataFrame,
    model: str,
    revision: str,
    dtype: str,
    batch_size: int = 32,
) -> List[List[float]]:
    print("🧠 Computing logits using parallel GPU processing...")
    start_time = time.time()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected for parallel logit computation")
    print(f"Using {num_gpus} GPUs for parallel computation")

    sequences_data = clinvar_df[["sequence", "hash_index"]].to_dict("records")
    total_sequences = len(sequences_data)

    shard_size = (total_sequences + num_gpus - 1) // num_gpus
    shards = []

    for i in range(num_gpus):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, total_sequences)
        if start_idx < total_sequences:
            shards.append(
                {
                    "shard_id": i,
                    "sequences_data": sequences_data[start_idx:end_idx],
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )

    print(f"Data divided into {len(shards)} shards")

    args_list = []
    for shard in shards:
        args_list.append(
            (
                shard["shard_id"],
                shard["sequences_data"],
                model,
                revision,
                dtype,
                batch_size,
            )
        )

    all_logits_dict = {}

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_gpus) as pool:
        results = list(
            tqdm(
                pool.imap(compute_logits_shard, args_list),
                total=len(args_list),
                desc="Processing Shards",
            )
        )

    for shard_result in results:
        for item in shard_result:
            all_logits_dict[item["hash_index"]] = item["logits"]

    all_logits = [
        all_logits_dict[hash_index] for hash_index in clinvar_df["hash_index"]
    ]

    missing_count = len([x for x in all_logits if x is None])
    if missing_count > 0:
        print(f"Warning: {missing_count} sequences missing logits")
        tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=True
        )
        vocab_size = len(tokenizer)
        for i in range(len(all_logits)):
            if all_logits[i] is None:
                all_logits[i] = [0.0] * vocab_size

    print(
        f"✅ Parallel logit computation completed in {time.time() - start_time:.2f} seconds"
    )
    return all_logits


def get_char_indices(vocab: Dict[str, int]) -> Dict[str, List[int]]:
    tokens = list(vocab.keys())
    token_ids = list(vocab.values())
    sorted_pairs = sorted(zip(token_ids, tokens))
    sorted_tokens = [token for _, token in sorted_pairs]

    char_indices = {}
    for i, token in enumerate(sorted_tokens):
        if isinstance(token, str) and len(token) > 0:
            first_char = token[0]
            if first_char not in char_indices:
                char_indices[first_char] = []
            char_indices[first_char].append(i)

    return char_indices


def compute_prob(
    args: Tuple[str, str, List[float], Dict[str, List[int]]],
) -> Tuple[float, float]:
    ref, alt, logits, char_indices = args
    p_ref = sum(logits[i] for i in char_indices.get(ref, []) if i < len(logits))
    p_alt = sum(logits[i] for i in char_indices.get(alt, []) if i < len(logits))
    return p_ref, p_alt


def parallel_compute_probabilities(
    clinvar_df: pd.DataFrame,
    logits: List[List[float]],
    tokenizer: PreTrainedTokenizer,
    num_processes: int = 16,
) -> Tuple[List[float], List[float]]:
    print(f"🧮 Computing variant probabilities with {num_processes} processes...")
    start_time = time.time()

    vocab = tokenizer.get_vocab()
    char_indices = get_char_indices(vocab)

    args_list = [
        (clinvar_df["ref"][i], clinvar_df["alt"][i], logits[i], char_indices)
        for i in range(len(clinvar_df))
    ]

    chunksize = max(1, len(args_list) // (num_processes * 4))
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(compute_prob, args_list, chunksize=chunksize),
                total=len(args_list),
                desc="Computing Probabilities",
            )
        )

    p_ref, p_alt = zip(*results)
    print(
        f"✅ Probability computation completed in {time.time() - start_time:.2f} seconds"
    )
    return list(p_ref), list(p_alt)


def compute_probabilities_evo2(clinvar_df: pd.DataFrame, model_name: str) -> Tuple[List[float], List[float]]:
    try:
        from evo2 import Evo2
    except Exception as e:
        raise RuntimeError("Evo2 library not available; install evo2 to use --use_evo2") from e

    torch.cuda.set_device(0)
    model = Evo2(model_name)

    base_ids = {}
    for b in ["A", "C", "G", "T"]:
        ids = model.tokenizer.tokenize(b)
        base_ids[b] = ids[0] if ids else None

    p_ref = []
    p_alt = []

    for i in tqdm(range(len(clinvar_df)), desc="Evo2 Probabilities"):
        seq = clinvar_df["sequence"][i]
        ref = clinvar_df["ref"][i]
        alt = clinvar_df["alt"][i]

        input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to("cuda:0")
        outputs, _ = model(input_ids)
        logits = outputs[0][0, -1, :]
        probs = torch.softmax(logits, dim=0)

        ref_id = base_ids.get(ref)
        alt_id = base_ids.get(alt)
        p_ref.append(float(probs[ref_id]) if ref_id is not None else 0.0)
        p_alt.append(float(probs[alt_id]) if alt_id is not None else 0.0)

    return p_ref, p_alt


def evaluate_predictions(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    print("📊 Evaluating model predictions...")
    start_time = time.time()

    auroc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = auc(recall, precision)

    print(f"⏱️ Evaluation completed in {time.time() - start_time:.2f} seconds")
    return {"AUROC": float(auroc), "AUPRC": float(auprc)}


def save_results(df: pd.DataFrame, path: str) -> None:
    print(f"💾 Saving predictions to {path}")
    start_time = time.time()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)

    print(f"✅ Results saved in {time.time() - start_time:.2f} seconds")
    print(f"📊 Saved {len(df)} variant predictions")


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


def main() -> None:
    args = parse_args()
    dtype = "bfloat16" if args.bf16 else "float32"

    print("\n" + "=" * 80)
    print("🧬  CLINVAR VEP EVAL  🧬")
    print("=" * 80 + "\n")
    print(f"Model: {args.model}")
    if args.revision:
        print(f"Revision: {args.revision}")

    clinvar_df = load_and_prepare_data(
        args.hg38_path, args.clinvar_path, args.context_length
    )

    if args.use_evo2:
        p_ref, p_alt = compute_probabilities_evo2(clinvar_df, args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=True
        )

        logits = compute_logits_parallel(
            clinvar_df,
            args.model,
            args.revision,
            dtype,
            batch_size=args.batch_size,
        )

        p_ref, p_alt = parallel_compute_probabilities(
            clinvar_df, logits, tokenizer, num_processes=args.num_processes
        )

    clinvar_df["p_ref"] = p_ref
    clinvar_df["p_alt"] = p_alt

    clinvar_df["label"] = clinvar_df["label"].astype(int)
    clinvar_df["score"] = np.log(clinvar_df["p_ref"] / (clinvar_df["p_alt"] + 1e-10))

    metrics = evaluate_predictions(
        clinvar_df["label"].values, clinvar_df["score"].values
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    revision_tag = args.revision or "main"
    output_basename = f"{model_name}_{revision_tag}_clinvar_{dtype}"

    output_path = os.path.join(args.output_dir, f"{output_basename}.parquet")
    save_results(clinvar_df.drop(columns=["sequence", "hash_index"]), output_path)

    summary = {
        "model": args.model,
        "revision": args.revision,
        "context_length": args.context_length,
        "num_variants": int(len(clinvar_df)),
        "dtype": dtype,
        **metrics,
    }
    summary_path = os.path.join(args.output_dir, f"{output_basename}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"🏆 AUROC: {metrics['AUROC']:.4f}")
    print(f"📈 AUPRC: {metrics['AUPRC']:.4f}")
    print("=" * 80)

    maybe_push_to_hub(args, output_path, summary_path)


if __name__ == "__main__":
    main()
