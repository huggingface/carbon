import argparse
import hashlib
import json
import os
import time

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CDS half-shuffle discrimination eval (post-training)"
    )
    parser.add_argument(
        "--dataset",
        default="hf-carbon/carbon_tasks",
        help="HF dataset name or local parquet path",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional HF dataset config/subset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to evaluate",
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
        default="./eval_results/cds_half_shuffle",
        help="Directory to save eval outputs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max token length for scoring (truncation applied)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for scoring",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16",
    )
    parser.add_argument(
        "--use_evo2",
        action="store_true",
        help="Use Evo2 scoring (official evo2 library) instead of HF AutoModel",
    )
    parser.add_argument(
        "--keep_sequences",
        action="store_true",
        help="Store raw sequences in outputs (default stores only hashes)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload outputs to the Hub",
    )
    parser.add_argument(
        "--hub_repo_id",
        default=None,
        help="HF repo to upload results (e.g., hf-carbon/cds-half-shuffle-results)",
    )
    parser.add_argument(
        "--hub_repo_type",
        default="dataset",
        choices=["dataset", "model"],
        help="HF repo type",
    )
    return parser.parse_args()


def _hash_seq(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]


def _load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if args.dataset.endswith(".parquet") or args.dataset.startswith("hf://"):
        return pd.read_parquet(args.dataset)

    ds = load_dataset(args.dataset, args.subset, split=args.split)
    return ds.to_pandas()


def _score_sequences_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sequences: list,
    max_length: int,
    batch_size: int,
) -> list:
    scores = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        with torch.no_grad():
            logits = model(input_ids).logits
            logits = logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
            target_mask = attention_mask[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            token_log_probs = token_log_probs * target_mask
            denom = target_mask.sum(dim=1).clamp(min=1)
            batch_scores = (token_log_probs.sum(dim=1) / denom).tolist()
            scores.extend(batch_scores)
    return scores


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


def _score_sequences_evo2(sequences: list, batch_size: int, model_name: str) -> list:
    try:
        from evo2 import Evo2
    except Exception as e:
        raise RuntimeError("Evo2 library not available; install evo2 to use --use_evo2") from e
    torch.cuda.set_device(0)
    _patch_evo2_config_no_flash(model_name)
    model = Evo2(model_name)
    return model.score_sequences(
        sequences,
        batch_size=batch_size,
        reduce_method="mean",
        average_reverse_complement=False,
    )


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
    print("🧬  CDS HALF-SHUFFLE EVAL  🧬")
    print("=" * 80 + "\n")
    print(f"Model: {args.model}")
    if args.revision:
        print(f"Revision: {args.revision}")
    print(f"Dataset: {args.dataset}")
    if args.subset:
        print(f"Subset: {args.subset}")
    print(f"Split: {args.split}")

    df = _load_dataset(args)
    pos_col = "original_sequence"
    neg_col = "input"

    pos_seqs = df[pos_col].astype(str).tolist()
    neg_seqs = df[neg_col].astype(str).tolist()

    if args.use_evo2:
        model_name = _evo2_model_name(args.model)
        pos_scores = _score_sequences_evo2(pos_seqs, args.batch_size, model_name)
        neg_scores = _score_sequences_evo2(neg_seqs, args.batch_size, model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            device_map="auto",
        )
        pos_scores = _score_sequences_hf(
            model, tokenizer, pos_seqs, args.max_length, args.batch_size
        )
        neg_scores = _score_sequences_hf(
            model, tokenizer, neg_seqs, args.max_length, args.batch_size
        )

    preds = [0 if p >= n else 1 for p, n in zip(pos_scores, neg_scores)]
    labels = [0] * len(preds)

    correct = [int(p == label) for p, label in zip(preds, labels)]
    accuracy = sum(correct) / max(len(correct), 1)

    results = {
        "pos_score": pos_scores,
        "neg_score": neg_scores,
        "pred": preds,
        "label": labels,
        "correct": correct,
    }
    if args.keep_sequences:
        results["pos_seq"] = pos_seqs
        results["neg_seq"] = neg_seqs
    else:
        results["pos_hash"] = [_hash_seq(s) for s in pos_seqs]
        results["neg_hash"] = [_hash_seq(s) for s in neg_seqs]
        results["pos_len"] = [len(s) for s in pos_seqs]
        results["neg_len"] = [len(s) for s in neg_seqs]

    output_df = pd.DataFrame(results)

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    revision_tag = args.revision or "main"
    output_basename = f"{model_name}_{revision_tag}_cds_half_shuffle_{dtype}"
    output_path = os.path.join(args.output_dir, f"{output_basename}.parquet")
    output_df.to_parquet(output_path)

    summary = {
        "model": args.model,
        "revision": args.revision,
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "pos_col": pos_col,
        "neg_col": neg_col,
        "accuracy": accuracy,
        "num_examples": len(output_df),
        "timestamp": time.time(),
    }
    summary_path = os.path.join(args.output_dir, f"{output_basename}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    maybe_push_to_hub(args, output_path, summary_path)

    print(f"✅ Results saved to {output_path}")
    print(f"📊 Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
