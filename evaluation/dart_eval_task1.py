"""
DART-Eval Task 1: Prioritizing Known Regulatory Elements (Zero-Shot Likelihood)

Self-contained script that uses the DART-Eval repo's dataset classes but defines
its own GENERator evaluator, so no modifications to the DART-Eval repo are needed.

Usage:
    python evaluation/dart_eval_task1.py \
        --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
        --dart_work_dir /fsx/kashif/dart_work \
        --dart_eval_dir /fsx/kashif/DART-Eval \
        --batch_size 512 \
        --bf16
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import wilcoxon
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DART-Eval Task 1 (Prioritizing Known Regulatory Elements)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model name or path (e.g., GenerTeam/GENERator-v2-eukaryote-1.2b-base)",
    )
    parser.add_argument(
        "--dart_eval_dir",
        default=None,
        help="Path to DART-Eval repo (default: ../DART-Eval)",
    )
    parser.add_argument(
        "--dart_work_dir",
        default=None,
        help="Work dir for DART-Eval data (overrides $DART_WORK_DIR)",
    )
    parser.add_argument(
        "--hub_dataset",
        default="hf-carbon/dart-eval-task1",
        help="HF Hub dataset with DART-Eval Task 1 data",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: $DART_WORK_DIR/task_1_ccre/zero_shot_outputs/likelihoods/$MODEL)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16",
    )
    parser.add_argument(
        "--chroms",
        nargs="+",
        default=["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"],
        help="Chromosomes to evaluate on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dinucleotide shuffle",
    )
    return parser.parse_args()


ALPHABET = np.array(["A", "C", "G", "T"], dtype="S1")


def onehot_to_chars(onehot):
    chararray = ALPHABET[np.argmax(onehot, axis=2)]
    return [b"".join(row).decode() for row in chararray]


def score_causal(model, tokenizer, seqs_onehot, starts, ends, device):
    """Score sequences using causal LM log-likelihood within the element region."""
    seqs_str = onehot_to_chars(seqs_onehot)
    encoded = tokenizer.batch_encode_plus(seqs_str, return_tensors="pt", padding=True)
    tokens = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Find start/end token positions
    # For causal LM with BOS: sequence tokens start after BOS
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model(tokens, attention_mask=attention_mask)
        logits = outputs.logits.swapaxes(1, 2)  # (batch, vocab, seq_len)
        # Causal: predict token t from position t-1
        lls = torch.zeros(tokens.shape[:2], device=device)
        lls[:, 1:] = -F.cross_entropy(
            logits[:, :, :-1], tokens[:, 1:], reduction="none"
        )

    # Clip to element region using token positions
    # Map bp-level starts/ends to token-level
    # The tokenizer maps the full sequence; we need to figure out which tokens
    # correspond to the element region.
    # For simplicity with variable-length tokenizers (6-mer), sum all token LLs
    # between BOS and EOS (the standard approach for causal zero-shot scoring)
    clip_mask = torch.zeros_like(lls)
    if bos_id is not None:
        tok_starts = torch.where(tokens == bos_id)[1] + 1
    else:
        tok_starts = torch.zeros(tokens.shape[0], dtype=torch.long, device=device)

    if eos_id is not None:
        eos_positions = torch.where(tokens == eos_id)
        tok_ends = torch.zeros(tokens.shape[0], dtype=torch.long, device=device)
        for idx, pos in zip(eos_positions[0], eos_positions[1]):
            tok_ends[idx] = pos
    else:
        tok_ends = attention_mask.sum(dim=1) if attention_mask is not None else torch.full(
            (tokens.shape[0],), tokens.shape[1], device=device
        )

    for i in range(lls.shape[1]):
        clip_mask[:, i] = ((i >= tok_starts) & (i < tok_ends)).float()

    out = (lls * clip_mask).sum(1).cpu().numpy()
    return out


def evaluate(model, tokenizer, dataloader, out_dir, device, progress_bar=True):
    """Run the paired-control zero-shot evaluation."""
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.tsv")
    metrics_path = os.path.join(out_dir, "metrics.json")

    with open(scores_path, "w") as f:
        f.write("idx\tseq_score\tctrl_score\n")

        diffs_lst = []
        corrects_lst = []

        for seqs, ctrls, inds in tqdm(dataloader, disable=(not progress_bar), ncols=120):
            seq_scores = score_causal(
                model, tokenizer, seqs, None, None, device
            )
            ctrl_scores = score_causal(
                model, tokenizer, ctrls, None, None, device
            )

            for ind, seq_score, ctrl_score in zip(inds, seq_scores, ctrl_scores):
                f.write(f"{ind}\t{seq_score}\t{ctrl_score}\n")
            f.flush()

            diff_batch = seq_scores - ctrl_scores
            correct_batch = diff_batch > 0

            diffs_lst.append(diff_batch)
            corrects_lst.append(correct_batch)

        diffs = np.concatenate(diffs_lst)
        corrects = np.concatenate(corrects_lst)

    metrics = {}
    metrics["acc"] = float(corrects.mean())

    wilcox = wilcoxon(diffs, alternative="greater")
    metrics["pval"] = float(wilcox.pvalue)
    metrics["signed_rank_sum"] = float(wilcox.statistic)
    metrics["mean_diff"] = float(diffs.mean())
    metrics["q05_diff"] = float(np.percentile(diffs, 5))
    metrics["q25_diff"] = float(np.percentile(diffs, 25))
    metrics["median_diff"] = float(np.median(diffs))
    metrics["q75_diff"] = float(np.percentile(diffs, 75))
    metrics["q95_diff"] = float(np.percentile(diffs, 95))

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def main():
    args = parse_args()

    # Setup DART-Eval repo on PYTHONPATH for PairedControlDataset
    script_dir = Path(__file__).resolve().parent
    dart_eval_dir = (
        Path(args.dart_eval_dir).resolve()
        if args.dart_eval_dir
        else (script_dir.parent.parent / "DART-Eval").resolve()
    )
    dart_src = dart_eval_dir / "src"
    if str(dart_src) not in sys.path:
        sys.path.insert(0, str(dart_src))

    from dnalm_bench.task_1_paired_control.components import PairedControlDataset

    # Resolve work dir — download from Hub if needed
    dart_work_dir = Path(
        args.dart_work_dir or os.environ.get("DART_WORK_DIR", "")
    ).resolve()

    genome_fa_path = dart_work_dir / "refs" / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    elements_tsv_path = dart_work_dir / "task_1_ccre" / "processed_inputs" / "ENCFF420VPZ_processed.tsv"

    if not genome_fa_path.exists() or not elements_tsv_path.exists():
        print(f"Local data not found, downloading from {args.hub_dataset} ...")
        from huggingface_hub import hf_hub_download

        dart_work_dir.mkdir(parents=True, exist_ok=True)

        for repo_file in [
            "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta",
            "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.fai",
            "task_1_ccre/processed_inputs/ENCFF420VPZ_processed.tsv",
        ]:
            local_path = dart_work_dir / repo_file
            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"  Downloading {repo_file} ...")
                downloaded = hf_hub_download(
                    repo_id=args.hub_dataset,
                    filename=repo_file,
                    repo_type="dataset",
                    local_dir=str(dart_work_dir),
                )
                print(f"  -> {downloaded}")

        genome_fa_path = dart_work_dir / "refs" / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
        elements_tsv_path = dart_work_dir / "task_1_ccre" / "processed_inputs" / "ENCFF420VPZ_processed.tsv"

    genome_fa = str(genome_fa_path)
    elements_tsv = str(elements_tsv_path)

    model_short = args.model.split("/")[-1]
    out_dir = args.output_dir or str(
        dart_work_dir / "task_1_ccre" / "zero_shot_outputs" / "likelihoods" / model_short
    )

    print("=" * 80)
    print("DART-Eval Task 1: Zero-Shot Likelihood")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"DART work dir: {dart_work_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Chroms: {args.chroms}")
    print(f"Batch size: {args.batch_size}")

    # Load dataset
    dataset = PairedControlDataset(genome_fa, elements_tsv, args.chroms, args.seed)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"Dataset: {len(dataset)} elements")

    # Load model
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="right"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()

    # Run evaluation
    metrics = evaluate(model, tokenizer, dataloader, out_dir, device)

    print("\n" + "=" * 80)
    print(f"Results for {model_short}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
