"""
Finetune Carbon with supervised fine-tuning (SFT) using FNSTrainer.

This script performs autoregressive language modeling on DNA sequences using
the FNSTrainer, which applies base-pair level loss for DNA k-mer tokens and
token-level loss for BPE tokens.

For Carbon hybrid models, DNA sequences are wrapped with <dna>...</dna> tags
so the tokenizer routes them to 6-mer mode.

Example (single GPU):
  python finetune_sft.py \
      --model HuggingFaceBio/Carbon-3B \
      --dataset your/dataset \
      --output_dir ./outputs/sft-carbon-3B

Example (multi-GPU via torchrun):
  torchrun --nproc_per_node=8 finetune_sft.py \
      --model HuggingFaceBio/Carbon-3B \
      --dataset your/dataset \
      --batch_size 4 --grad_accum 4 \
      --output_dir ./outputs/sft-carbon-3B
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from fns_trainer import FNSTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF repo or local checkpoint")
    p.add_argument("--revision", default=None)
    p.add_argument("--dataset", required=True, help="HF dataset name or local path")
    p.add_argument("--dataset_config", default=None, help="Dataset config name")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=2048,
                   help="Maximum sequence length for training")
    p.add_argument("--add_dna_tag", action="store_true",
                   help="Wrap sequences with <dna> tags (for Carbon hybrid models)")
    p.add_argument("--dna_loss_only", action="store_true",
                   help="Only compute loss on DNA k-mer tokens (ignore BPE tokens)")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_split", default="test", help="Evaluation split name")
    p.add_argument("--sequence_column", default="sequence", help="Column name for sequences")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f"SFT with FNSTrainer · model={args.model} · dataset={args.dataset}")
    print("=" * 70)

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    # Prepare DNA tag prefix
    prefix = "<dna>" if args.add_dna_tag else ""
    suffix = "</dna>" if args.add_dna_tag else ""

    def tokenize(batch):
        """Tokenize sequences and create labels for causal LM."""
        seqs = [prefix + s + suffix for s in batch[args.sequence_column]]
        tokenized = tok(
            seqs,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Load dataset
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config)
    else:
        ds = load_dataset(args.dataset)

    # Tokenize dataset
    ds = ds.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in ds["train"].column_names if c != args.sequence_column]
    )

    print(f"  train: {len(ds['train']):,} examples")
    if args.eval_split in ds:
        print(f"  {args.eval_split}: {len(ds[args.eval_split]):,} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=50,
        eval_strategy="epoch" if args.eval_split in ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )

    # Data collator for causal LM with padding
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False,  # Causal LM, not masked LM
    )

    # Initialize FNSTrainer
    trainer = FNSTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get(args.eval_split),
        tokenizer=tok,
        data_collator=data_collator,
        dna_loss_only=args.dna_loss_only,
    )

    # Train
    trainer.train()

    # Evaluate
    if args.eval_split in ds:
        metrics = trainer.evaluate()
        print("\n" + "=" * 70)
        print(f"Final {args.eval_split} metrics:")
        for k, v in metrics.items():
            if k.startswith("eval_") and isinstance(v, float):
                print(f"  {k[5:]}: {v:.4f}")
        print("=" * 70)

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"\nModel saved to {os.path.join(args.output_dir, 'final')}")


if __name__ == "__main__":
    main()
