"""
Finetune Carbon for promoter detection.

Binary classification on the Nucleotide Transformer downstream `promoter_all`
task (300 bp human promoter sequences, balanced positives/negatives). This is
the same task the Evo2 and GENERator papers report on; we ship it here as a
minimal, end-to-end finetuning example.

Dataset: InstaDeepAI/nucleotide_transformer_downstream_tasks  config=promoter_all
  - train: 53,276 sequences
  - test:  5,919 sequences
  - sequence: 300 bp;  label: 0 (not promoter) / 1 (promoter)

Architecture: `AutoModelForSequenceClassification` with the Carbon backbone.
The classification head is a single linear layer on top of the pooled hidden
states — Transformers initialises it automatically.

For Carbon hybrid models we wrap each DNA sequence with `<dna>` so the
tokenizer routes it to 6-mer mode. See ../evaluation/README.md for the DNA-tag
explanation.

Example (single GPU):
  python finetune_promoter.py \
      --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
      --add_dna_tag \
      --output_dir ./outputs/promoter-carbon-3B

Example (multi-GPU via torchrun):
  torchrun --nproc_per_node=8 finetune_promoter.py \
      --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
      --add_dna_tag --batch_size 4 --grad_accum 4 \
      --output_dir ./outputs/promoter-carbon-3B
"""

import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF repo or local checkpoint")
    p.add_argument("--revision", default=None)
    p.add_argument("--dataset", default="InstaDeepAI/nucleotide_transformer_downstream_tasks")
    p.add_argument("--config", default="promoter_all")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=320,
                   help="Promoter sequences are 300 bp; 320 leaves room for the <dna> tag.")
    p.add_argument("--add_dna_tag", action="store_true",
                   help="Wrap with <dna> (Carbon hybrid models). Off for pure-DNA models.")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "mcc": matthews_corrcoef(labels, preds),
        "auroc": roc_auc_score(labels, probs),
    }


def main():
    args = parse_args()

    print("=" * 70)
    print(f"Promoter finetune · model={args.model} · dataset={args.dataset}/{args.config}")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        num_labels=2,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    prefix = "<dna>" if args.add_dna_tag else ""

    def tokenize(batch):
        seqs = [prefix + s for s in batch["sequence"]]
        return tok(seqs, truncation=True, max_length=args.max_length, add_special_tokens=False)

    ds = load_dataset(args.dataset, args.config)
    ds = ds.map(tokenize, batched=True, remove_columns=[c for c in ds["train"].column_names
                                                         if c not in {"label"}])

    print(f"  train: {len(ds['train']):,} examples")
    print(f"  test:  {len(ds['test']):,} examples")

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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    print("\n" + "=" * 70)
    print("Final test metrics:")
    for k, v in metrics.items():
        if k.startswith("eval_") and isinstance(v, float):
            print(f"  {k[5:]}: {v:.4f}")
    print("=" * 70)

    trainer.save_model(os.path.join(args.output_dir, "best"))


if __name__ == "__main__":
    main()
