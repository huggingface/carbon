import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KEGG DNA-only classifier training/eval"
    )
    parser.add_argument(
        "--model", required=True, help="Model name or path (HF hub or local)"
    )
    parser.add_argument(
        "--revision", default=None, help="Optional model revision/tag/commit"
    )
    parser.add_argument(
        "--output_dir",
        default="./eval_results/kegg_dna_classifier",
        help="Output directory",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=5, help="Max epochs")
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Max steps (override epochs)"
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument(
        "--truncate_dna_per_side",
        type=int,
        default=1024,
        help="Trim this many bases from each side",
    )
    parser.add_argument(
        "--merge_val_test_set", action="store_true", help="Merge val+test for eval"
    )
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--eval_only", action="store_true", help="Skip training, only evaluate"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint path for eval_only"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Upload outputs to the Hub"
    )
    parser.add_argument("--hub_repo_id", default=None, help="HF repo to upload results")
    parser.add_argument(
        "--hub_repo_type",
        default="dataset",
        choices=["dataset", "model"],
        help="HF repo type",
    )
    return parser.parse_args()


def _truncate(seq: str, n: int) -> str:
    if n <= 0:
        return seq
    if len(seq) <= 2 * n:
        return seq
    return seq[n:-n]


def prepare_dataset(truncate_dna_per_side: int) -> Tuple[DatasetDict, Dict[str, int]]:
    ds = load_dataset("wanglab/kegg", "default")
    labels = sorted(list(set(ds["train"]["answer"])))
    label2id = {lbl: i for i, lbl in enumerate(labels)}

    def map_fn(ex):
        ref = _truncate(ex["reference_sequence"], truncate_dna_per_side)
        var = _truncate(ex["variant_sequence"], truncate_dna_per_side)
        seq = ref + var
        return {"text": seq, "label": label2id[ex["answer"]]}

    ds = ds.map(map_fn, remove_columns=ds["train"].column_names)
    return ds, label2id


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(
            k in name.lower() for k in ["classifier", "score", "classification", "head"]
        ):
            param.requires_grad = True


def main() -> None:
    args = parse_args()

    ds, label2id = prepare_dataset(args.truncate_dna_per_side)

    if args.merge_val_test_set:
        eval_val_ds = concatenate_datasets([ds["test"], ds["val"]])
    else:
        eval_val_ds = ds["val"]
    eval_test_ds = ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    train_ds = ds["train"].map(tokenize_fn, batched=True)
    eval_val_ds = eval_val_ds.map(tokenize_fn, batched=True)
    eval_test_ds = eval_test_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        num_labels=len(label2id),
        id2label={v: k for k, v in label2id.items()},
        label2id=label2id,
    )

    freeze_backbone(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.max_epochs,
        max_steps=args.max_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None if args.eval_only else train_ds,
        eval_dataset=eval_val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    if args.eval_only:
        if args.checkpoint:
            trainer.model = AutoModelForSequenceClassification.from_pretrained(
                args.checkpoint,
                trust_remote_code=True,
                num_labels=len(label2id),
                id2label={v: k for k, v in label2id.items()},
                label2id=label2id,
            )
    else:
        trainer.train()

    metrics_val = trainer.evaluate(eval_dataset=eval_val_ds, metric_key_prefix="val")
    metrics_test = trainer.evaluate(eval_dataset=eval_test_ds, metric_key_prefix="test")

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "kegg_dna_classifier_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({**metrics_val, **metrics_test}, f, indent=2)

    print(metrics_val)
    print(metrics_test)

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("--hub_repo_id is required when --push_to_hub is set")
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=summary_path,
            path_in_repo=os.path.basename(summary_path),
            repo_id=args.hub_repo_id,
            repo_type=args.hub_repo_type,
        )


if __name__ == "__main__":
    main()
