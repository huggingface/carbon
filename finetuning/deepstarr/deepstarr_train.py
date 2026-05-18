"""Fine-tune Carbon on DeepSTARR regression with Hugging Face Trainer."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed

DATASET_NAME = "GenerTeam/DeepSTARR-enhancer-activity"
MODEL_NAME = "HuggingFaceBio/Carbon-3B"
TARGET_NAMES = ("dev", "hk")
SCALED_COLUMNS = ("Dev_log2_enrichment_scaled", "Hk_log2_enrichment_scaled")
RAW_LOG_COLUMNS = ("Dev_log2_enrichment", "Hk_log2_enrichment")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument(
        "--output_dir",
        default="scratch/deepstarr/carbon-3b-regression-train",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_train_epochs", type=float, default=2.5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", default="reduce_lr_on_plateau")
    parser.add_argument(
        "--lr_scheduler_kwargs",
        default=None,
        help="Optional JSON kwargs passed to the Transformers LR scheduler.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--dna_mode",
        choices=["auto_dna_tags", "dna_tags", "plain"],
        default="auto_dna_tags",
        help="How raw DNA strings are passed to the Carbon tokenizer.",
    )
    parser.add_argument(
        "--truncate_to_multiple_of",
        type=int,
        default=6,
        help="Trim raw DNA sequence lengths to this multiple before tokenization.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--map_num_proc", type=int, default=8)

    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--attn_implementation",
        default="kernels-community/flash-attn3",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    if args.fp16 and args.bf16:
        parser.error("Use at most one of --fp16 and --bf16")
    if args.lr_scheduler_kwargs is None:
        args.lr_scheduler_kwargs = default_scheduler_kwargs(args.lr_scheduler_type)
    else:
        try:
            args.lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)
        except json.JSONDecodeError as exc:
            parser.error(f"--lr_scheduler_kwargs must be valid JSON: {exc}")
        if not isinstance(args.lr_scheduler_kwargs, dict):
            parser.error("--lr_scheduler_kwargs must decode to a JSON object")
    return args


def default_scheduler_kwargs(scheduler_type: str) -> dict[str, Any]:
    if scheduler_type != "reduce_lr_on_plateau":
        return {}
    return {"mode": "max", "factor": 0.5, "patience": 2, "threshold": 0.0001}


def torch_dtype_from_arg(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def pearson_corr(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mask = np.isfinite(predictions) & np.isfinite(labels)
    if int(mask.sum()) < 2:
        return float("nan")
    predictions = predictions[mask]
    labels = labels[mask]
    if float(np.std(predictions)) == 0.0 or float(np.std(labels)) == 0.0:
        return float("nan")
    return float(np.corrcoef(predictions, labels)[0, 1])


def pearson_loss(
    logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    logits = logits.float()
    labels = labels.to(dtype=logits.dtype)
    logits = logits - logits.mean(dim=0, keepdim=True)
    labels = labels - labels.mean(dim=0, keepdim=True)
    numerator = (logits * labels).sum(dim=0)
    denominator = torch.sqrt(
        (logits.square().sum(dim=0) * labels.square().sum(dim=0)).clamp_min(eps)
    )
    return 1.0 - (numerator / denominator).mean()


def as_2d_array(value: Any) -> np.ndarray:
    if isinstance(value, tuple):
        value = value[0]
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def label_values(batch: dict[str, list[Any]]) -> list[list[float]]:
    labels = [np.asarray(batch[column], dtype=np.float32) for column in SCALED_COLUMNS]
    return np.stack(labels, axis=1).tolist()


def trim_to_multiple(sequence: str, multiple: int) -> str:
    if multiple <= 0:
        return sequence
    usable_length = len(sequence) - (len(sequence) % multiple)
    return sequence[:usable_length]


def prepare_sequences(sequences: list[Any], dna_mode: str, multiple: int) -> list[str]:
    prepared = [
        trim_to_multiple(str(sequence).strip(), multiple) for sequence in sequences
    ]
    if dna_mode == "dna_tags":
        return [f"<dna>{sequence}</dna>" for sequence in prepared]
    return prepared


def select_limit(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    predictions = as_2d_array(eval_pred.predictions)
    labels = as_2d_array(eval_pred.label_ids)
    metrics = {}
    pcc_values = []
    for idx, target in enumerate(TARGET_NAMES):
        pcc = pearson_corr(predictions[:, idx], labels[:, idx])
        metrics[f"pcc_{target}"] = pcc
        pcc_values.append(pcc)
    metrics["pcc_mean"] = float(np.nanmean(pcc_values))
    return metrics


def compute_log_pcc_metrics(
    prefix: str,
    predictions: np.ndarray,
    raw_dataset: Any,
) -> dict[str, float]:
    metrics = {}
    pcc_values = []
    for idx, target in enumerate(TARGET_NAMES):
        labels = np.asarray(raw_dataset[RAW_LOG_COLUMNS[idx]], dtype=np.float64)
        pcc = pearson_corr(predictions[:, idx], labels)
        metrics[f"{prefix}_log_pcc_{target}"] = pcc
        pcc_values.append(pcc)
    metrics[f"{prefix}_log_pcc_mean"] = float(np.nanmean(pcc_values))
    return metrics


def initialize_missing_regression_head(
    model: nn.Module, missing_keys: list[str]
) -> None:
    classifier = getattr(model, "score", None)
    if not isinstance(classifier, nn.Linear):
        return
    if "score.weight" not in missing_keys and "score.bias" not in missing_keys:
        return

    initializer_range = float(getattr(model.config, "initializer_range", 0.02))
    with torch.no_grad():
        weight = torch.empty(
            classifier.weight.shape,
            dtype=torch.float32,
            device=classifier.weight.device,
        )
        nn.init.normal_(weight, mean=0.0, std=initializer_range)
        classifier.weight.copy_(weight.to(dtype=classifier.weight.dtype))
        if classifier.bias is not None:
            classifier.bias.zero_()

    if not torch.isfinite(classifier.weight.float()).all():
        raise RuntimeError("Regression head initialization produced non-finite values")
    logger.info("Initialized missing regression head in fp32")


class PearsonRegressionTrainer(Trainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss = pearson_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)

    from accelerate import PartialState
    from datasets import load_dataset

    state = PartialState()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset %s", args.dataset_name)
    raw_dataset = load_dataset(args.dataset_name)
    train_raw = select_limit(raw_dataset["train"], args.max_train_samples)
    validation_raw = select_limit(raw_dataset["validation"], args.max_eval_samples)
    test_raw = select_limit(raw_dataset["test"], args.max_eval_samples)

    logger.info("Loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    added_pad_token = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            added_pad_token = True
        else:
            tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenizer_kwargs = {}
        if args.dna_mode == "auto_dna_tags":
            tokenizer_kwargs["auto_dna_tags"] = True
        tokenized = tokenizer(
            prepare_sequences(
                batch["sequence"],
                dna_mode=args.dna_mode,
                multiple=args.truncate_to_multiple_of,
            ),
            truncation=True,
            max_length=args.max_length,
            padding=False,
            **tokenizer_kwargs,
        )
        tokenized["labels"] = label_values(batch)
        return tokenized

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "remove_columns": train_raw.column_names,
        "desc": "Tokenizing DeepSTARR",
    }
    if args.map_num_proc and args.map_num_proc > 1:
        map_kwargs["num_proc"] = args.map_num_proc

    with state.main_process_first():
        train_dataset = train_raw.map(tokenize, **map_kwargs)
        validation_dataset = validation_raw.map(tokenize, **map_kwargs)
        test_dataset = test_raw.map(tokenize, **map_kwargs)

    logger.info("Loading model from %s", args.model)
    model_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype_from_arg(args.torch_dtype),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(TARGET_NAMES),
        problem_type="regression",
        id2label={idx: f"{target}_scaled" for idx, target in enumerate(TARGET_NAMES)},
        label2id={f"{target}_scaled": idx for idx, target in enumerate(TARGET_NAMES)},
        output_loading_info=True,
        **model_kwargs,
    )
    initialize_missing_regression_head(
        model, list(loading_info.get("missing_keys", []))
    )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if added_pad_token and len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.config.use_cache = False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * max(1, world_size)
    )
    steps_per_epoch = math.ceil(len(train_dataset) / global_batch_size)
    eval_steps = args.eval_steps or max(1, math.ceil(steps_per_epoch / 10))

    run_config = {
        **vars(args),
        "effective_eval_steps": eval_steps,
        "global_train_batch_size": global_batch_size,
        "steps_per_epoch": steps_per_epoch,
    }
    if state.is_main_process:
        write_json(output_dir / "run_config.json", run_config)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="pcc_mean",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        run_name=args.run_name,
        remove_unused_columns=True,
        label_names=["labels"],
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
    )

    trainer = PearsonRegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if args.save_model:
        trainer.save_model(str(output_dir / "best_model"))

    run_config["trainer_best_model_checkpoint"] = trainer.state.best_model_checkpoint
    if state.is_main_process:
        write_json(output_dir / "run_config.json", run_config)

    def evaluate_split(split: str, tokenized_dataset: Any, raw_split: Any) -> None:
        output = trainer.predict(tokenized_dataset, metric_key_prefix=split)
        predictions = as_2d_array(output.predictions)
        metrics = {
            **output.metrics,
            **compute_log_pcc_metrics(split, predictions, raw_split),
        }
        if trainer.is_world_process_zero():
            write_json(output_dir / f"{split}_metrics.json", metrics)

    evaluate_split("validation", validation_dataset, validation_raw)
    if not args.skip_test:
        evaluate_split("test", test_dataset, test_raw)


if __name__ == "__main__":
    main()
