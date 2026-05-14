"""Minimal DeepSTARR regression fine-tuning script for Carbon.

The defaults reproduce the strongest Carbon 3B regression setup we found:
full fine-tuning, sequence-classification regression head, Pearson loss,
dataset-scaled Dev/Hk labels, auto DNA tags, no weight decay, and validation
by mean PCC.
"""

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
    parser = argparse.ArgumentParser(
        description="Carbon DeepSTARR regression fine-tuning"
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument(
        "--output_dir",
        default="scratch/deepstarr/carbon-3b-regression-train",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--num_train_epochs", type=float, default=2.5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", default="reduce_lr_on_plateau")
    parser.add_argument(
        "--lr_scheduler_kwargs",
        default='{"mode":"max","factor":0.5,"patience":2,"threshold":0.0001}',
        help="JSON kwargs passed to the Transformers LR scheduler.",
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
        help="Use auto_dna_tags=True, explicit <dna>...</dna> wrapping, or raw DNA.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
        "--trust_remote_code", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--save_final_model", action="store_true")
    args = parser.parse_args()

    try:
        args.lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)
    except json.JSONDecodeError as exc:
        parser.error(f"--lr_scheduler_kwargs must be valid JSON: {exc}")
    if not isinstance(args.lr_scheduler_kwargs, dict):
        parser.error("--lr_scheduler_kwargs must decode to a JSON object")
    if args.fp16 and args.bf16:
        parser.error("Use at most one of --fp16 and --bf16")
    return args


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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pearson_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    predictions = predictions.float()
    labels = labels.to(dtype=predictions.dtype)
    predictions = predictions - predictions.mean(dim=0, keepdim=True)
    labels = labels - labels.mean(dim=0, keepdim=True)
    numerator = (predictions * labels).sum(dim=0)
    pred_ss = predictions.square().sum(dim=0)
    label_ss = labels.square().sum(dim=0)
    corr = numerator / torch.sqrt((pred_ss * label_ss).clamp_min(eps))
    return 1.0 - corr.mean()


def normalize_predictions(predictions: Any) -> np.ndarray:
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    return predictions


def normalize_label_ids(label_ids: Any) -> np.ndarray:
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]
    labels = np.asarray(label_ids, dtype=np.float64)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    return labels


def label_values_from_batch(batch: dict[str, list[Any]]) -> list[list[float]]:
    columns = [np.asarray(batch[column], dtype=np.float64) for column in SCALED_COLUMNS]
    return np.stack(columns, axis=1).astype(np.float32).tolist()


def fit_linear_calibration_from_arrays(
    scaled: np.ndarray, raw_log: np.ndarray
) -> tuple[float, float]:
    scaled = np.asarray(scaled, dtype=np.float64)
    raw_log = np.asarray(raw_log, dtype=np.float64)
    mask = np.isfinite(scaled) & np.isfinite(raw_log)
    if int(mask.sum()) < 2:
        raise ValueError("Need at least two finite examples for calibration")
    scaled = scaled[mask]
    raw_log = raw_log[mask]
    variance = float(np.var(scaled))
    if variance == 0.0:
        raise ValueError("Cannot calibrate with zero-variance labels")
    slope = float(np.cov(scaled, raw_log, bias=True)[0, 1] / variance)
    intercept = float(np.mean(raw_log) - slope * np.mean(scaled))
    return slope, intercept


def fit_log_calibration(train_dataset: Any) -> dict[str, dict[str, float]]:
    calibration = {}
    for target, scaled_col, raw_col in zip(
        TARGET_NAMES, SCALED_COLUMNS, RAW_LOG_COLUMNS
    ):
        slope, intercept = fit_linear_calibration_from_arrays(
            np.asarray(train_dataset[scaled_col], dtype=np.float64),
            np.asarray(train_dataset[raw_col], dtype=np.float64),
        )
        calibration[target] = {"slope": slope, "intercept": intercept}
    return calibration


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    predictions = normalize_predictions(eval_pred.predictions)
    labels = normalize_label_ids(eval_pred.label_ids)
    metrics = {}
    pcc_values = []
    for idx, target in enumerate(TARGET_NAMES):
        pcc = pearson_corr(predictions[:, idx], labels[:, idx])
        metrics[f"pcc_{target}_scaled"] = pcc
        pcc_values.append(pcc)
    metrics["pcc_mean"] = float(np.nanmean(pcc_values))
    return metrics


def compute_full_metrics(
    *,
    prefix: str,
    predictions: np.ndarray,
    labels_scaled: np.ndarray,
    raw_dataset: Any,
    calibration: dict[str, dict[str, float]],
) -> dict[str, float]:
    predictions = normalize_predictions(predictions)
    labels_scaled = np.asarray(labels_scaled, dtype=np.float64)
    metrics = {}
    scaled_pcc_values = []
    log_pcc_values = []
    for idx, target in enumerate(TARGET_NAMES):
        scaled_pcc = pearson_corr(predictions[:, idx], labels_scaled[:, idx])
        metrics[f"{prefix}_pcc_{target}_scaled"] = scaled_pcc
        scaled_pcc_values.append(scaled_pcc)

        cal = calibration[target]
        pred_log = cal["slope"] * predictions[:, idx] + cal["intercept"]
        raw_labels = np.asarray(raw_dataset[RAW_LOG_COLUMNS[idx]], dtype=np.float64)
        log_pcc = pearson_corr(pred_log, raw_labels)
        metrics[f"{prefix}_log_pcc_{target}"] = log_pcc
        log_pcc_values.append(log_pcc)

    metrics[f"{prefix}_pcc_mean"] = float(np.nanmean(scaled_pcc_values))
    metrics[f"{prefix}_log_pcc_mean"] = float(np.nanmean(log_pcc_values))
    return metrics


def select_limit(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def prepare_sequences(sequences: list[Any], dna_mode: str) -> list[str]:
    prepared = [str(sequence).strip() for sequence in sequences]
    if dna_mode == "dna_tags":
        prepared = [
            (
                sequence
                if sequence.startswith("<dna>") and sequence.endswith("</dna>")
                else f"<dna>{sequence}</dna>"
            )
            for sequence in prepared
        ]
    return prepared


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    from accelerate import PartialState
    from datasets import load_dataset

    state = PartialState()
    logger.info("Loading dataset %s", args.dataset_name)
    raw_ds = load_dataset(args.dataset_name)
    train_raw = select_limit(raw_ds["train"], args.max_train_samples)
    validation_raw = select_limit(raw_ds["validation"], args.max_eval_samples)
    test_raw = select_limit(raw_ds["test"], args.max_eval_samples)
    calibration = fit_log_calibration(train_raw)

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

    def tokenize_fn(batch: dict[str, list[Any]]) -> dict[str, Any]:
        sequences = prepare_sequences(batch["sequence"], args.dna_mode)
        tokenizer_kwargs = {}
        if args.dna_mode == "auto_dna_tags":
            tokenizer_kwargs["auto_dna_tags"] = True
        tokenized = tokenizer(
            sequences,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            **tokenizer_kwargs,
        )
        tokenized["labels"] = label_values_from_batch(batch)
        return tokenized

    map_kwargs: dict[str, Any] = {"batched": True, "desc": "Tokenizing DeepSTARR"}
    if args.map_num_proc and args.map_num_proc > 1:
        map_kwargs["num_proc"] = args.map_num_proc

    with state.main_process_first():
        train_ds = train_raw.map(
            tokenize_fn,
            remove_columns=train_raw.column_names,
            **map_kwargs,
        )
        validation_ds = validation_raw.map(
            tokenize_fn,
            remove_columns=validation_raw.column_names,
            **map_kwargs,
        )
        test_ds = test_raw.map(
            tokenize_fn,
            remove_columns=test_raw.column_names,
            **map_kwargs,
        )

    logger.info("Loading model from %s", args.model)
    model_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype_from_arg(args.torch_dtype),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        problem_type="regression",
        id2label={0: "Dev_scaled", 1: "Hk_scaled"},
        label2id={"Dev_scaled": 0, "Hk_scaled": 1},
        **model_kwargs,
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
    steps_per_epoch = math.ceil(len(train_ds) / global_batch_size)
    eval_steps = args.eval_steps or math.ceil(steps_per_epoch / 10)

    run_config = {
        **vars(args),
        "effective_eval_steps": eval_steps,
        "global_train_batch_size": global_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "calibration": calibration,
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
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if args.save_final_model:
        trainer.save_model(str(output_dir / "best_model"))

    run_config["trainer_best_model_checkpoint"] = trainer.state.best_model_checkpoint
    if state.is_main_process:
        write_json(output_dir / "run_config.json", run_config)

    def write_split_results(
        split: str, tokenized_dataset: Any, raw_dataset: Any
    ) -> None:
        prediction_output = trainer.predict(
            tokenized_dataset,
            metric_key_prefix=split,
        )
        predictions = normalize_predictions(prediction_output.predictions)
        labels_scaled = normalize_label_ids(prediction_output.label_ids)
        metrics = {
            **run_config,
            **prediction_output.metrics,
            **compute_full_metrics(
                prefix=split,
                predictions=predictions,
                labels_scaled=labels_scaled,
                raw_dataset=raw_dataset,
                calibration=calibration,
            ),
        }
        if trainer.is_world_process_zero():
            write_json(output_dir / f"{split}_metrics.json", metrics)

    write_split_results("validation", validation_ds, validation_raw)
    if not args.skip_test:
        write_split_results("test", test_ds, test_raw)


if __name__ == "__main__":
    main()
