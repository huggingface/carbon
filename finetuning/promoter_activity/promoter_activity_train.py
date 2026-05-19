"""Fine-tune Carbon-500M on Random Promoter DREAM 2022 activity prediction."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed

DATASET_NAME = "HuggingFaceBio/random-promoter-dream-2022"
DATASET_CONFIG = "supervised"
MODEL_NAME = "HuggingFaceBio/Carbon-500M-remote"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--dataset_config", default=DATASET_CONFIG)
    parser.add_argument(
        "--output_dir",
        default="scratch/promoter_activity/carbon-500m-pearson-huber-200k",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument(
        "--lr_scheduler_kwargs",
        default="{}",
        help="Optional JSON kwargs passed to the Transformers LR scheduler.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
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
    parser.add_argument("--max_train_samples", type=int, default=200_000)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--map_num_proc", type=int, default=8)

    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help="Delta for the Huber term in 1 - Pearson r + Huber.",
    )
    parser.add_argument(
        "--huber_loss_weight",
        type=float,
        default=0.2,
        help="Weight for the Huber term in 1 - Pearson r + Huber.",
    )
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
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument(
        "--save_predictions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_model", action="store_true")
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
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    return pearson_corr(rankdata_average(x[mask]), rankdata_average(y[mask]))


def normalize_predictions(value: Any) -> np.ndarray:
    if isinstance(value, tuple):
        value = value[0]
    return np.asarray(value, dtype=np.float64).reshape(-1)


def compute_label_stats(train_dataset: Any) -> dict[str, float]:
    labels = np.asarray(train_dataset["activity"], dtype=np.float64)
    label_std = float(np.std(labels))
    if not math.isfinite(label_std) or label_std == 0.0:
        raise ValueError("Cannot standardize labels with zero or invalid std")
    return {"activity_mean": float(np.mean(labels)), "activity_std": label_std}


def standardize_values(values: Any, label_stats: dict[str, float]) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float64) - label_stats["activity_mean"]
    ) / label_stats["activity_std"]


def inverse_standardize(values: Any, label_stats: dict[str, float]) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float64) * label_stats["activity_std"]
        + label_stats["activity_mean"]
    )


def compute_regression_metrics(
    predictions_scaled: Any,
    labels_scaled: Any,
    label_stats: dict[str, float],
    prefix: str | None = None,
) -> dict[str, float]:
    predictions = inverse_standardize(
        normalize_predictions(predictions_scaled), label_stats
    )
    labels = inverse_standardize(normalize_predictions(labels_scaled), label_stats)
    pearson = pearson_corr(predictions, labels)
    spearman = spearman_corr(predictions, labels)
    diff = predictions - labels
    key_prefix = "" if prefix is None else f"{prefix}_"
    return {
        f"{key_prefix}pearson": pearson,
        f"{key_prefix}pearson_r2": (
            pearson * pearson if math.isfinite(pearson) else float("nan")
        ),
        f"{key_prefix}spearman": spearman,
        f"{key_prefix}mse": float(np.mean(np.square(diff))),
        f"{key_prefix}mae": float(np.mean(np.abs(diff))),
    }


def gather_loss_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a 1D tensor across ranks while preserving prediction gradients."""
    if not (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    ):
        return tensor.contiguous().view(-1)

    world_size = dist.get_world_size()
    tensor = tensor.contiguous().view(-1)
    local_size = torch.tensor([tensor.numel()], device=tensor.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes_int = [int(size.item()) for size in sizes]
    max_size = max(sizes_int)
    if tensor.numel() < max_size:
        tensor = F.pad(tensor, (0, max_size - tensor.numel()))
    gathered = dist_nn.all_gather(tensor)
    return torch.cat(
        [rank_tensor[:size] for rank_tensor, size in zip(gathered, sizes_int)], dim=0
    )


def pearson_huber_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    huber_delta: float,
    huber_loss_weight: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    predictions = predictions.float().view(-1)
    labels = labels.to(dtype=predictions.dtype).view(-1)
    predictions = gather_loss_tensor(predictions)
    labels = gather_loss_tensor(labels)
    predictions_centered = predictions - predictions.mean()
    labels_centered = labels - labels.mean()
    numerator = (predictions_centered * labels_centered).sum()
    pred_ss = predictions_centered.square().sum()
    label_ss = labels_centered.square().sum()
    pearson = numerator / torch.sqrt((pred_ss * label_ss).clamp_min(eps))
    huber = F.huber_loss(predictions, labels, delta=huber_delta)
    return (1.0 - pearson) + huber_loss_weight * huber


def select_limit(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def prepare_sequences(sequences: list[Any], dna_mode: str) -> list[str]:
    prepared = [str(sequence).strip().upper() for sequence in sequences]
    if dna_mode == "dna_tags":
        return [f"<dna>{sequence}</dna>" for sequence in prepared]
    return prepared


def label_values_from_batch(
    batch: dict[str, list[Any]],
    label_stats: dict[str, float],
) -> list[float]:
    return (
        standardize_values(batch["activity"], label_stats).astype(np.float32).tolist()
    )


def write_predictions_tsv(
    path: Path,
    raw_dataset: Any,
    labels_raw: np.ndarray,
    predictions_raw: np.ndarray,
) -> None:
    row_ids = (
        list(raw_dataset["row_id"])
        if "row_id" in raw_dataset.column_names
        else list(range(len(predictions_raw)))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row_id", "activity", "prediction"])
        for row_id, label, prediction in zip(row_ids, labels_raw, predictions_raw):
            writer.writerow([row_id, float(label), float(prediction)])


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


class PromoterActivityTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        huber_delta: float,
        huber_loss_weight: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.huber_delta = huber_delta
        self.huber_loss_weight = huber_loss_weight

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
        loss = pearson_huber_loss(
            logits,
            labels,
            huber_delta=self.huber_delta,
            huber_loss_weight=self.huber_loss_weight,
        )
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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset %s / %s", args.dataset_name, args.dataset_config)
    raw_dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_raw = select_limit(raw_dataset["train"], args.max_train_samples)
    validation_raw = select_limit(raw_dataset["validation"], args.max_eval_samples)
    test_raw = (
        None
        if args.skip_test
        else select_limit(raw_dataset["test"], args.max_eval_samples)
    )
    label_stats = compute_label_stats(train_raw)

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
            prepare_sequences(batch["sequence"], args.dna_mode),
            truncation=True,
            max_length=args.max_length,
            padding=False,
            **tokenizer_kwargs,
        )
        tokenized["labels"] = label_values_from_batch(batch, label_stats)
        return tokenized

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "remove_columns": train_raw.column_names,
        "desc": "Tokenizing Random Promoter DREAM",
    }
    if args.map_num_proc and args.map_num_proc > 1:
        map_kwargs["num_proc"] = args.map_num_proc

    with state.main_process_first():
        train_dataset = train_raw.map(tokenize, **map_kwargs)
        validation_dataset = validation_raw.map(tokenize, **map_kwargs)
        test_dataset = None
        if test_raw is not None:
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
        num_labels=1,
        problem_type="regression",
        id2label={0: "activity_scaled"},
        label2id={"activity_scaled": 0},
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
        "label_stats": label_stats,
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
        metric_for_best_model="pearson",
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

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        return compute_regression_metrics(
            eval_pred.predictions,
            eval_pred.label_ids,
            label_stats,
        )

    trainer = PromoterActivityTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        huber_delta=args.huber_delta,
        huber_loss_weight=args.huber_loss_weight,
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
        predictions_scaled = normalize_predictions(output.predictions)
        labels_scaled = normalize_predictions(output.label_ids)
        predictions_raw = inverse_standardize(predictions_scaled, label_stats)
        labels_raw = inverse_standardize(labels_scaled, label_stats)
        metrics = {
            **output.metrics,
            **compute_regression_metrics(
                predictions_scaled,
                labels_scaled,
                label_stats,
                prefix=split,
            ),
        }
        if trainer.is_world_process_zero():
            write_json(output_dir / f"{split}_metrics.json", metrics)
            if args.save_predictions:
                write_predictions_tsv(
                    output_dir / f"{split}_predictions.tsv",
                    raw_split,
                    labels_raw,
                    predictions_raw,
                )

    evaluate_split("validation", validation_dataset, validation_raw)
    if test_dataset is not None and test_raw is not None:
        evaluate_split("test", test_dataset, test_raw)


if __name__ == "__main__":
    main()
