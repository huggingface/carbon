"""Fine-tune Carbon on the Malinois MPRA regression task."""

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

SOURCE_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41586-024-08070-z/"
    "MediaObjects/41586_2024_8070_MOESM4_ESM.txt"
)
MODEL_NAME = "HuggingFaceBio/Carbon-3B"
TARGET_NAMES = ("K562", "HepG2", "SKNSH")
TARGET_COLUMNS = ("K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC")
SE_COLUMNS = ("K562_lfcSE", "HepG2_lfcSE", "SKNSH_lfcSE")
VALIDATION_CHROMS = ("19", "21", "X")
TEST_CHROMS = ("7", "13")
DNA_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--source_url", default=SOURCE_URL)
    parser.add_argument("--output_dir", default="scratch/malinois/carbon-3b-mse")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", default="reduce_lr_on_plateau")
    parser.add_argument(
        "--lr_scheduler_kwargs",
        default=None,
        help="Optional JSON kwargs passed to the Transformers LR scheduler.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--dna_mode",
        choices=["auto_dna_tags", "dna_tags", "plain"],
        default="auto_dna_tags",
        help="How raw DNA strings are passed to the Carbon tokenizer.",
    )
    parser.add_argument("--metric_se_threshold", type=float, default=1.0)
    parser.add_argument(
        "--train_reverse_complement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--duplicate_high_activity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--high_activity_threshold", type=float, default=0.5)
    parser.add_argument(
        "--rc_eval_average",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--map_num_proc", type=int, default=8)

    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--attn_implementation", default="kernels-community/flash-attn3"
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


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_corr(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mask = np.isfinite(predictions) & np.isfinite(labels)
    if int(mask.sum()) < 2:
        return float("nan")
    return pearson_corr(rankdata(predictions[mask]), rankdata(labels[mask]))


def as_2d_array(value: Any) -> np.ndarray:
    if isinstance(value, tuple):
        value = value[0]
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def normalize_chromosome(value: Any) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return text.upper()


def reverse_complement(sequence: Any) -> str:
    return str(sequence).strip().translate(DNA_COMPLEMENT)[::-1]


def finite_rows(batch: dict[str, list[Any]], columns: tuple[str, ...]) -> list[bool]:
    values = [np.asarray(batch[column], dtype=np.float64) for column in columns]
    finite = np.ones(len(values[0]), dtype=bool)
    for value in values:
        finite &= np.isfinite(value)
    finite &= np.asarray(
        [bool(str(sequence).strip()) for sequence in batch["sequence"]]
    )
    return finite.tolist()


def load_malinois_dataset(args: argparse.Namespace) -> Any:
    from datasets import Features
    from datasets import Value
    from datasets import load_dataset

    features = Features(
        {
            "IDs": Value("string"),
            "chr": Value("string"),
            "data_project": Value("string"),
            "OL": Value("string"),
            "class": Value("string"),
            TARGET_COLUMNS[0]: Value("float64"),
            TARGET_COLUMNS[1]: Value("float64"),
            TARGET_COLUMNS[2]: Value("float64"),
            SE_COLUMNS[0]: Value("float64"),
            SE_COLUMNS[1]: Value("float64"),
            SE_COLUMNS[2]: Value("float64"),
            "sequence": Value("string"),
        }
    )
    raw = load_dataset(
        "csv",
        data_files=args.source_url,
        delimiter="\t",
        split="train",
        features=features,
    )
    required = {"IDs", "chr", "sequence", *TARGET_COLUMNS, *SE_COLUMNS}
    missing = sorted(required.difference(raw.column_names))
    if missing:
        raise ValueError(f"Missing required columns in Malinois table: {missing}")
    return raw.filter(
        lambda batch: finite_rows(batch, (*TARGET_COLUMNS, *SE_COLUMNS)),
        batched=True,
        desc="Filtering finite Malinois rows",
    )


def split_by_chromosome(dataset: Any) -> dict[str, Any]:
    validation = {normalize_chromosome(chrom) for chrom in VALIDATION_CHROMS}
    test = {normalize_chromosome(chrom) for chrom in TEST_CHROMS}

    def split_name(chrom: Any) -> str:
        chrom = normalize_chromosome(chrom)
        if chrom in validation:
            return "validation"
        if chrom in test:
            return "test"
        return "train"

    return {
        split: dataset.filter(
            lambda row, split=split: split_name(row["chr"]) == split,
            desc=f"Selecting Malinois {split} split",
        )
        for split in ("train", "validation", "test")
    }


def select_limit(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def filter_metric_rows(dataset: Any, threshold: float) -> Any:
    def keep(batch: dict[str, list[Any]]) -> list[bool]:
        values = [np.asarray(batch[column], dtype=np.float64) for column in SE_COLUMNS]
        mask = np.ones(len(values[0]), dtype=bool)
        for value in values:
            mask &= np.isfinite(value) & (value <= threshold)
        return mask.tolist()

    return dataset.filter(
        keep,
        batched=True,
        desc=f"Filtering eval rows with all lfcSE <= {threshold:g}",
    )


def fit_label_transform(train_raw: Any) -> dict[str, dict[str, float]]:
    transform = {}
    for target, column in zip(TARGET_NAMES, TARGET_COLUMNS):
        values = np.asarray(train_raw[column], dtype=np.float64)
        center = float(np.mean(values))
        scale = float(np.std(values))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        transform[target] = {"column": column, "center": center, "scale": scale}
    return transform


def label_values(
    batch: dict[str, list[Any]],
    label_transform: dict[str, dict[str, float]],
) -> list[list[float]]:
    labels = []
    for target in TARGET_NAMES:
        spec = label_transform[target]
        values = np.asarray(batch[spec["column"]], dtype=np.float32)
        labels.append((values - spec["center"]) / spec["scale"])
    return np.stack(labels, axis=1).astype(np.float32).tolist()


def labels_from_dataset(
    dataset: Any,
    label_transform: dict[str, dict[str, float]],
) -> np.ndarray:
    return np.asarray(label_values(dataset[:], label_transform), dtype=np.float64)


def augment_train_dataset(train_raw: Any, args: argparse.Namespace) -> Any:
    if not args.train_reverse_complement and not args.duplicate_high_activity:
        return train_raw

    columns = list(train_raw.column_names)

    def augment(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output = {column: [] for column in columns}
        target_arrays = [
            np.asarray(batch[column], dtype=np.float64) for column in TARGET_COLUMNS
        ]
        high_activity = np.zeros(len(target_arrays[0]), dtype=bool)
        for values in target_arrays:
            high_activity |= values > args.high_activity_threshold

        for row_idx, sequence in enumerate(batch["sequence"]):
            sequence = str(sequence).strip().upper()
            views = [sequence]
            if args.train_reverse_complement:
                views.append(reverse_complement(sequence))
            copies = 2 if args.duplicate_high_activity and high_activity[row_idx] else 1
            for _ in range(copies):
                for view in views:
                    for column in columns:
                        output[column].append(
                            view if column == "sequence" else batch[column][row_idx]
                        )
        return output

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "desc": "Augmenting Malinois train split",
        "load_from_cache_file": False,
    }
    if args.map_num_proc and args.map_num_proc > 1:
        map_kwargs["num_proc"] = args.map_num_proc
    return train_raw.map(augment, **map_kwargs)


def prepare_sequences(sequences: list[Any], dna_mode: str) -> list[str]:
    prepared = [str(sequence).strip().upper() for sequence in sequences]
    if dna_mode == "dna_tags":
        return [
            (
                sequence
                if sequence.startswith("<dna>") and sequence.endswith("</dna>")
                else f"<dna>{sequence}</dna>"
            )
            for sequence in prepared
        ]
    return prepared


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    predictions = as_2d_array(eval_pred.predictions)
    labels = as_2d_array(eval_pred.label_ids)
    metrics = {}
    pcc_values = []
    for idx, target in enumerate(TARGET_NAMES):
        pcc = pearson_corr(predictions[:, idx], labels[:, idx])
        metrics[f"pcc_{target}_scaled"] = pcc
        pcc_values.append(pcc)
    metrics["pcc_mean"] = float(np.nanmean(pcc_values))
    return metrics


def full_metrics(
    prefix: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    raw_dataset: Any,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {f"{prefix}_num_examples": int(len(raw_dataset))}
    pcc_values = []
    spearman_values = []
    for idx, target in enumerate(TARGET_NAMES):
        pcc = pearson_corr(predictions[:, idx], labels[:, idx])
        spearman = spearman_corr(predictions[:, idx], labels[:, idx])
        metrics[f"{prefix}_pcc_{target}_scaled"] = pcc
        metrics[f"{prefix}_spearman_{target}_scaled"] = spearman
        pcc_values.append(pcc)
        spearman_values.append(spearman)
    metrics[f"{prefix}_pcc_mean"] = float(np.nanmean(pcc_values))
    metrics[f"{prefix}_spearman_mean"] = float(np.nanmean(spearman_values))
    return metrics


def average_predictions(forward_output: Any, reverse_output: Any | None) -> np.ndarray:
    forward = as_2d_array(forward_output.predictions)
    if reverse_output is None:
        return forward
    reverse = as_2d_array(reverse_output.predictions)
    return 0.5 * (forward + reverse)


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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)

    from accelerate import PartialState

    state = PartialState()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Malinois data from %s", args.source_url)
    with state.main_process_first():
        raw = load_malinois_dataset(args)
    split_raw = split_by_chromosome(raw)

    train_raw = select_limit(split_raw["train"], args.max_train_samples)
    validation_raw = select_limit(
        filter_metric_rows(split_raw["validation"], args.metric_se_threshold),
        args.max_eval_samples,
    )
    test_raw = select_limit(
        filter_metric_rows(split_raw["test"], args.metric_se_threshold),
        args.max_eval_samples,
    )

    label_transform = fit_label_transform(train_raw)
    unaugmented_train_size = len(train_raw)
    train_raw = augment_train_dataset(train_raw, args)
    augmented_train_size = len(train_raw)
    if state.is_main_process and augmented_train_size != unaugmented_train_size:
        logger.info(
            "Augmented train split from %d to %d rows",
            unaugmented_train_size,
            augmented_train_size,
        )

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

    def tokenize(batch: dict[str, list[Any]], *, rc: bool = False) -> dict[str, Any]:
        sequences = batch["sequence"]
        if rc:
            sequences = [reverse_complement(sequence) for sequence in sequences]
        tokenizer_kwargs = {}
        if args.dna_mode == "auto_dna_tags":
            tokenizer_kwargs["auto_dna_tags"] = True
        tokenized = tokenizer(
            prepare_sequences(sequences, args.dna_mode),
            truncation=True,
            max_length=args.max_length,
            padding=False,
            **tokenizer_kwargs,
        )
        tokenized["labels"] = label_values(batch, label_transform)
        return tokenized

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "remove_columns": train_raw.column_names,
        "desc": "Tokenizing Malinois",
    }
    if args.map_num_proc and args.map_num_proc > 1:
        map_kwargs["num_proc"] = args.map_num_proc

    with state.main_process_first():
        train_dataset = train_raw.map(tokenize, **map_kwargs)
        validation_dataset = validation_raw.map(tokenize, **map_kwargs)
        test_dataset = test_raw.map(tokenize, **map_kwargs)
        validation_rc_dataset = (
            validation_raw.map(
                lambda batch: tokenize(batch, rc=True),
                **{**map_kwargs, "desc": "Tokenizing Malinois validation RC"},
            )
            if args.rc_eval_average
            else None
        )
        test_rc_dataset = (
            test_raw.map(
                lambda batch: tokenize(batch, rc=True),
                **{**map_kwargs, "desc": "Tokenizing Malinois test RC"},
            )
            if args.rc_eval_average
            else None
        )

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
        "target_columns": list(TARGET_COLUMNS),
        "se_columns": list(SE_COLUMNS),
        "target_names": list(TARGET_NAMES),
        "validation_chromosomes": list(VALIDATION_CHROMS),
        "test_chromosomes": list(TEST_CHROMS),
        "label_transform": label_transform,
        "unaugmented_train_size": unaugmented_train_size,
        "augmented_train_size": augmented_train_size,
        "validation_size_metric_filtered": len(validation_raw),
        "test_size_metric_filtered": len(test_raw),
        "effective_eval_steps": eval_steps,
        "global_train_batch_size": global_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "loss_type": "mse",
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

    trainer = Trainer(
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

    def evaluate_split(
        split: str,
        tokenized_dataset: Any,
        tokenized_rc_dataset: Any | None,
        raw_split: Any,
    ) -> None:
        forward_output = trainer.predict(tokenized_dataset, metric_key_prefix=split)
        reverse_output = (
            trainer.predict(tokenized_rc_dataset, metric_key_prefix=f"{split}_rc")
            if tokenized_rc_dataset is not None
            else None
        )
        predictions = average_predictions(forward_output, reverse_output)
        labels = labels_from_dataset(raw_split, label_transform)
        metrics = {
            **forward_output.metrics,
            **full_metrics(split, predictions, labels, raw_split),
        }
        if reverse_output is not None:
            metrics[f"{split}_rc_eval_loss"] = reverse_output.metrics.get(
                f"{split}_rc_loss"
            )
        if trainer.is_world_process_zero():
            write_json(output_dir / f"{split}_metrics.json", metrics)

    evaluate_split(
        "validation", validation_dataset, validation_rc_dataset, validation_raw
    )
    if not args.skip_test:
        evaluate_split("test", test_dataset, test_rc_dataset, test_raw)


if __name__ == "__main__":
    main()
