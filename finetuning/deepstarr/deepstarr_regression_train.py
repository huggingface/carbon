"""Fine-tune Carbon models on DeepSTARR enhancer activity regression.

The script trains on the train split, selects the best checkpoint by validation
PCC, and reports final validation/test PCC plus log-space PCC.
"""

import argparse
import json
import logging
import math
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_callback import TrainerCallback

DATASET_NAME = "GenerTeam/DeepSTARR-enhancer-activity"
TARGET_NAMES = ("dev", "hk")
SCALED_COLUMNS = ("Dev_log2_enrichment_scaled", "Hk_log2_enrichment_scaled")
RAW_LOG_COLUMNS = ("Dev_log2_enrichment", "Hk_log2_enrichment")
MIN_TRACKIO_VERSION = (0, 25, 1)
DEFAULT_LRS = {
    "frozen_lm": 3e-4,
    "full_finetune": 1e-5,
}

logger = logging.getLogger(__name__)
DNA_BASES = ("A", "C", "G", "T")
DNA_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regression fine-tuning on GenerTeam/DeepSTARR-enhancer-activity"
    )
    parser.add_argument("--model", required=True, help="HF model repo or local path")
    parser.add_argument("--revision", default=None, help="Optional model revision")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for --eval_only. Tokenizer still loads from --model.",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        default=None,
        help=(
            "Optional model checkpoint to warm-start training from. Tokenizer "
            "still loads from --model and optimizer/scheduler state is fresh."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        default=DATASET_NAME,
        help="HF dataset name",
    )
    parser.add_argument(
        "--output_dir",
        default="scratch/deepstarr_regression/run",
        help="Run output directory",
    )
    parser.add_argument(
        "--finetune_mode",
        choices=["frozen_lm", "full_finetune"],
        required=True,
        help="Whether to freeze the LM backbone or train all parameters",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate. Defaults are mode-specific if omitted.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument(
        "--lr_scheduler_kwargs",
        default=None,
        help="Optional JSON object passed to the Transformers LR scheduler.",
    )
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--dna_tokenization_mode",
        choices=["plain", "dna_tags", "auto_dna_tags"],
        default="dna_tags",
        help=(
            "How to present DeepSTARR DNA strings to the Carbon tokenizer. "
            "`dna_tags` wraps raw strings as <dna>...</dna>; `auto_dna_tags` "
            "passes auto_dna_tags=True to tokenizers that support it; `plain` "
            "preserves the previous raw-string behavior."
        ),
    )
    parser.add_argument(
        "--truncate_dna_to_multiple",
        type=int,
        default=0,
        help=(
            "If >0, truncate raw DNA sequence length to this multiple before "
            "wrapping/tokenization. Use 6 to avoid Carbon tail k-mer padding."
        ),
    )
    parser.add_argument(
        "--truncate_dna_side",
        choices=["right", "left"],
        default="right",
        help="Which side to trim when --truncate_dna_to_multiple removes bases.",
    )
    parser.add_argument(
        "--kmer_phase_augment",
        action="store_true",
        help=(
            "Add train-only Carbon 6-mer phase-jittered sequence copies before "
            "tokenization."
        ),
    )
    parser.add_argument(
        "--kmer_phase_augment_copies",
        type=int,
        default=1,
        help="Number of phase-jittered copies to add for each train sequence view.",
    )
    parser.add_argument(
        "--kmer_phase_max_shift",
        type=int,
        default=5,
        help="Maximum left/right base shift for 6-mer phase augmentation.",
    )
    parser.add_argument(
        "--kmer_phase_output_length",
        type=int,
        default=246,
        help="Length of phase-augmented DNA sequences. Use multiples of 6.",
    )
    parser.add_argument(
        "--kmer_phase_keep_original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep unjittered train rows when phase augmentation is enabled.",
    )
    parser.add_argument(
        "--reverse_complement_augment",
        choices=["none", "duplicate", "stochastic"],
        default="none",
        help=(
            "Train-only reverse-complement augmentation. `duplicate` adds RC "
            "copies; `stochastic` materializes one sampled orientation per row."
        ),
    )
    parser.add_argument(
        "--reverse_complement_probability",
        type=float,
        default=0.5,
        help="RC probability for --reverse_complement_augment stochastic.",
    )
    parser.add_argument(
        "--train_token_mask_rate",
        type=float,
        default=0.0,
        help="Train-only probability of masking each Carbon DNA 6-mer token.",
    )
    parser.add_argument(
        "--train_token_mask_mode",
        choices=["oov", "random_kmer"],
        default="oov",
        help="How to replace masked DNA 6-mer tokens during training.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="bfloat16",
        help="Weight dtype used in from_pretrained",
    )
    parser.add_argument(
        "--require_fp32_master_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail at train start unless trainable FSDP flat/master parameters "
            "are materialized as float32."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional Transformers attention implementation",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--map_num_proc", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--eval_strategy",
        choices=["no", "epoch", "steps"],
        default="epoch",
    )
    parser.add_argument(
        "--save_strategy",
        choices=["no", "epoch", "steps", "best"],
        default="epoch",
    )
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--save_only_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only save model weights in checkpoints, without optimizer/scheduler state.",
    )
    parser.add_argument("--eval_accumulation_steps", type=int, default=None)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Stop after this many evals without improvement in the best-model metric.",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum improvement required by EarlyStoppingCallback.",
    )
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--load_best_model_at_end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the best validation checkpoint at the end of training.",
    )
    parser.add_argument(
        "--head_type",
        choices=[
            "sequence",
            "mean_pool",
            "mean_max_pool",
            "attention_pool",
            "cnn_pool",
        ],
        default="sequence",
        help="Regression head/pooling strategy. `sequence` uses AutoModelForSequenceClassification.",
    )
    parser.add_argument(
        "--loss_type",
        choices=[
            "mse",
            "huber",
            "pearson",
            "mse_pearson",
            "ccc",
            "mse_ccc",
            "pearson_calibrated",
            "cross_entropy",
        ],
        default="mse",
        help="Training loss. Use cross_entropy with --prediction_mode categorical.",
    )
    parser.add_argument(
        "--pearson_loss_weight",
        type=float,
        default=0.1,
        help="Weight for pearson component when --loss_type=mse_pearson.",
    )
    parser.add_argument(
        "--ccc_loss_weight",
        type=float,
        default=0.1,
        help="Weight for CCC component when --loss_type=mse_ccc.",
    )
    parser.add_argument(
        "--calibration_mean_weight",
        type=float,
        default=1.0,
        help="Mean-alignment penalty weight for --loss_type=pearson_calibrated.",
    )
    parser.add_argument(
        "--calibration_std_weight",
        type=float,
        default=1.0,
        help="Std-alignment penalty weight for --loss_type=pearson_calibrated.",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help="Delta/beta for Huber loss.",
    )
    parser.add_argument(
        "--label_transform",
        choices=[
            "dataset_scaled",
            "train_zscore_scaled",
            "raw_log_train_zscore",
            "raw_log_train_robust",
        ],
        default="dataset_scaled",
        help=(
            "Target transform used for training labels. All modes are affine "
            "per target, so PCC remains comparable while MSE/CCC optimization "
            "sees different label scales."
        ),
    )
    parser.add_argument(
        "--prediction_mode",
        choices=["regression", "categorical"],
        default="regression",
        help=(
            "Whether the head predicts continuous targets directly or target "
            "categories. Categorical mode trains cross-entropy over target-wise "
            "bins and evaluates PCC from expected bin-center values."
        ),
    )
    parser.add_argument(
        "--num_label_bins",
        type=int,
        default=10,
        help="Number of train-quantile bins per target for categorical mode.",
    )
    parser.add_argument(
        "--label_density_weighting",
        choices=["none", "lds"],
        default="none",
        help=(
            "Apply label-distribution smoothing sample weights to the training "
            "loss. Validation/test loss remains unweighted."
        ),
    )
    parser.add_argument("--label_density_num_bins", type=int, default=40)
    parser.add_argument("--label_density_kernel_size", type=int, default=5)
    parser.add_argument("--label_density_kernel_sigma", type=float, default=2.0)
    parser.add_argument("--label_density_power", type=float, default=1.0)
    parser.add_argument("--label_density_min_weight", type=float, default=0.25)
    parser.add_argument("--label_density_max_weight", type=float, default=4.0)
    parser.add_argument(
        "--report_to",
        default="none",
        help="Non-Trackio Trainer reporters only. Use --trackio for Trackio.",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--trackio",
        action="store_true",
        help="Enable Trackio logging even when --trackio_space_id is not set",
    )
    parser.add_argument(
        "--disable_trackio",
        action="store_true",
        help="Disable Trackio even if TRACKIO_SPACE_ID is present",
    )
    parser.add_argument(
        "--trackio_space_id",
        default=os.environ.get("TRACKIO_SPACE_ID"),
        help="Trackio Hugging Face Space id, e.g. org/space-name",
    )
    parser.add_argument(
        "--trackio_project",
        default=os.environ.get("TRACKIO_PROJECT", "deepstarr-regression"),
        help="Trackio project name",
    )
    parser.add_argument(
        "--trackio_run_name",
        default=os.environ.get("TRACKIO_RUN_NAME"),
        help="Trackio run name. Defaults to --run_name or output directory name.",
    )
    parser.add_argument(
        "--trackio_group",
        default=os.environ.get("TRACKIO_GROUP"),
        help="Optional Trackio group name",
    )
    parser.add_argument(
        "--trackio_private",
        action=argparse.BooleanOptionalAction,
        default=env_bool("TRACKIO_PRIVATE"),
        help="Whether Trackio should create a private Space when creating one",
    )
    parser.add_argument(
        "--trackio_auto_log_gpu",
        action=argparse.BooleanOptionalAction,
        default=env_bool("TRACKIO_AUTO_LOG_GPU"),
    )
    parser.add_argument("--trackio_gpu_log_interval", type=float, default=10.0)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument(
        "--save_final_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the final in-memory model to output_dir/best_model after training.",
    )
    parser.add_argument(
        "--save_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write JSONL prediction files for evaluated splits",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.lr_scheduler_kwargs is not None:
        try:
            args.lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)
        except json.JSONDecodeError as exc:
            parser.error(f"--lr_scheduler_kwargs must be valid JSON: {exc}")
        if not isinstance(args.lr_scheduler_kwargs, dict):
            parser.error("--lr_scheduler_kwargs must decode to a JSON object")

    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LRS[args.finetune_mode]
    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_train_batch_size
    if args.eval_only and not args.checkpoint:
        parser.error("--checkpoint is required with --eval_only")
    if (
        args.load_best_model_at_end
        and not args.eval_only
        and not args.skip_validation
        and args.eval_strategy != args.save_strategy
        and args.save_strategy != "best"
    ):
        parser.error(
            "--eval_strategy and --save_strategy must match when selecting "
            "the best checkpoint"
        )
    if (
        args.load_best_model_at_end
        and not args.eval_only
        and not args.skip_validation
        and (args.eval_strategy == "no" or args.save_strategy == "no")
    ):
        parser.error(
            "--load_best_model_at_end requires eval/save strategies other than 'no'"
        )
    if args.fp16 and args.bf16:
        parser.error("Use at most one of --fp16 and --bf16")
    if args.require_fp32_master_weights and args.torch_dtype != "float32":
        parser.error(
            "--require_fp32_master_weights requires --torch_dtype float32. "
            "FSDP2 optimizer/master shards keep the original loaded dtype."
        )
    if args.skip_validation and args.skip_test:
        parser.error("At least one eval split must be enabled")
    if args.huber_delta <= 0.0:
        parser.error("--huber_delta must be positive")
    if args.pearson_loss_weight < 0.0:
        parser.error("--pearson_loss_weight must be non-negative")
    if args.ccc_loss_weight < 0.0:
        parser.error("--ccc_loss_weight must be non-negative")
    if args.calibration_mean_weight < 0.0:
        parser.error("--calibration_mean_weight must be non-negative")
    if args.calibration_std_weight < 0.0:
        parser.error("--calibration_std_weight must be non-negative")
    if args.num_label_bins < 2:
        parser.error("--num_label_bins must be at least 2")
    if args.prediction_mode == "regression" and args.loss_type == "cross_entropy":
        parser.error("--loss_type cross_entropy requires --prediction_mode categorical")
    if args.prediction_mode == "categorical" and args.loss_type != "cross_entropy":
        parser.error("--prediction_mode categorical requires --loss_type cross_entropy")
    if args.label_density_num_bins < 2:
        parser.error("--label_density_num_bins must be at least 2")
    if args.label_density_kernel_size < 1:
        parser.error("--label_density_kernel_size must be positive")
    if args.label_density_kernel_sigma <= 0.0:
        parser.error("--label_density_kernel_sigma must be positive")
    if args.label_density_power < 0.0:
        parser.error("--label_density_power must be non-negative")
    if args.label_density_min_weight <= 0.0:
        parser.error("--label_density_min_weight must be positive")
    if args.label_density_max_weight < args.label_density_min_weight:
        parser.error("--label_density_max_weight must be >= min weight")
    if args.truncate_dna_to_multiple < 0:
        parser.error("--truncate_dna_to_multiple must be non-negative")
    if args.kmer_phase_augment_copies < 0:
        parser.error("--kmer_phase_augment_copies must be non-negative")
    if args.kmer_phase_augment and args.kmer_phase_augment_copies < 1:
        parser.error("--kmer_phase_augment requires at least one augment copy")
    if args.kmer_phase_max_shift < 0:
        parser.error("--kmer_phase_max_shift must be non-negative")
    if args.kmer_phase_output_length <= 0:
        parser.error("--kmer_phase_output_length must be positive")
    if args.kmer_phase_output_length % 6 != 0:
        parser.error("--kmer_phase_output_length must be a multiple of 6")
    if not 0.0 <= args.reverse_complement_probability <= 1.0:
        parser.error("--reverse_complement_probability must be between 0 and 1")
    if not 0.0 <= args.train_token_mask_rate <= 1.0:
        parser.error("--train_token_mask_rate must be between 0 and 1")
    if str(args.report_to).lower() in {"trackio", "all"}:
        parser.error(
            "Do not use Trainer's native Trackio reporting. Use --trackio or "
            "TRACKIO_SPACE_ID for direct Trackio logging."
        )
    return args


def trackio_enabled(args: argparse.Namespace) -> bool:
    if args.disable_trackio:
        return False
    return bool(args.trackio or args.trackio_space_id)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit_linear_calibration_from_arrays(
    scaled: np.ndarray, raw_log: np.ndarray
) -> tuple[float, float]:
    scaled = np.asarray(scaled, dtype=np.float64)
    raw_log = np.asarray(raw_log, dtype=np.float64)
    mask = np.isfinite(scaled) & np.isfinite(raw_log)
    if int(mask.sum()) < 2:
        raise ValueError("Need at least two finite points to fit calibration")
    scaled = scaled[mask]
    raw_log = raw_log[mask]
    variance = float(np.var(scaled))
    if variance == 0.0:
        raise ValueError("Cannot fit calibration with zero-variance scaled labels")
    slope = float(np.cov(scaled, raw_log, bias=True)[0, 1] / variance)
    intercept = float(np.mean(raw_log) - slope * np.mean(scaled))
    return slope, intercept


def finite_mean_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if int(finite.size) == 0:
        raise ValueError("Cannot compute label transform stats without finite labels")
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if std <= 0.0 or not math.isfinite(std):
        raise ValueError("Cannot compute label transform with zero/non-finite std")
    return mean, std


def fit_label_transform(train_dataset: Any, mode: str) -> dict[str, Any]:
    targets = {}
    for target, scaled_col, raw_col in zip(
        TARGET_NAMES, SCALED_COLUMNS, RAW_LOG_COLUMNS
    ):
        if mode == "dataset_scaled":
            source_col = scaled_col
            center = 0.0
            scale = 1.0
            stats = {}
        elif mode == "train_zscore_scaled":
            source_col = scaled_col
            center, scale = finite_mean_std(
                np.asarray(train_dataset[source_col], dtype=np.float64)
            )
            stats = {}
        elif mode == "raw_log_train_zscore":
            source_col = raw_col
            center, scale = finite_mean_std(
                np.asarray(train_dataset[source_col], dtype=np.float64)
            )
            stats = {}
        elif mode == "raw_log_train_robust":
            source_col = raw_col
            values = np.asarray(train_dataset[source_col], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if int(finite.size) == 0:
                raise ValueError(
                    "Cannot compute robust label transform stats without finite labels"
                )
            center = float(np.median(finite))
            q25, q75 = np.percentile(finite, [25, 75])
            iqr = float(q75 - q25)
            scale = iqr / 1.349 if iqr > 0.0 else 0.0
            if scale <= 0.0 or not math.isfinite(scale):
                _, scale = finite_mean_std(finite)
            stats = {"q25": float(q25), "q75": float(q75), "iqr": iqr}
        else:
            raise ValueError(f"Unknown label transform mode: {mode}")

        targets[target] = {
            "source_column": source_col,
            "center": center,
            "scale": scale,
            **stats,
        }

    return {"mode": mode, "targets": targets}


def label_values_from_dataset(
    dataset: Any, label_transform: dict[str, Any]
) -> np.ndarray:
    columns = []
    for target in TARGET_NAMES:
        spec = label_transform["targets"][target]
        values = np.asarray(dataset[spec["source_column"]], dtype=np.float64)
        columns.append((values - spec["center"]) / spec["scale"])
    return np.stack(columns, axis=1)


def label_values_from_batch(
    batch: dict[str, list[Any]], label_transform: dict[str, Any]
) -> list[list[float]]:
    transformed_columns = []
    for target in TARGET_NAMES:
        spec = label_transform["targets"][target]
        values = np.asarray(batch[spec["source_column"]], dtype=np.float64)
        transformed_columns.append((values - spec["center"]) / spec["scale"])
    stacked = np.stack(transformed_columns, axis=1)
    return stacked.astype(np.float32).tolist()


def fit_category_metadata(
    train_dataset: Any,
    label_transform: dict[str, Any],
    num_bins: int,
) -> dict[str, Any]:
    labels = label_values_from_dataset(train_dataset, label_transform)
    targets = {}
    for idx, target in enumerate(TARGET_NAMES):
        values = labels[:, idx]
        finite = values[np.isfinite(values)]
        if int(finite.size) == 0:
            raise ValueError(f"No finite labels found for categorical target {target}")

        quantile_edges = np.quantile(
            finite,
            np.linspace(0.0, 1.0, num_bins + 1),
        )
        thresholds = np.asarray(quantile_edges[1:-1], dtype=np.float64)
        if thresholds.size and np.any(np.diff(thresholds) <= 0.0):
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            if lo == hi:
                thresholds = np.asarray([], dtype=np.float64)
            else:
                thresholds = np.linspace(lo, hi, num_bins + 1, dtype=np.float64)[1:-1]

        bin_ids = np.searchsorted(thresholds, finite, side="right")
        centers = []
        for bin_idx in range(num_bins):
            in_bin = finite[bin_ids == bin_idx]
            if int(in_bin.size):
                center = float(np.mean(in_bin))
            else:
                lo = (
                    float(np.min(finite))
                    if bin_idx == 0
                    else float(thresholds[bin_idx - 1])
                )
                hi = (
                    float(np.max(finite))
                    if bin_idx >= thresholds.size
                    else float(thresholds[bin_idx])
                )
                center = 0.5 * (lo + hi)
            centers.append(center)

        targets[target] = {
            "thresholds": thresholds.tolist(),
            "centers": centers,
            "train_min": float(np.min(finite)),
            "train_max": float(np.max(finite)),
        }

    return {
        "mode": "train_quantile",
        "num_bins": num_bins,
        "targets": targets,
    }


def category_labels_from_values(
    values: np.ndarray,
    category_metadata: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = []
    for idx, target in enumerate(TARGET_NAMES):
        thresholds = np.asarray(
            category_metadata["targets"][target]["thresholds"],
            dtype=np.float64,
        )
        labels.append(np.searchsorted(thresholds, values[:, idx], side="right"))
    return np.stack(labels, axis=1).astype(np.int64)


def category_labels_from_batch(
    batch: dict[str, list[Any]],
    label_transform: dict[str, Any],
    category_metadata: dict[str, Any],
) -> list[list[int]]:
    values = np.asarray(
        label_values_from_batch(batch, label_transform), dtype=np.float64
    )
    return category_labels_from_values(values, category_metadata).tolist()


def categorical_logits_to_expected_values(
    predictions: Any,
    category_metadata: dict[str, Any],
) -> np.ndarray:
    logits = normalize_predictions(predictions)
    num_bins = int(category_metadata["num_bins"])
    if logits.shape[-1] != len(TARGET_NAMES) * num_bins:
        raise ValueError(
            f"Expected categorical logits width {len(TARGET_NAMES) * num_bins}, "
            f"got {logits.shape[-1]}"
        )
    logits = logits.reshape(logits.shape[0], len(TARGET_NAMES), num_bins)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    expected = []
    for idx, target in enumerate(TARGET_NAMES):
        centers = np.asarray(
            category_metadata["targets"][target]["centers"],
            dtype=np.float64,
        )
        expected.append(np.sum(probs[:, idx, :] * centers.reshape(1, -1), axis=-1))
    return np.stack(expected, axis=1)


def predictions_to_label_values(
    predictions: Any,
    prediction_mode: str,
    category_metadata: dict[str, Any] | None,
) -> np.ndarray:
    if prediction_mode == "categorical":
        if category_metadata is None:
            raise ValueError(
                "category_metadata is required for categorical predictions"
            )
        return categorical_logits_to_expected_values(predictions, category_metadata)
    return normalize_predictions(predictions)


def make_compute_metrics(
    prediction_mode: str,
    category_metadata: dict[str, Any] | None,
) -> Any:
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        predictions = predictions_to_label_values(
            eval_pred.predictions,
            prediction_mode,
            category_metadata,
        )
        labels = normalize_label_ids(eval_pred.label_ids)
        metrics = {}
        pcc_values = []
        for idx, target in enumerate(TARGET_NAMES):
            pcc = pearson_corr(predictions[:, idx], labels[:, idx])
            metrics[f"pcc_{target}_scaled"] = pcc
            pcc_values.append(pcc)
        metrics["pcc_mean"] = float(np.nanmean(pcc_values))
        return metrics

    return compute_metrics


def fit_log_calibration(
    train_dataset: Any, label_transform: dict[str, Any] | None = None
) -> dict[str, dict[str, float]]:
    if label_transform is None:
        label_transform = fit_label_transform(train_dataset, "dataset_scaled")
    train_labels = label_values_from_dataset(train_dataset, label_transform)
    calibration = {}
    for idx, (target, raw_col) in enumerate(zip(TARGET_NAMES, RAW_LOG_COLUMNS)):
        slope, intercept = fit_linear_calibration_from_arrays(
            train_labels[:, idx],
            np.asarray(train_dataset[raw_col], dtype=np.float64),
        )
        calibration[target] = {"slope": slope, "intercept": intercept}
    return calibration


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


def compute_scaled_metrics(eval_pred: Any) -> dict[str, float]:
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


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(x / sigma))
    kernel_sum = float(kernel.sum())
    if kernel_sum == 0.0:
        return np.ones_like(kernel) / float(kernel.size)
    return kernel / kernel_sum


def fit_lds_label_density_weights(
    train_dataset: Any,
    *,
    num_bins: int,
    kernel_size: int,
    kernel_sigma: float,
    power: float,
    min_weight: float,
    max_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.stack(
        [
            np.asarray(train_dataset[column], dtype=np.float64)
            for column in SCALED_COLUMNS
        ],
        axis=1,
    )
    kernel = gaussian_kernel(kernel_size, kernel_sigma)
    target_weights = []
    target_metadata = {}

    for idx, target in enumerate(TARGET_NAMES):
        values = labels[:, idx]
        finite = np.isfinite(values)
        if int(finite.sum()) == 0:
            raise ValueError(f"No finite labels found for target {target}")
        lo = float(np.min(values[finite]))
        hi = float(np.max(values[finite]))
        if lo == hi:
            target_weight = np.ones_like(values, dtype=np.float64)
            edges = np.linspace(lo - 0.5, hi + 0.5, num_bins + 1)
            smoothed = np.ones(num_bins, dtype=np.float64)
        else:
            counts, edges = np.histogram(values[finite], bins=num_bins, range=(lo, hi))
            smoothed = np.convolve(counts.astype(np.float64), kernel, mode="same")
            bin_ids = np.searchsorted(edges, values, side="right") - 1
            bin_ids = np.clip(bin_ids, 0, num_bins - 1)
            effective_density = np.maximum(smoothed[bin_ids], 1e-12)
            target_weight = np.power(effective_density, -power)
            target_weight[~finite] = 1.0
        target_weights.append(target_weight)
        target_metadata[target] = {
            "label_min": lo,
            "label_max": hi,
            "num_bins": num_bins,
            "kernel_size": int(kernel.size),
            "kernel_sigma": kernel_sigma,
            "smoothed_density_min": float(np.min(smoothed)),
            "smoothed_density_max": float(np.max(smoothed)),
        }

    weights = np.mean(np.stack(target_weights, axis=1), axis=1)
    weights = np.clip(weights, min_weight, max_weight)
    mean_weight = float(np.mean(weights))
    if mean_weight <= 0.0 or not math.isfinite(mean_weight):
        weights = np.ones_like(weights, dtype=np.float64)
    else:
        weights = weights / mean_weight
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / float(np.mean(weights))

    metadata = {
        "mode": "lds",
        "num_bins": num_bins,
        "kernel_size": int(kernel.size),
        "kernel_sigma": kernel_sigma,
        "power": power,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "train_weight_min": float(np.min(weights)),
        "train_weight_max": float(np.max(weights)),
        "train_weight_mean": float(np.mean(weights)),
        "train_weight_std": float(np.std(weights)),
        "targets": target_metadata,
    }
    return weights.astype(np.float32), metadata


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


def split_dna_tags(sequence: str) -> tuple[str, str, str]:
    prefix = "<dna>"
    suffix = "</dna>"
    if sequence.startswith(prefix) and sequence.endswith(suffix):
        return prefix, sequence[len(prefix) : -len(suffix)], suffix
    return "", sequence, ""


def random_dna_bases(length: int, rng: np.random.Generator) -> str:
    if length <= 0:
        return ""
    return "".join(rng.choice(DNA_BASES, size=length).tolist())


def reverse_complement_dna_sequence(sequence: Any) -> str:
    prefix, dna, suffix = split_dna_tags(str(sequence).strip())
    return f"{prefix}{dna.translate(DNA_COMPLEMENT)[::-1].upper()}{suffix}"


def augmentation_rng(
    seed: int, dataset_index: int, view_index: int, copy_index: int, salt: int
) -> np.random.Generator:
    value = (
        int(seed) * 1_000_003
        + int(dataset_index) * 9_176
        + int(view_index) * 1_009
        + int(copy_index) * 131
        + int(salt)
    ) % (2**32)
    return np.random.default_rng(value)


def kmer_phase_jitter_dna_sequence(
    sequence: Any,
    *,
    max_shift: int,
    output_length: int,
    rng: np.random.Generator,
) -> str:
    prefix, dna, suffix = split_dna_tags(str(sequence).strip())
    dna = dna.upper()
    shift = 0
    if max_shift > 0:
        shift = int(rng.integers(-max_shift, max_shift + 1))

    left_flank = random_dna_bases(max_shift, rng)
    right_flank = random_dna_bases(max_shift, rng)
    padded = f"{left_flank}{dna}{right_flank}"
    start = max_shift + shift
    window = padded[start : start + output_length]
    if len(window) < output_length:
        window = f"{window}{random_dna_bases(output_length - len(window), rng)}"
    return f"{prefix}{window}{suffix}"


def truncate_dna_sequence(sequence: str, multiple: int, side: str) -> str:
    if multiple <= 1:
        return sequence

    prefix, dna, suffix = split_dna_tags(sequence)
    remainder = len(dna) % multiple
    if remainder == 0:
        return sequence

    keep = len(dna) - remainder
    if keep <= 0:
        dna = ""
    elif side == "left":
        dna = dna[-keep:]
    else:
        dna = dna[:keep]
    return f"{prefix}{dna}{suffix}"


def prepare_dna_sequences(sequences: list[Any], args: argparse.Namespace) -> list[str]:
    prepared = [str(sequence).strip() for sequence in sequences]
    if args.truncate_dna_to_multiple > 0:
        prepared = [
            truncate_dna_sequence(
                sequence,
                args.truncate_dna_to_multiple,
                args.truncate_dna_side,
            )
            for sequence in prepared
        ]
    if args.dna_tokenization_mode == "dna_tags":
        prepared = [
            (
                sequence
                if sequence.startswith("<dna>") and sequence.endswith("</dna>")
                else f"<dna>{sequence}</dna>"
            )
            for sequence in prepared
        ]
    return prepared


def sequence_augmentation_enabled(args: argparse.Namespace) -> bool:
    return bool(args.kmer_phase_augment or args.reverse_complement_augment != "none")


def append_augmented_row(
    output: dict[str, list[Any]],
    batch: dict[str, list[Any]],
    row_idx: int,
    sequence: str,
    columns: list[str],
) -> None:
    for column in columns:
        output[column].append(
            sequence if column == "sequence" else batch[column][row_idx]
        )


def augment_deepstarr_train_dataset(train_raw: Any, args: argparse.Namespace) -> Any:
    if not sequence_augmentation_enabled(args):
        return train_raw

    columns = list(train_raw.column_names)

    def augment_batch(
        batch: dict[str, list[Any]], indices: list[int]
    ) -> dict[str, list[Any]]:
        output: dict[str, list[Any]] = {column: [] for column in columns}
        for row_idx, dataset_index in enumerate(indices):
            sequence = str(batch["sequence"][row_idx]).strip()
            if args.reverse_complement_augment == "stochastic":
                rng = augmentation_rng(args.seed, dataset_index, 0, 0, salt=17)
                if float(rng.random()) < args.reverse_complement_probability:
                    base_views = [reverse_complement_dna_sequence(sequence)]
                else:
                    base_views = [sequence]
            else:
                base_views = [sequence]
                if args.reverse_complement_augment == "duplicate":
                    base_views.append(reverse_complement_dna_sequence(sequence))

            include_base_views = (
                not args.kmer_phase_augment or args.kmer_phase_keep_original
            )
            for view_idx, base_sequence in enumerate(base_views):
                if include_base_views:
                    append_augmented_row(output, batch, row_idx, base_sequence, columns)
                if args.kmer_phase_augment:
                    for copy_idx in range(args.kmer_phase_augment_copies):
                        rng = augmentation_rng(
                            args.seed,
                            dataset_index,
                            view_idx,
                            copy_idx,
                            salt=29,
                        )
                        augmented_sequence = kmer_phase_jitter_dna_sequence(
                            base_sequence,
                            max_shift=args.kmer_phase_max_shift,
                            output_length=args.kmer_phase_output_length,
                            rng=rng,
                        )
                        append_augmented_row(
                            output,
                            batch,
                            row_idx,
                            augmented_sequence,
                            columns,
                        )
        return output

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "with_indices": True,
        "desc": "Augmenting DeepSTARR train sequences",
        "load_from_cache_file": False,
    }
    if args.map_num_proc and args.map_num_proc > 1:
        map_kwargs["num_proc"] = args.map_num_proc
    return train_raw.map(augment_batch, **map_kwargs)


def float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def tokenized_cache_tag(args: argparse.Namespace) -> str:
    parts = [
        f"maxlen{args.max_length}",
        f"dna-{args.dna_tokenization_mode}",
        f"trunc{args.truncate_dna_to_multiple}-{args.truncate_dna_side}",
        f"rc-{args.reverse_complement_augment}",
    ]
    if args.reverse_complement_augment == "stochastic":
        parts.append(f"rcp{float_tag(args.reverse_complement_probability)}")
    if args.kmer_phase_augment:
        parts.extend(
            [
                f"phase-c{args.kmer_phase_augment_copies}",
                f"s{args.kmer_phase_max_shift}",
                f"l{args.kmer_phase_output_length}",
                f"keep{int(args.kmer_phase_keep_original)}",
            ]
        )
    else:
        parts.append("phase-none")
    return "_".join(parts)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def report_parameter_counts(model: Any) -> dict[str, int | float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    pct = 100.0 * trainable / total if total else 0.0
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": pct,
    }


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    scalar = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                continue
            scalar[key] = value
    return scalar


def parse_version_triplet(version_string: str) -> tuple[int, int, int]:
    parts = []
    for piece in version_string.split(".")[:3]:
        digits = []
        for char in piece:
            if not char.isdigit():
                break
            digits.append(char)
        parts.append(int("".join(digits) or "0"))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


class TrackioMetricsCallback(TrainerCallback):
    def __init__(self, trackio_module: Any):
        self.trackio = trackio_module

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero or not logs:
            return
        payload = scalar_metrics(logs)
        self.trackio.log(payload, step=int(state.global_step))

    def on_predict(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero or not metrics:
            return
        self.trackio.log(scalar_metrics(metrics), step=int(state.global_step))


def summarize_optimizer_param_dtypes(optimizer: Any | None) -> dict[str, int]:
    if optimizer is None:
        return {}

    dtype_counts: dict[str, int] = {}
    for group in getattr(optimizer, "param_groups", []):
        for param in group.get("params", []):
            if not getattr(param, "requires_grad", False):
                continue
            dtype_name = str(getattr(param, "dtype", "unknown"))
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
    return dtype_counts


def empty_fsdp_master_weight_report() -> dict[str, Any]:
    return {
        "fsdp_available": False,
        "fsdp_module_count": 0,
        "trainable_flat_param_count": 0,
        "flat_param_dtype_counts": {},
        "meta_trainable_flat_param_count": 0,
        "non_fp32_trainable_flat_params": [],
    }


def summarize_fsdp_master_weight_modules(
    fsdp_modules: list[Any],
    *,
    fsdp_available: bool = True,
) -> dict[str, Any]:
    report = empty_fsdp_master_weight_report()
    report["fsdp_available"] = fsdp_available
    report["fsdp_module_count"] = len(fsdp_modules)

    dtype_counts: dict[str, int] = {}
    bad_params: list[dict[str, Any]] = []
    meta_count = 0
    trainable_count = 0

    for module_index, module in enumerate(fsdp_modules):
        if not getattr(module, "_has_params", False):
            continue
        param = getattr(module, "_flat_param", None)
        if param is None or not getattr(param, "requires_grad", False):
            continue

        trainable_count += 1
        dtype_name = str(getattr(param, "dtype", "unknown"))
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

        device = getattr(param, "device", None)
        if device == torch.device("meta"):
            meta_count += 1
            continue
        if getattr(param, "dtype", None) != torch.float32:
            managed_names = list(getattr(param, "_fqns", []))
            wrapped_module = getattr(module, "module", module)
            bad_params.append(
                {
                    "module_index": module_index,
                    "module_class": wrapped_module.__class__.__name__,
                    "dtype": dtype_name,
                    "device": str(device),
                    "managed_params": managed_names[:8],
                }
            )

    report["trainable_flat_param_count"] = trainable_count
    report["flat_param_dtype_counts"] = dtype_counts
    report["meta_trainable_flat_param_count"] = meta_count
    report["non_fp32_trainable_flat_params"] = bad_params[:20]
    return report


def summarize_fsdp_master_weight_dtypes(model: nn.Module | None) -> dict[str, Any]:
    report = empty_fsdp_master_weight_report()
    if model is None:
        report["error"] = "No model was provided to the callback"
        return report

    try:
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception as exc:  # pragma: no cover - depends on torch build
        report["error"] = f"Could not import torch FSDP: {exc}"
        return report

    report["fsdp_available"] = True
    fsdp2_modules = [
        module for module in model.modules() if isinstance(module, FSDPModule)
    ]
    if fsdp2_modules:
        report = summarize_fsdp2_master_weight_dtypes(model, fsdp2_modules)
        report["fsdp_available"] = True
        return report

    try:
        fsdp_modules = list(FSDP.fsdp_modules(model))
    except Exception as exc:
        report["error"] = f"Could not enumerate FSDP modules: {exc}"
        return report

    return summarize_fsdp_master_weight_modules(fsdp_modules)


def summarize_fsdp2_master_weight_dtypes(
    model: nn.Module,
    fsdp2_modules: list[Any],
) -> dict[str, Any]:
    report = empty_fsdp_master_weight_report()
    report["fsdp_available"] = True
    report["fsdp_module_count"] = len(fsdp2_modules)
    report["fsdp_version"] = 2

    dtype_counts: dict[str, int] = {}
    bad_params: list[dict[str, Any]] = []
    meta_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        trainable_count += 1
        dtype_name = str(getattr(param, "dtype", "unknown"))
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
        device = getattr(param, "device", None)
        if device == torch.device("meta"):
            meta_count += 1
            continue
        if getattr(param, "dtype", None) != torch.float32:
            bad_params.append(
                {
                    "param_name": name,
                    "dtype": dtype_name,
                    "device": str(device),
                }
            )

    report["trainable_flat_param_count"] = trainable_count
    report["flat_param_dtype_counts"] = dtype_counts
    report["meta_trainable_flat_param_count"] = meta_count
    report["non_fp32_trainable_flat_params"] = bad_params[:20]
    return report


class FSDPMasterWeightsCallback(TrainerCallback):
    def __init__(self, require_fp32: bool = False) -> None:
        self.require_fp32 = require_fp32
        self._checked = False

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: nn.Module | None = None,
        optimizer: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if self._checked:
            return
        self._checked = True

        report = summarize_fsdp_master_weight_dtypes(model)
        report["optimizer_param_dtype_counts"] = summarize_optimizer_param_dtypes(
            optimizer
        )

        if state.is_world_process_zero:
            logger.info(
                "FSDP master weight dtype report: %s",
                json.dumps(jsonable(report), sort_keys=True),
            )

        if not self.require_fp32:
            return

        if not report["fsdp_available"]:
            raise RuntimeError(
                "--require_fp32_master_weights was set, but torch FSDP is not available"
            )
        if report["fsdp_module_count"] == 0:
            raise RuntimeError(
                "--require_fp32_master_weights was set, but the model has no FSDP modules"
            )
        if report["trainable_flat_param_count"] == 0:
            raise RuntimeError(
                "--require_fp32_master_weights was set, but no trainable FSDP flat "
                "parameters were found"
            )
        if report["meta_trainable_flat_param_count"]:
            raise RuntimeError(
                "--require_fp32_master_weights could not verify all trainable FSDP "
                "flat parameters because some are still on the meta device"
            )
        if report["non_fp32_trainable_flat_params"]:
            raise RuntimeError(
                "--require_fp32_master_weights expected all trainable FSDP flat "
                f"parameters to be torch.float32, got {json.dumps(jsonable(report), sort_keys=True)}"
            )

        if state.is_world_process_zero:
            logger.info("fp32_master_weights_verified=true")


def masked_mean(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


def masked_max(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1)
    floor = torch.finfo(hidden_states.dtype).min
    return hidden_states.masked_fill(~mask, floor).amax(dim=1)


class PooledRegressionModel(PreTrainedModel):
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Any,
        backbone: nn.Module | None = None,
        head_type: str | None = None,
    ) -> None:
        super().__init__(config)
        if backbone is None:
            from transformers import AutoModel

            backbone = AutoModel.from_config(config, trust_remote_code=True)
        self.backbone = backbone
        self.head_type = head_type or getattr(config, "head_type", "mean_pool")
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.config.head_type = self.head_type

        hidden_size = getattr(config, "hidden_size", None) or getattr(
            config, "d_model", None
        )
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model config")

        dropout_prob = (
            getattr(config, "classifier_dropout", None)
            if getattr(config, "classifier_dropout", None) is not None
            else getattr(config, "hidden_dropout_prob", 0.0)
        )
        self.dropout = nn.Dropout(float(dropout_prob or 0.0))

        head_dim = int(hidden_size)
        if self.head_type == "mean_max_pool":
            head_dim = int(hidden_size) * 2
        elif self.head_type == "attention_pool":
            self.attention_head = nn.Linear(int(hidden_size), 1)
        elif self.head_type == "cnn_pool":
            self.cnn_head = nn.Conv1d(
                int(hidden_size),
                int(hidden_size),
                kernel_size=7,
                padding=3,
            )
        elif self.head_type != "mean_pool":
            raise ValueError(f"Unsupported pooled head type: {self.head_type}")

        self.score = nn.Linear(head_dim, self.num_labels)
        self._init_new_head_weights()
        self._align_new_head_dtype()

    def _init_new_head_weights(self) -> None:
        initializer_range = float(getattr(self.config, "initializer_range", 0.02))
        modules = [self.score]
        if hasattr(self, "attention_head"):
            modules.append(self.attention_head)
        if hasattr(self, "cnn_head"):
            modules.append(self.cnn_head)
        for module in modules:
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _align_new_head_dtype(self) -> None:
        try:
            reference_param = next(self.backbone.parameters())
        except StopIteration:
            return
        modules = [self.score]
        if hasattr(self, "attention_head"):
            modules.append(self.attention_head)
        if hasattr(self, "cnn_head"):
            modules.append(self.cnn_head)
        for module in modules:
            module.to(device=reference_param.device, dtype=reference_param.dtype)

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        if not hasattr(self.backbone, "gradient_checkpointing_enable"):
            return
        try:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        except TypeError:
            self.backbone.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def pool_hidden_states(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.long,
                device=hidden_states.device,
            )

        if self.head_type == "mean_pool":
            return masked_mean(hidden_states, attention_mask)
        if self.head_type == "mean_max_pool":
            return torch.cat(
                [
                    masked_mean(hidden_states, attention_mask),
                    masked_max(hidden_states, attention_mask),
                ],
                dim=-1,
            )
        if self.head_type == "attention_pool":
            attention_logits = self.attention_head(hidden_states).squeeze(-1)
            attention_logits = attention_logits.masked_fill(
                ~attention_mask.to(dtype=torch.bool),
                torch.finfo(attention_logits.dtype).min,
            )
            attention_weights = torch.softmax(attention_logits, dim=-1).unsqueeze(-1)
            return (hidden_states * attention_weights).sum(dim=1)
        if self.head_type == "cnn_pool":
            mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            convolved = self.cnn_head((hidden_states * mask).transpose(1, 2))
            convolved = F.gelu(convolved.transpose(1, 2))
            return masked_max(convolved, attention_mask)
        raise ValueError(f"Unsupported pooled head type: {self.head_type}")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
            **kwargs,
        )
        hidden_states = outputs[0]
        pooled = self.pool_hidden_states(hidden_states, attention_mask)
        logits = self.score(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = regression_loss(logits, labels, "mse", 0.1, 0.1, 1.0, 1.0, 1.0)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


def pearson_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    predictions = predictions.float()
    labels = labels.float()
    if sample_weight is None:
        weights = torch.ones(
            (predictions.shape[0], 1),
            dtype=predictions.dtype,
            device=predictions.device,
        )
    else:
        weights = sample_weight.to(dtype=predictions.dtype, device=predictions.device)
        weights = weights.reshape(-1, 1)
    weight_sum = weights.sum(dim=0, keepdim=True).clamp_min(eps)
    predictions = (
        predictions - (weights * predictions).sum(dim=0, keepdim=True) / weight_sum
    )
    labels = labels - (weights * labels).sum(dim=0, keepdim=True) / weight_sum
    numerator = (weights * predictions * labels).sum(dim=0)
    pred_ss = (weights * predictions.square()).sum(dim=0)
    label_ss = (weights * labels.square()).sum(dim=0)
    denominator = torch.sqrt((pred_ss * label_ss).clamp_min(eps))
    corr = numerator / denominator
    return 1.0 - corr.mean()


def weighted_regression_loss(
    per_example_loss: torch.Tensor,
    sample_weight: torch.Tensor | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if per_example_loss.ndim > 1:
        per_example_loss = per_example_loss.mean(
            dim=tuple(range(1, per_example_loss.ndim))
        )
    if sample_weight is None:
        return per_example_loss.mean()
    weights = sample_weight.to(
        dtype=per_example_loss.dtype,
        device=per_example_loss.device,
    ).reshape(-1)
    return (per_example_loss * weights).sum() / weights.sum().clamp_min(eps)


def weighted_mse_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor | None,
) -> torch.Tensor:
    per_example = F.mse_loss(predictions, labels, reduction="none").mean(dim=-1)
    return weighted_regression_loss(per_example, sample_weight)


def weighted_huber_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor | None,
    huber_delta: float,
) -> torch.Tensor:
    per_example = F.huber_loss(
        predictions,
        labels,
        delta=huber_delta,
        reduction="none",
    ).mean(dim=-1)
    return weighted_regression_loss(per_example, sample_weight)


def weighted_moments(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor | None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = predictions.float()
    labels = labels.float()
    if sample_weight is None:
        weights = torch.ones(
            (predictions.shape[0], 1),
            dtype=predictions.dtype,
            device=predictions.device,
        )
    else:
        weights = sample_weight.to(dtype=predictions.dtype, device=predictions.device)
        weights = weights.reshape(-1, 1)
    weight_sum = weights.sum(dim=0, keepdim=True).clamp_min(eps)
    pred_mean = (weights * predictions).sum(dim=0, keepdim=True) / weight_sum
    label_mean = (weights * labels).sum(dim=0, keepdim=True) / weight_sum
    pred_centered = predictions - pred_mean
    label_centered = labels - label_mean
    pred_var = (weights * pred_centered.square()).sum(dim=0) / weight_sum.squeeze(0)
    label_var = (weights * label_centered.square()).sum(dim=0) / weight_sum.squeeze(0)
    covariance = (weights * pred_centered * label_centered).sum(
        dim=0
    ) / weight_sum.squeeze(0)
    return pred_mean.squeeze(0), label_mean.squeeze(0), pred_var, label_var, covariance


def concordance_corr_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_mean, label_mean, pred_var, label_var, covariance = weighted_moments(
        predictions,
        labels,
        sample_weight,
        eps=eps,
    )
    ccc = (2.0 * covariance) / (
        pred_var + label_var + (pred_mean - label_mean).square()
    ).clamp_min(eps)
    return 1.0 - ccc.mean()


def pearson_calibrated_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor | None,
    mean_weight: float,
    std_weight: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_mean, label_mean, pred_var, label_var, _ = weighted_moments(
        predictions,
        labels,
        sample_weight,
        eps=eps,
    )
    mean_penalty = (pred_mean - label_mean).square().mean()
    std_penalty = (
        (torch.sqrt(pred_var.clamp_min(eps)) - torch.sqrt(label_var.clamp_min(eps)))
        .square()
        .mean()
    )
    return (
        pearson_loss(predictions, labels, sample_weight, eps=eps)
        + mean_weight * mean_penalty
        + std_weight * std_penalty
    )


def regression_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str,
    pearson_loss_weight: float,
    ccc_loss_weight: float,
    calibration_mean_weight: float,
    calibration_std_weight: float,
    huber_delta: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    labels = labels.to(dtype=predictions.dtype)
    if loss_type == "mse":
        return weighted_mse_loss(predictions, labels, sample_weight)
    if loss_type == "huber":
        return weighted_huber_loss(predictions, labels, sample_weight, huber_delta)
    if loss_type == "pearson":
        return pearson_loss(predictions, labels, sample_weight)
    if loss_type == "mse_pearson":
        return weighted_mse_loss(
            predictions, labels, sample_weight
        ) + pearson_loss_weight * pearson_loss(predictions, labels, sample_weight)
    if loss_type == "ccc":
        return concordance_corr_loss(predictions, labels, sample_weight)
    if loss_type == "mse_ccc":
        return weighted_mse_loss(
            predictions, labels, sample_weight
        ) + ccc_loss_weight * concordance_corr_loss(predictions, labels, sample_weight)
    if loss_type == "pearson_calibrated":
        return pearson_calibrated_loss(
            predictions,
            labels,
            sample_weight,
            calibration_mean_weight,
            calibration_std_weight,
        )
    raise ValueError(f"Unsupported loss type: {loss_type}")


def categorical_loss(
    logits: torch.Tensor,
    category_labels: torch.Tensor,
    num_bins: int,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = logits.shape[0]
    logits = logits.float().reshape(batch_size, len(TARGET_NAMES), num_bins)
    category_labels = category_labels.to(dtype=torch.long, device=logits.device)
    per_target = F.cross_entropy(
        logits.reshape(-1, num_bins),
        category_labels.reshape(-1),
        reduction="none",
    ).reshape(batch_size, len(TARGET_NAMES))
    per_example = per_target.mean(dim=-1)
    return weighted_regression_loss(per_example, sample_weight)


def resolve_dna_kmer_id_range(tokenizer: Any) -> tuple[int, int]:
    dna_token_to_id = getattr(tokenizer, "dna_token_to_id", None)
    kmers = getattr(tokenizer, "kmers", None)
    if not dna_token_to_id or not kmers:
        raise ValueError(
            "Train token masking requires a Carbon DNA tokenizer with "
            "`dna_token_to_id` and `kmers` attributes."
        )

    kmer_ids = [int(dna_token_to_id[kmer]) for kmer in kmers]
    kmer_start = min(kmer_ids)
    kmer_end = max(kmer_ids) + 1
    if kmer_end - kmer_start != len(kmer_ids):
        raise ValueError("Carbon DNA k-mer token IDs are expected to be contiguous")
    return kmer_start, kmer_end


def resolve_oov_token_id(tokenizer: Any) -> int:
    oov_token_id = getattr(tokenizer, "oov_token_id", None)
    if oov_token_id is not None:
        return int(oov_token_id)
    dna_token_to_id = getattr(tokenizer, "dna_token_to_id", None)
    if dna_token_to_id and "<oov>" in dna_token_to_id:
        return int(dna_token_to_id["<oov>"])
    raise ValueError("Train token masking mode 'oov' requires a Carbon <oov> token")


class DNATokenMaskingDataCollator:
    def __init__(
        self,
        base_collator: Any,
        tokenizer: Any,
        *,
        mask_rate: float,
        mask_mode: str,
    ) -> None:
        self.base_collator = base_collator
        self.mask_rate = float(mask_rate)
        self.mask_mode = mask_mode
        self.kmer_start_id, self.kmer_end_id = resolve_dna_kmer_id_range(tokenizer)
        self.oov_token_id = (
            resolve_oov_token_id(tokenizer) if mask_mode == "oov" else None
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base_collator(features)
        if self.mask_rate <= 0.0 or "input_ids" not in batch:
            return batch

        input_ids = batch["input_ids"]
        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids)

        candidate_mask = (input_ids >= self.kmer_start_id) & (
            input_ids < self.kmer_end_id
        )
        if "attention_mask" in batch:
            candidate_mask = candidate_mask & batch["attention_mask"].to(
                dtype=torch.bool
            )

        mask = candidate_mask & (
            torch.rand(input_ids.shape, device=input_ids.device) < self.mask_rate
        )
        if not bool(mask.any()):
            batch["input_ids"] = input_ids
            return batch

        masked_input_ids = input_ids.clone()
        if self.mask_mode == "oov":
            masked_input_ids[mask] = int(self.oov_token_id)
        elif self.mask_mode == "random_kmer":
            replacements = torch.randint(
                self.kmer_start_id,
                self.kmer_end_id,
                input_ids.shape,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            masked_input_ids[mask] = replacements[mask]
        else:
            raise ValueError(f"Unsupported token mask mode: {self.mask_mode}")
        batch["input_ids"] = masked_input_ids
        return batch


class DeepstarrRegressionTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        train_data_collator: Any | None = None,
        prediction_mode: str = "regression",
        num_label_bins: int = 10,
        loss_type: str = "mse",
        pearson_loss_weight: float = 0.1,
        ccc_loss_weight: float = 0.1,
        calibration_mean_weight: float = 1.0,
        calibration_std_weight: float = 1.0,
        huber_delta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.train_data_collator = train_data_collator
        self.prediction_mode = prediction_mode
        self.num_label_bins = num_label_bins
        self.loss_type = loss_type
        self.pearson_loss_weight = pearson_loss_weight
        self.ccc_loss_weight = ccc_loss_weight
        self.calibration_mean_weight = calibration_mean_weight
        self.calibration_std_weight = calibration_std_weight
        self.huber_delta = huber_delta

    def get_train_dataloader(self) -> Any:
        if self.train_data_collator is None:
            return super().get_train_dataloader()

        data_collator = self.data_collator
        self.data_collator = self.train_data_collator
        try:
            return super().get_train_dataloader()
        finally:
            self.data_collator = data_collator

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        category_labels = inputs.pop("category_labels", None)
        sample_weight = inputs.pop("loss_weight", None)
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if self.prediction_mode == "categorical":
            if category_labels is None:
                raise ValueError("category_labels are required for categorical mode")
            loss = categorical_loss(
                logits,
                category_labels,
                self.num_label_bins,
                sample_weight=sample_weight,
            )
        else:
            loss = regression_loss(
                logits,
                labels,
                self.loss_type,
                self.pearson_loss_weight,
                self.ccc_loss_weight,
                self.calibration_mean_weight,
                self.calibration_std_weight,
                self.huber_delta,
                sample_weight=sample_weight,
            )
        return (loss, outputs) if return_outputs else loss


def init_trackio_if_requested(
    args: argparse.Namespace,
    output_dir: Path,
    run_metadata: dict[str, Any],
    is_main_process: bool,
) -> Any | None:
    if not trackio_enabled(args) or not is_main_process:
        return None

    try:
        import trackio
    except ImportError as exc:
        raise ImportError(
            "Trackio logging was requested, but the `trackio` package is not "
            "installed. Install `trackio>=0.25.1`, or disable Trackio."
        ) from exc
    try:
        trackio_version = package_version("trackio")
    except PackageNotFoundError as exc:
        raise ImportError("Could not determine installed Trackio version") from exc
    if parse_version_triplet(trackio_version) < MIN_TRACKIO_VERSION:
        raise ImportError(
            "Trackio logging requires `trackio>=0.25.1`; found "
            f"`trackio=={trackio_version}`."
        )

    run_name = args.trackio_run_name or args.run_name or output_dir.name
    group = args.trackio_group or f"{Path(str(args.model)).name}/{args.finetune_mode}"
    init_kwargs = {
        "project": args.trackio_project,
        "name": run_name,
        "group": group,
        "space_id": args.trackio_space_id,
        "config": jsonable(run_metadata),
        "resume": "allow",
    }
    if args.trackio_private is not None:
        init_kwargs["private"] = args.trackio_private
    if args.trackio_auto_log_gpu is not None:
        init_kwargs["auto_log_gpu"] = args.trackio_auto_log_gpu
        init_kwargs["gpu_log_interval"] = args.trackio_gpu_log_interval

    init_kwargs = {
        key: value for key, value in init_kwargs.items() if value is not None
    }
    logger.info(
        "Initializing Trackio project=%s run=%s space_id=%s",
        args.trackio_project,
        run_name,
        args.trackio_space_id,
    )
    trackio.init(**init_kwargs)
    return trackio


def freeze_lm_backbone(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad = False

    trainable_count = 0
    head_markers = ("classifier", "classification", "score", "regression", "head")
    for name, param in model.named_parameters():
        if any(marker in name.lower() for marker in head_markers):
            param.requires_grad = True
            trainable_count += param.numel()

    if trainable_count == 0:
        raise ValueError(
            "Could not identify a regression head to train. Expected parameter "
            "names containing one of: classifier, classification, score, "
            "regression, head."
        )


def prepare_fsdp2_eval_only_trainer(trainer: Any) -> bool:
    fsdp_plugin = getattr(
        getattr(trainer.accelerator, "state", None), "fsdp_plugin", None
    )
    is_fsdp2 = trainer.is_fsdp_enabled and getattr(fsdp_plugin, "fsdp_version", 1) == 2
    if not is_fsdp2 or len(getattr(trainer.accelerator, "_models", [])) != 0:
        return False

    trainer.create_optimizer()
    model, optimizer = trainer.accelerator.prepare(trainer.model, trainer.optimizer)
    trainer.model = model
    trainer.model_wrapped = model
    trainer.optimizer = optimizer
    return True


def torch_dtype_from_arg(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    import torch

    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from accelerate import PartialState
    from datasets import load_dataset
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    state = PartialState()

    logger.info("Loading dataset %s", args.dataset_name)
    raw_ds = load_dataset(args.dataset_name)
    train_raw = select_limit(raw_ds["train"], args.max_train_samples)
    validation_raw = select_limit(raw_ds["validation"], args.max_eval_samples)
    test_raw = select_limit(raw_ds["test"], args.max_eval_samples)
    label_transform = fit_label_transform(train_raw, args.label_transform)
    calibration = fit_log_calibration(train_raw, label_transform)
    category_metadata = (
        fit_category_metadata(train_raw, label_transform, args.num_label_bins)
        if args.prediction_mode == "categorical"
        else None
    )
    label_density_metadata: dict[str, Any] = {"mode": "none"}
    use_loss_weight = args.label_density_weighting != "none"
    if args.label_density_weighting == "lds":
        train_weights, label_density_metadata = fit_lds_label_density_weights(
            train_raw,
            num_bins=args.label_density_num_bins,
            kernel_size=args.label_density_kernel_size,
            kernel_sigma=args.label_density_kernel_sigma,
            power=args.label_density_power,
            min_weight=args.label_density_min_weight,
            max_weight=args.label_density_max_weight,
        )
        train_raw = train_raw.add_column("loss_weight", train_weights.tolist())
        validation_raw = validation_raw.add_column(
            "loss_weight",
            [1.0] * len(validation_raw),
        )
        test_raw = test_raw.add_column("loss_weight", [1.0] * len(test_raw))
        if state.is_main_process:
            logger.info(
                "Using LDS label-density weights: %s",
                json.dumps(jsonable(label_density_metadata), sort_keys=True),
            )

    unaugmented_train_size = len(train_raw)
    train_raw = augment_deepstarr_train_dataset(train_raw, args)
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

    def tokenize_fn(batch: dict[str, list[Any]]) -> dict[str, Any]:
        sequences = prepare_dna_sequences(batch["sequence"], args)
        tokenizer_kwargs = {}
        if args.dna_tokenization_mode == "auto_dna_tags":
            tokenizer_kwargs["auto_dna_tags"] = True
        tokenized = tokenizer(
            sequences,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            **tokenizer_kwargs,
        )
        tokenized["labels"] = label_values_from_batch(batch, label_transform)
        if category_metadata is not None:
            tokenized["category_labels"] = category_labels_from_batch(
                batch,
                label_transform,
                category_metadata,
            )
        if "loss_weight" in batch:
            tokenized["loss_weight"] = [
                float(weight) for weight in batch["loss_weight"]
            ]
        return tokenized

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "desc": "Tokenizing DeepSTARR",
    }
    if args.map_num_proc and args.map_num_proc > 1:
        map_kwargs["num_proc"] = args.map_num_proc

    tokenized_cache_dir = output_dir / "tokenized_cache" / tokenized_cache_tag(args)
    tokenized_cache_dir.mkdir(parents=True, exist_ok=True)

    with state.main_process_first():
        train_ds = train_raw.map(
            tokenize_fn,
            cache_file_name=str(tokenized_cache_dir / "train.arrow"),
            **map_kwargs,
        )
        validation_ds = validation_raw.map(
            tokenize_fn,
            cache_file_name=str(tokenized_cache_dir / "validation.arrow"),
            **map_kwargs,
        )
        test_ds = test_raw.map(
            tokenize_fn,
            cache_file_name=str(tokenized_cache_dir / "test.arrow"),
            **map_kwargs,
        )

    model_source = (
        args.checkpoint if args.eval_only else (args.init_from_checkpoint or args.model)
    )
    logger.info("Loading model from %s", model_source)
    torch_dtype = torch_dtype_from_arg(args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if not args.eval_only and args.revision and args.init_from_checkpoint is None:
        model_kwargs["revision"] = args.revision

    if args.prediction_mode == "categorical":
        label_names = [
            f"{target}_bin_{bin_idx}"
            for target in TARGET_NAMES
            for bin_idx in range(args.num_label_bins)
        ]
        label_config = {
            "num_labels": len(label_names),
            "problem_type": "single_label_classification",
            "id2label": dict(enumerate(label_names)),
            "label2id": {name: idx for idx, name in enumerate(label_names)},
        }
    else:
        label_config = {
            "num_labels": 2,
            "problem_type": "regression",
            "id2label": {0: "Dev_scaled", 1: "Hk_scaled"},
            "label2id": {"Dev_scaled": 0, "Hk_scaled": 1},
        }
    if args.head_type == "sequence":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            **label_config,
            **model_kwargs,
        )
    else:
        config_source = (
            model_source if args.eval_only or args.init_from_checkpoint else args.model
        )
        config_kwargs: dict[str, Any] = {
            "trust_remote_code": args.trust_remote_code,
            **label_config,
        }
        if not args.eval_only and args.revision and args.init_from_checkpoint is None:
            config_kwargs["revision"] = args.revision
        config = AutoConfig.from_pretrained(config_source, **config_kwargs)
        config.head_type = args.head_type
        if args.eval_only or args.init_from_checkpoint:
            model = PooledRegressionModel.from_pretrained(
                model_source,
                config=config,
                head_type=args.head_type,
                **model_kwargs,
            )
        else:
            backbone = AutoModel.from_pretrained(
                args.model,
                config=config,
                **model_kwargs,
            )
            model = PooledRegressionModel(
                config=config,
                backbone=backbone,
                head_type=args.head_type,
            )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if added_pad_token and len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.config.use_cache = False
    if args.finetune_mode == "frozen_lm":
        freeze_lm_backbone(model)

    parameter_counts = report_parameter_counts(model)
    if state.is_main_process:
        logger.info(
            "Parameters: total=%d trainable=%d (%.4f%%)",
            parameter_counts["total_parameters"],
            parameter_counts["trainable_parameters"],
            parameter_counts["trainable_percent"],
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * max(1, world_size)
    )

    run_metadata = {
        "model": args.model,
        "init_from_checkpoint": args.init_from_checkpoint,
        "revision": args.revision,
        "dataset_name": args.dataset_name,
        "finetune_mode": args.finetune_mode,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optim": args.optim,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "lr_scheduler_type": args.lr_scheduler_type,
        "lr_scheduler_kwargs": args.lr_scheduler_kwargs,
        "head_type": args.head_type,
        "prediction_mode": args.prediction_mode,
        "num_label_bins": args.num_label_bins,
        "category_metadata": category_metadata,
        "loss_type": args.loss_type,
        "pearson_loss_weight": args.pearson_loss_weight,
        "ccc_loss_weight": args.ccc_loss_weight,
        "calibration_mean_weight": args.calibration_mean_weight,
        "calibration_std_weight": args.calibration_std_weight,
        "huber_delta": args.huber_delta,
        "label_transform": args.label_transform,
        "label_transform_metadata": label_transform,
        "label_density_weighting": args.label_density_weighting,
        "label_density_metadata": label_density_metadata,
        "seed": args.seed,
        "max_length": args.max_length,
        "dna_tokenization_mode": args.dna_tokenization_mode,
        "truncate_dna_to_multiple": args.truncate_dna_to_multiple,
        "truncate_dna_side": args.truncate_dna_side,
        "kmer_phase_augment": args.kmer_phase_augment,
        "kmer_phase_augment_copies": args.kmer_phase_augment_copies,
        "kmer_phase_max_shift": args.kmer_phase_max_shift,
        "kmer_phase_output_length": args.kmer_phase_output_length,
        "kmer_phase_keep_original": args.kmer_phase_keep_original,
        "reverse_complement_augment": args.reverse_complement_augment,
        "reverse_complement_probability": args.reverse_complement_probability,
        "train_token_mask_rate": args.train_token_mask_rate,
        "train_token_mask_mode": args.train_token_mask_mode,
        "unaugmented_train_size": unaugmented_train_size,
        "augmented_train_size": augmented_train_size,
        "tokenized_cache_tag": tokenized_cache_tag(args),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world_size,
        "global_train_batch_size": global_batch_size,
        "torch_dtype": args.torch_dtype,
        "require_fp32_master_weights": args.require_fp32_master_weights,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "calibration": calibration,
        **parameter_counts,
    }
    if state.is_main_process:
        write_json(output_dir / "run_config.json", {**vars(args), **run_metadata})
    trackio_module = init_trackio_if_requested(
        args,
        output_dir,
        {**vars(args), **run_metadata},
        state.is_main_process,
    )
    effective_report_to = "none" if trackio_module is not None else args.report_to
    effective_eval_strategy = (
        "no" if args.eval_only or args.skip_validation else args.eval_strategy
    )
    effective_save_strategy = "no" if args.eval_only else args.save_strategy

    trainer_label_names = ["labels"]
    if args.prediction_mode == "categorical":
        trainer_label_names.append("category_labels")
    if use_loss_weight:
        trainer_label_names.append("loss_weight")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=not args.eval_only,
        do_eval=not args.skip_validation,
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
        max_grad_norm=args.max_grad_norm,
        eval_strategy=effective_eval_strategy,
        save_strategy=effective_save_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        load_best_model_at_end=(
            not args.eval_only
            and args.load_best_model_at_end
            and not args.skip_validation
        ),
        metric_for_best_model="pcc_mean",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        eval_accumulation_steps=args.eval_accumulation_steps,
        report_to=effective_report_to,
        run_name=args.run_name,
        remove_unused_columns=True,
        label_names=trainer_label_names,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
    )

    callbacks = []
    if args.require_fp32_master_weights and not args.eval_only:
        callbacks.append(FSDPMasterWeightsCallback(require_fp32=True))
    if trackio_module is not None:
        callbacks.append(TrackioMetricsCallback(trackio_module))
    if args.early_stopping_patience is not None and not args.eval_only:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    callbacks = callbacks or None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_data_collator = None
    if args.train_token_mask_rate > 0.0 and not args.eval_only:
        train_data_collator = DNATokenMaskingDataCollator(
            data_collator,
            tokenizer,
            mask_rate=args.train_token_mask_rate,
            mask_mode=args.train_token_mask_mode,
        )

    trainer = DeepstarrRegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=None if args.eval_only else train_ds,
        eval_dataset=None if args.skip_validation else validation_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_data_collator=train_data_collator,
        compute_metrics=make_compute_metrics(args.prediction_mode, category_metadata),
        callbacks=callbacks,
        prediction_mode=args.prediction_mode,
        num_label_bins=args.num_label_bins,
        loss_type=args.loss_type,
        pearson_loss_weight=args.pearson_loss_weight,
        ccc_loss_weight=args.ccc_loss_weight,
        calibration_mean_weight=args.calibration_mean_weight,
        calibration_std_weight=args.calibration_std_weight,
        huber_delta=args.huber_delta,
    )

    if args.eval_only:
        prepared_for_fsdp2 = prepare_fsdp2_eval_only_trainer(trainer)
        if prepared_for_fsdp2 and trainer.is_world_process_zero():
            logger.info("Prepared eval-only model with optimizer for FSDP2")

    if not args.eval_only:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if args.save_final_model:
            trainer.save_model(str(output_dir / "best_model"))

    trainer_best_checkpoint = trainer.state.best_model_checkpoint
    best_checkpoint = (
        str(output_dir / "best_model")
        if args.save_final_model
        else trainer_best_checkpoint
    )
    if args.eval_only:
        best_checkpoint = args.checkpoint
        trainer_best_checkpoint = args.checkpoint
    run_metadata["best_model_checkpoint"] = best_checkpoint
    run_metadata["trainer_best_model_checkpoint"] = trainer_best_checkpoint

    def write_split_results(
        split: str, tokenized_dataset: Any, raw_dataset: Any
    ) -> None:
        prediction_output = trainer.predict(
            tokenized_dataset,
            metric_key_prefix=split,
        )
        predictions = predictions_to_label_values(
            prediction_output.predictions,
            args.prediction_mode,
            category_metadata,
        )
        labels_scaled = normalize_label_ids(prediction_output.label_ids)
        metrics = {
            **run_metadata,
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
            if trackio_module is not None:
                trackio_module.log(
                    scalar_metrics(metrics),
                    step=int(trainer.state.global_step),
                )
            if args.save_predictions:
                write_predictions(
                    output_dir / f"{split}_predictions.jsonl",
                    raw_dataset,
                    predictions,
                    labels_scaled,
                    calibration,
                )

    if not args.skip_validation:
        write_split_results("validation", validation_ds, validation_raw)
    if not args.skip_test:
        write_split_results("test", test_ds, test_raw)
    if trackio_module is not None and trainer.is_world_process_zero():
        trackio_module.finish()


def write_predictions(
    path: Path,
    raw_dataset: Any,
    predictions: np.ndarray,
    labels_scaled: np.ndarray,
    calibration: dict[str, dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = raw_dataset["id"]
    raw_columns = [
        np.asarray(raw_dataset[raw_col], dtype=np.float64)
        for raw_col in RAW_LOG_COLUMNS
    ]
    pred_log = []
    for idx, target in enumerate(TARGET_NAMES):
        cal = calibration[target]
        pred_log.append(cal["slope"] * predictions[:, idx] + cal["intercept"])

    with path.open("w", encoding="utf-8") as f:
        for idx in range(predictions.shape[0]):
            row = {
                "id": ids[idx],
                "pred_dev_scaled": float(predictions[idx, 0]),
                "pred_hk_scaled": float(predictions[idx, 1]),
                "target_dev_scaled": float(labels_scaled[idx, 0]),
                "target_hk_scaled": float(labels_scaled[idx, 1]),
                "pred_dev_log2": float(pred_log[0][idx]),
                "pred_hk_log2": float(pred_log[1][idx]),
                "target_dev_log2": float(raw_columns[0][idx]),
                "target_hk_log2": float(raw_columns[1][idx]),
            }
            f.write(json.dumps(jsonable(row), sort_keys=True))
            f.write("\n")


if __name__ == "__main__":
    main()
