#!/usr/bin/env bash
#SBATCH --job-name=carbon-malinois
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
set -euo pipefail

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SLURM_NNODES="${SLURM_NNODES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-$((SLURM_NNODES * GPUS_PER_NODE))}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-finetuning/deepstarr/fsdp2_carbon.yaml}"
ACCELERATE="${ACCELERATE:-.venv/bin/accelerate}"
MODEL="${MODEL:-HuggingFaceBio/Carbon-3B}"
RUN_NAME="${RUN_NAME:-carbon-3b-malinois-mse}"
OUTPUT_DIR="${OUTPUT_DIR:-scratch/malinois/${RUN_NAME}}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ -x ".venv/bin/hf" ]]; then
  .venv/bin/hf auth whoami || true
fi

EXTRA_ARGS=()
if [[ -n "${SOURCE_URL:-}" ]]; then
  EXTRA_ARGS+=(--source_url "${SOURCE_URL}")
fi
if [[ -n "${EVAL_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--eval_steps "${EVAL_STEPS}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_EVAL_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${LR_SCHEDULER_KWARGS:-}" ]]; then
  EXTRA_ARGS+=(--lr_scheduler_kwargs "${LR_SCHEDULER_KWARGS}")
fi
if [[ "${NO_TRAIN_REVERSE_COMPLEMENT:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--no-train_reverse_complement)
fi
if [[ "${NO_DUPLICATE_HIGH_ACTIVITY:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--no-duplicate_high_activity)
fi
if [[ "${NO_RC_EVAL_AVERAGE:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--no-rc_eval_average)
fi
if [[ "${SKIP_TEST:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_test)
fi
if [[ "${SAVE_MODEL:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--save_model)
fi

"${ACCELERATE}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  finetuning/malinois/malinois_train.py \
  --model "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-reduce_lr_on_plateau}" \
  --dna_mode "${DNA_MODE:-auto_dna_tags}" \
  --max_length "${MAX_LENGTH:-512}" \
  --metric_se_threshold "${METRIC_SE_THRESHOLD:-1.0}" \
  --high_activity_threshold "${HIGH_ACTIVITY_THRESHOLD:-0.5}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-32}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-8}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --torch_dtype "${TORCH_DTYPE:-float32}" \
  --bf16 \
  --gradient_checkpointing \
  --attn_implementation "${ATTN_IMPLEMENTATION:-kernels-community/flash-attn3}" \
  "${EXTRA_ARGS[@]}"
