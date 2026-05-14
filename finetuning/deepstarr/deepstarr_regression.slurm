#!/usr/bin/env bash
#SBATCH --job-name=carbon-deepstarr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
SLURM_NNODES="${SLURM_NNODES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-$((SLURM_NNODES * GPUS_PER_NODE))}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-finetuning/deepstarr/fsdp2_carbon.yaml}"
MODEL="${MODEL:-HuggingFaceBio/Carbon-3B}"
RUN_NAME="${RUN_NAME:-carbon-3b-deepstarr-regression-train}"
OUTPUT_DIR="${OUTPUT_DIR:-scratch/deepstarr/${RUN_NAME}}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

EXTRA_ARGS=()
if [[ -n "${EVAL_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--eval_steps "${EVAL_STEPS}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  finetuning/deepstarr/deepstarr_train.py \
  --model "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-reduce_lr_on_plateau}" \
  --lr_scheduler_kwargs "${LR_SCHEDULER_KWARGS:-{\"mode\":\"max\",\"factor\":0.5,\"patience\":2,\"threshold\":0.0001}}" \
  --dna_mode "${DNA_MODE:-auto_dna_tags}" \
  --max_length "${MAX_LENGTH:-512}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-2.5}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-32}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-64}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --bf16 \
  --gradient_checkpointing \
  --attn_implementation "${ATTN_IMPLEMENTATION:-kernels-community/flash-attn3}" \
  "${EXTRA_ARGS[@]}"
