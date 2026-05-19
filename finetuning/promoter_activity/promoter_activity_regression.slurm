#!/usr/bin/env bash
#SBATCH --job-name=carbon-promoter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
set -euo pipefail

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-finetuning/deepstarr/fsdp2_carbon.yaml}"
ACCELERATE="${ACCELERATE:-.venv/bin/accelerate}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
SLURM_NNODES="${SLURM_NNODES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-$((SLURM_NNODES * GPUS_PER_NODE))}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
MODEL="${MODEL:-HuggingFaceBio/Carbon-500M-remote}"
RUN_NAME="${RUN_NAME:-carbon-500m-promoter-pearson-huber-200k}"
OUTPUT_DIR="${OUTPUT_DIR:-scratch/promoter_activity/${RUN_NAME}}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

EXTRA_ARGS=()
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
if [[ -n "${ATTN_IMPLEMENTATION:-}" ]]; then
  EXTRA_ARGS+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
fi
if [[ "${SKIP_TEST:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_test)
fi
if [[ "${SAVE_PREDICTIONS:-true}" == "false" ]]; then
  EXTRA_ARGS+=(--no-save_predictions)
fi
if [[ "${SAVE_MODEL:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--save_model)
fi

"${ACCELERATE}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  finetuning/promoter_activity/promoter_activity_train.py \
  --model "${MODEL}" \
  --dataset_name "${DATASET_NAME:-HuggingFaceBio/random-promoter-dream-2022}" \
  --dataset_config "${DATASET_CONFIG:-supervised}" \
  --dataset_revision "${DATASET_REVISION:-5b18e5067fd20e589f698a4406ca6e4f2344d68f}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --dna_mode "${DNA_MODE:-auto_dna_tags}" \
  --max_length "${MAX_LENGTH:-512}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1.0}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-16}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-32}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --huber_delta "${HUBER_DELTA:-1.0}" \
  --huber_loss_weight "${HUBER_LOSS_WEIGHT:-0.2}" \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --bf16 \
  --gradient_checkpointing \
  "${EXTRA_ARGS[@]}"
