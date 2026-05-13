#!/usr/bin/env bash
#SBATCH --job-name=carbon-deepstarr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SLURM_NNODES="${SLURM_NNODES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-$((SLURM_NNODES * GPUS_PER_NODE))}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-finetuning/deepstarr/fsdp2_carbon.yaml}"
MODEL="${MODEL:-hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1}"
RUN_NAME="${RUN_NAME:-carbon-3b-deepstarr-full-ft}"
OUTPUT_DIR="${OUTPUT_DIR:-scratch/deepstarr/${RUN_NAME}}"
TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-}"
TRACKIO_PROJECT="${TRACKIO_PROJECT:-deepstarr-regression}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TRACKIO_SPACE_ID
export TRACKIO_PROJECT

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  finetuning/deepstarr/deepstarr_regression_train.py \
  --model "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --trackio_run_name "${RUN_NAME}" \
  --finetune_mode full_finetune \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --loss_type "${LOSS_TYPE:-pearson}" \
  --label_transform "${LABEL_TRANSFORM:-dataset_scaled}" \
  --head_type "${HEAD_TYPE:-sequence}" \
  --dna_tokenization_mode "${DNA_TOKENIZATION_MODE:-auto_dna_tags}" \
  --truncate_dna_to_multiple "${TRUNCATE_DNA_TO_MULTIPLE:-6}" \
  --max_length "${MAX_LENGTH:-512}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-5}" \
  --eval_strategy steps \
  --save_strategy steps \
  --eval_steps "${EVAL_STEPS:-8555}" \
  --save_steps "${SAVE_STEPS:-8555}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-8}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --torch_dtype "${TORCH_DTYPE:-float32}" \
  --bf16 \
  --require_fp32_master_weights \
  --gradient_checkpointing \
  --attn_implementation "${ATTN_IMPLEMENTATION:-kernels-community/flash-attn3}" \
  --trackio
