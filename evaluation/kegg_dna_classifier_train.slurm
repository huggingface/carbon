#!/bin/bash
#SBATCH -J kegg_dna_cls
#SBATCH -o slurm_logs/kegg_dna_cls_%j.out
#SBATCH -e slurm_logs/kegg_dna_cls_%j.err
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Example usage:
# sbatch --export=MODEL=/path/to/carbon/model-or-hub-repo,REVISION=checkpoint-10000 kegg_dna_classifier_train.slurm

set -euo pipefail

MODEL=${MODEL:-""}
REVISION=${REVISION:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_results/kegg_dna_classifier"}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_EPOCHS=${MAX_EPOCHS:-5}
MAX_STEPS=${MAX_STEPS:--1}
MAX_LENGTH=${MAX_LENGTH:-2048}
TRUNCATE_PER_SIDE=${TRUNCATE_PER_SIDE:-1024}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MERGE_VAL_TEST=${MERGE_VAL_TEST:-"true"}
BF16=${BF16:-"true"}
FP16=${FP16:-"false"}
EVAL_ONLY=${EVAL_ONLY:-"false"}
CHECKPOINT=${CHECKPOINT:-""}
PUSH_TO_HUB=${PUSH_TO_HUB:-"false"}
HUB_REPO_ID=${HUB_REPO_ID:-""}
HUB_REPO_TYPE=${HUB_REPO_TYPE:-"dataset"}

if [[ -z "$MODEL" ]]; then
  echo "MODEL env var is required (e.g., MODEL=/path/to/carbon/model-or-hub-repo)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD=(python "$SCRIPT_DIR/kegg_dna_classifier_train.py" \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_epochs "$MAX_EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --max_length "$MAX_LENGTH" \
  --truncate_dna_per_side "$TRUNCATE_PER_SIDE" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY")

if [[ -n "$REVISION" ]]; then
  CMD+=(--revision "$REVISION")
fi

if [[ "$MERGE_VAL_TEST" == "true" ]]; then
  CMD+=(--merge_val_test_set)
fi


if [[ "$BF16" == "true" ]]; then
  CMD+=(--bf16)
fi

if [[ "$FP16" == "true" ]]; then
  CMD+=(--fp16)
fi

if [[ "$EVAL_ONLY" == "true" ]]; then
  CMD+=(--eval_only)
fi

if [[ -n "$CHECKPOINT" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi

if [[ "$PUSH_TO_HUB" == "true" ]]; then
  CMD+=(--push_to_hub --hub_repo_id "$HUB_REPO_ID" --hub_repo_type "$HUB_REPO_TYPE")
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
