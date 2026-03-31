#!/bin/bash
#SBATCH -J tata_perturb_eval
#SBATCH -o slurm_logs/tata_perturb_eval_%j.out
#SBATCH -e slurm_logs/tata_perturb_eval_%j.err
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Example usage:
# sbatch --export=MODEL=/path/to/model,REVISION=checkpoint-10000 tata_perturbation_eval.slurm

set -euo pipefail

MODEL=${MODEL:-""}
REVISION=${REVISION:-""}
DATASET=${DATASET:-"hf-carbon/test_datasets"}
SUBSET=${SUBSET:-""}
SPLIT=${SPLIT:-"tata_perturbed"}
TASK=${TASK:-"tata_perturbation"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_results/tata_perturbation"}
MAX_LENGTH=${MAX_LENGTH:-2048}
BATCH_SIZE=${BATCH_SIZE:-16}
BF16=${BF16:-"true"}
USE_EVO2=${USE_EVO2:-"false"}
USE_DNA_TAGS=${USE_DNA_TAGS:-"false"}
NO_PREFIX=${NO_PREFIX:-"false"}
MODEL_NAME=${MODEL_NAME:-""}
PUSH_TO_HUB=${PUSH_TO_HUB:-"false"}
HUB_REPO_ID=${HUB_REPO_ID:-""}
HUB_REPO_TYPE=${HUB_REPO_TYPE:-"dataset"}

if [[ -z "$MODEL" ]]; then
  echo "MODEL env var is required (e.g., MODEL=/path/to/carbon/model-or-hub-repo)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD=(python "$SCRIPT_DIR/cds_half_shuffle_eval.py" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --subset "$SUBSET" \
  --split "$SPLIT" \
  --task "$TASK" \
  --output_dir "$OUTPUT_DIR" \
  --max_length "$MAX_LENGTH" \
  --batch_size "$BATCH_SIZE")

if [[ -n "$REVISION" ]]; then
  CMD+=(--revision "$REVISION")
fi

if [[ "$BF16" == "true" ]]; then
  CMD+=(--bf16)
fi

if [[ "$USE_EVO2" == "true" ]]; then
  CMD+=(--use_evo2)
fi

if [[ "$USE_DNA_TAGS" == "true" ]]; then
  CMD+=(--use_dna_tags)
fi

if [[ "$NO_PREFIX" == "true" ]]; then
  CMD+=(--no_prefix)
fi

if [[ -n "$MODEL_NAME" ]]; then
  CMD+=(--model_name "$MODEL_NAME")
fi

if [[ "$PUSH_TO_HUB" == "true" ]]; then
  CMD+=(--push_to_hub --hub_repo_id "$HUB_REPO_ID" --hub_repo_type "$HUB_REPO_TYPE")
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
