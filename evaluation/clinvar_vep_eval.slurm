#!/bin/bash
#SBATCH -J clinvar_vep_eval
#SBATCH -o slurm_logs/clinvar_vep_eval_%j.out
#SBATCH -e slurm_logs/clinvar_vep_eval_%j.err
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Example usage:
# sbatch --export=MODEL=/path/to/carbon/model-or-hub-repo,REVISION=checkpoint-10000,CONTEXT_LEN=96000 clinvar_vep_eval.slurm

set -euo pipefail

MODEL=${MODEL:-""}
REVISION=${REVISION:-""}
HG38_PATH=${HG38_PATH:-"hf://datasets/GenerTeam/variant-effect-prediction/hg38.parquet"}
CLINVAR_PATH=${CLINVAR_PATH:-"hf://datasets/GenerTeam/variant-effect-prediction/ClinVar_VEP_results.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_results/clinvar_vep"}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_PROCESSES=${NUM_PROCESSES:-32}
CONTEXT_LEN=${CONTEXT_LEN:-96000}
BF16=${BF16:-"true"}
USE_EVO2=${USE_EVO2:-"false"}
PUSH_TO_HUB=${PUSH_TO_HUB:-"false"}
HUB_REPO_ID=${HUB_REPO_ID:-""}
HUB_REPO_TYPE=${HUB_REPO_TYPE:-"dataset"}

if [[ -z "$MODEL" ]]; then
  echo "MODEL env var is required (e.g., MODEL=/path/to/carbon/model-or-hub-repo)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD=(python "$SCRIPT_DIR/clinvar_vep_eval.py" \
  --model "$MODEL" \
  --hg38_path "$HG38_PATH" \
  --clinvar_path "$CLINVAR_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_processes "$NUM_PROCESSES" \
  --context_length "$CONTEXT_LEN")

if [[ -n "$REVISION" ]]; then
  CMD+=(--revision "$REVISION")
fi

if [[ "$BF16" == "true" ]]; then
  CMD+=(--bf16)
fi

if [[ "$USE_EVO2" == "true" ]]; then
  CMD+=(--use_evo2)
fi

if [[ "$PUSH_TO_HUB" == "true" ]]; then
  CMD+=(--push_to_hub --hub_repo_id "$HUB_REPO_ID" --hub_repo_type "$HUB_REPO_TYPE")
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
