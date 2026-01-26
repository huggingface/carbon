#!/bin/bash
#SBATCH -J seq_recovery_eval
#SBATCH -o slurm_logs/seq_recovery_eval_%j.out
#SBATCH -e slurm_logs/seq_recovery_eval_%j.err
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Example usage:
# sbatch --export=MODEL=/path/to/carbon/model-or-hub-repo,REVISION=checkpoint-10000,DATA_TYPE=eukaryote sequence_recovery_eval.slurm

set -euo pipefail

MODEL=${MODEL:-""}
REVISION=${REVISION:-""}
DATA_TYPE=${DATA_TYPE:-"eukaryote"}
DATA_PATH=${DATA_PATH:-"hf://datasets/GenerTeam/sequence-recovery"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_results/sequence_recovery"}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-6144}
GEN_LEN=${GEN_LEN:-5}
GEN_LEN_BP=${GEN_LEN_BP:-30}
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
CMD=(python "$SCRIPT_DIR/sequence_recovery_eval.py" \
  --model "$MODEL" \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --gen_len "$GEN_LEN" \
  --gen_len_bp "$GEN_LEN_BP")

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
