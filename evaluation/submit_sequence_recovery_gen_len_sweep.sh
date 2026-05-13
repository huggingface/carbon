#!/bin/bash

set -euo pipefail

: <<'USAGE'
Usage examples:

  3B hybrid model:
    MODEL=hf-carbon/carbon-3B-600B-dna-generv2-fp32-lmhead \
    MODEL_NAME=Carbon-3B-600B-dna-generv2-fp32-lmhead \
    USE_DNA_TAGS=true \
    GEN_LENS="5 10 20 40 80 160 320 640 1280 2560" \
    ACCURACY_MODE=prediction_length \
    BASE_OUTPUT_DIR=./eval_results/sequence_recovery_long_rollouts_pow2 \
    evaluation/submit_sequence_recovery_gen_len_sweep.sh

  8B hybrid model:
    MODEL=hf-carbon/carbon-8B-600B-dna-fp32-lmhead \
    MODEL_NAME=Carbon-8B-600B-dna-fp32-lmhead \
    USE_DNA_TAGS=true \
    BATCH_SIZE=8 \
    GEN_LENS="5 10 20 40 80 160 320 640 1280 2560" \
    ACCURACY_MODE=prediction_length \
    BASE_OUTPUT_DIR=./eval_results/sequence_recovery_long_rollouts_pow2 \
    evaluation/submit_sequence_recovery_gen_len_sweep.sh

  Evo2 7B model:
    MODEL=evo2_7b \
    MODEL_NAME=Evo2-7B \
    USE_EVO2=true \
    BATCH_SIZE=1 \
    GEN_LENS="5 10 20 40 80 160 320 640 1280 2560" \
    ACCURACY_MODE=prediction_length \
    BASE_OUTPUT_DIR=./eval_results/sequence_recovery_long_rollouts_pow2 \
    evaluation/submit_sequence_recovery_gen_len_sweep.sh
USAGE

MODEL=${MODEL:-""}
MODEL_NAME=${MODEL_NAME:-""}
REVISION=${REVISION:-""}
DATA_TYPE=${DATA_TYPE:-"eukaryote"}
DATA_PATH=${DATA_PATH:-"hf://datasets/GenerTeam/sequence-recovery"}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"./eval_results/sequence_recovery_gen_len"}
GEN_LENS=${GEN_LENS:-"5 10 20 40 80 160 320 640 1280 2560"}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-6144}
GEN_LEN_BP_WAS_SET=${GEN_LEN_BP+x}
GEN_LEN_BP=${GEN_LEN_BP:-30}
ACCURACY_MODE=${ACCURACY_MODE:-"prediction_length"}
SCORE_LEN_BP=${SCORE_LEN_BP:-30}
LABEL_SOURCE=${LABEL_SOURCE:-"auto"}
BP_PER_TOKEN=${BP_PER_TOKEN:-6}
BF16=${BF16:-"true"}
USE_EVO2=${USE_EVO2:-"false"}
USE_VLLM=${USE_VLLM:-"false"}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-""}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-""}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-""}
VLLM_DATA_PARALLEL_SIZE=${VLLM_DATA_PARALLEL_SIZE:-""}
USE_DNA_TAGS=${USE_DNA_TAGS:-"false"}
NO_PREFIX=${NO_PREFIX:-"false"}
USE_SPECIES_TAGS=${USE_SPECIES_TAGS:-"false"}
UPCAST_LM_HEAD=${UPCAST_LM_HEAD:-"false"}
MAX_SAMPLES=${MAX_SAMPLES:-""}
SAMPLE_SEED=${SAMPLE_SEED:-0}
PUSH_TO_HUB=${PUSH_TO_HUB:-"false"}
HUB_REPO_ID=${HUB_REPO_ID:-""}
HUB_REPO_TYPE=${HUB_REPO_TYPE:-"dataset"}

if [[ -z "$MODEL" ]]; then
  echo "MODEL env var is required"
  exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME="${MODEL##*/}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/sequence_recovery_eval.slurm"

for gen_len in $GEN_LENS; do
  run_output_dir="${BASE_OUTPUT_DIR}/${MODEL_NAME}/${DATA_TYPE}/gen_len_${gen_len}"
  run_gen_len_bp="$GEN_LEN_BP"
  if [[ "$USE_EVO2" == "true" && -z "$GEN_LEN_BP_WAS_SET" ]]; then
    run_gen_len_bp=$((gen_len * BP_PER_TOKEN))
  fi
  export_vars=(
    "MODEL=$MODEL"
    "MODEL_NAME=$MODEL_NAME"
    "REVISION=$REVISION"
    "DATA_TYPE=$DATA_TYPE"
    "DATA_PATH=$DATA_PATH"
    "OUTPUT_DIR=$run_output_dir"
    "BATCH_SIZE=$BATCH_SIZE"
    "MAX_SEQ_LEN=$MAX_SEQ_LEN"
    "GEN_LEN=$gen_len"
    "GEN_LEN_BP=$run_gen_len_bp"
    "ACCURACY_MODE=$ACCURACY_MODE"
    "SCORE_LEN_BP=$SCORE_LEN_BP"
    "LABEL_SOURCE=$LABEL_SOURCE"
    "BP_PER_TOKEN=$BP_PER_TOKEN"
    "BF16=$BF16"
    "USE_EVO2=$USE_EVO2"
    "USE_VLLM=$USE_VLLM"
    "VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION"
    "VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN"
    "VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE"
    "VLLM_DATA_PARALLEL_SIZE=$VLLM_DATA_PARALLEL_SIZE"
    "USE_DNA_TAGS=$USE_DNA_TAGS"
    "NO_PREFIX=$NO_PREFIX"
    "USE_SPECIES_TAGS=$USE_SPECIES_TAGS"
    "UPCAST_LM_HEAD=$UPCAST_LM_HEAD"
    "MAX_SAMPLES=$MAX_SAMPLES"
    "SAMPLE_SEED=$SAMPLE_SEED"
    "PUSH_TO_HUB=$PUSH_TO_HUB"
    "HUB_REPO_ID=$HUB_REPO_ID"
    "HUB_REPO_TYPE=$HUB_REPO_TYPE"
  )

  export_string=$(IFS=,; echo "${export_vars[*]}")
  echo "Submitting gen_len=${gen_len}, gen_len_bp=${run_gen_len_bp} -> ${run_output_dir}"
  sbatch --export="$export_string" "$SLURM_SCRIPT"
done
