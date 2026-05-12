#!/bin/bash
#SBATCH -J seq_recovery_eval
#SBATCH -o slurm_logs/seq_recovery_eval_%j.out
#SBATCH -e slurm_logs/seq_recovery_eval_%j.err
# Long HF rollouts at higher gen_len can exceed six hours on a single GPU.
# Use a longer walltime so the sweep can complete without manual relaunches.
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Example usage:
# sbatch --export=MODEL=/path/to/carbon/model-or-hub-repo,REVISION=checkpoint-10000,DATA_TYPE=eukaryote sequence_recovery_eval.slurm
# Sweep usage:
#   Use evaluation/submit_sequence_recovery_gen_len_sweep.sh to submit one job per gen_len.

export HF_HUB_TRUST_REMOTE_CODE=1

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load cuda/12.9
fi

MODEL=${MODEL:-""}
MODEL_NAME=${MODEL_NAME:-""}
REVISION=${REVISION:-""}
DATA_TYPE=${DATA_TYPE:-"eukaryote"}
DATA_PATH=${DATA_PATH:-"hf://datasets/GenerTeam/sequence-recovery"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_results/sequence_recovery"}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-6144}
PROMPT_LEN_BP=${PROMPT_LEN_BP:-""}
GEN_LEN=${GEN_LEN:-5}
GEN_LEN_BP=${GEN_LEN_BP:-30}
ACCURACY_MODE=${ACCURACY_MODE:-"fixed_bp"}
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
  echo "MODEL env var is required (e.g., MODEL=/path/to/carbon/model-or-hub-repo)"
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="${REPO_ROOT}/evaluation"

# Torch 2.10.0+cu129 needs the CUDA libraries bundled in the eval venv to take
# precedence over the cluster's CUDA 12.1 libraries.
VENV_SITE_PACKAGES="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages"
if [[ -d "${VENV_SITE_PACKAGES}/nvidia/nvjitlink/lib" ]]; then
  export LD_LIBRARY_PATH="${VENV_SITE_PACKAGES}/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "${VENV_SITE_PACKAGES}/nvidia/cusparse/lib" ]]; then
  export LD_LIBRARY_PATH="${VENV_SITE_PACKAGES}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
fi

CMD=(uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/sequence_recovery_eval.py" \
  --model "$MODEL" \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --gen_len "$GEN_LEN" \
  --gen_len_bp "$GEN_LEN_BP" \
  --accuracy_mode "$ACCURACY_MODE" \
  --score_len_bp "$SCORE_LEN_BP" \
  --label_source "$LABEL_SOURCE" \
  --bp_per_token "$BP_PER_TOKEN" \
  --sample_seed "$SAMPLE_SEED")

if [[ -n "$PROMPT_LEN_BP" ]]; then
  CMD+=(--prompt_len_bp "$PROMPT_LEN_BP")
fi

if [[ -n "$MODEL_NAME" ]]; then
  CMD+=(--model_name "$MODEL_NAME")
fi

if [[ -n "$REVISION" ]]; then
  CMD+=(--revision "$REVISION")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max_samples "$MAX_SAMPLES")
fi

if [[ "$BF16" == "true" ]]; then
  CMD+=(--bf16)
fi

if [[ "$USE_EVO2" == "true" ]]; then
  CMD+=(--use_evo2)
fi

if [[ "$USE_VLLM" == "true" ]]; then
  CMD+=(--use_vllm)
fi

if [[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
  CMD+=(--vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi

if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  CMD+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
fi

if [[ -n "$VLLM_TENSOR_PARALLEL_SIZE" ]]; then
  CMD+=(--vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE")
fi

if [[ -n "$VLLM_DATA_PARALLEL_SIZE" ]]; then
  CMD+=(--vllm_data_parallel_size "$VLLM_DATA_PARALLEL_SIZE")
fi

if [[ "$USE_DNA_TAGS" == "true" ]]; then
  CMD+=(--use_dna_tags)
fi

if [[ "$NO_PREFIX" == "true" ]]; then
  CMD+=(--no_prefix)
fi

if [[ "$USE_SPECIES_TAGS" == "true" ]]; then
  CMD+=(--use_species_tags)
fi

if [[ "$UPCAST_LM_HEAD" == "true" ]]; then
  CMD+=(--upcast_lm_head)
fi

if [[ "$PUSH_TO_HUB" == "true" ]]; then
  CMD+=(--push_to_hub --hub_repo_id "$HUB_REPO_ID" --hub_repo_type "$HUB_REPO_TYPE")
fi

echo "Running: ${CMD[*]}"

# These wrapper variables are passed as CLI args above. Unset them before
# launching Python so vLLM does not interpret them as its own env vars.
unset VLLM_GPU_MEMORY_UTILIZATION
unset VLLM_MAX_MODEL_LEN
unset VLLM_TENSOR_PARALLEL_SIZE
unset VLLM_DATA_PARALLEL_SIZE

${CMD[@]}
