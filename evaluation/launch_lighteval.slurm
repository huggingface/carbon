#!/bin/bash
#SBATCH --job-name=carbon-eval
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=1-00:00:00

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sbatch evaluation/launch_lighteval.slurm --model MODEL [--revision REV] [--tp N] [--dp N]
    [--task TASK] [--output-dir DIR] [--results-org ORG] [-- ...extra lighteval args]

Examples:
  sbatch evaluation/launch_lighteval.slurm --model HuggingFaceTB/SmolLM3-3B-Base --revision main --tp 2 --dp 1
  sbatch evaluation/launch_lighteval.slurm --model HuggingFaceTB/SmolLM3-3B-Base --task mmlu_pro_biology_cf|0 --tp 2 --dp 1
USAGE
}

MODEL=""
MODEL_REVISION="main"
TP=2
DP=1
TASK="mmlu_pro_cf|0"
OUTPUT_DIR="."
RESULTS_ORG="hf-carbon"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --revision)
      MODEL_REVISION="$2"
      shift 2
      ;;
    --tp|--tensor-parallel-size)
      TP="$2"
      shift 2
      ;;
    --dp|--data-parallel-size)
      DP="$2"
      shift 2
      ;;
    --task|--tasks)
      TASK="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --results-org)
      RESULTS_ORG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Missing required --model argument." >&2
  usage
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${REPO_ROOT}/evaluation/.venv"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Expected virtual env at ${VENV_PATH}. Create it before running this job." >&2
  exit 1
fi

source "${VENV_PATH}/bin/activate"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face auth missing. Run 'hf auth login' after activating the venv." >&2
  exit 1
fi

MODEL_CONFIG="model_name=${MODEL},dtype=bfloat16,override_chat_template=false,tensor_parallel_size=${TP},data_parallel_size=${DP}"
if [[ -n "$MODEL_REVISION" ]]; then
  MODEL_CONFIG="${MODEL_CONFIG},revision=${MODEL_REVISION}"
fi

cd "$REPO_ROOT"

uv run --project "${REPO_ROOT}/evaluation" lighteval vllm \
  "$MODEL_CONFIG" \
  "$TASK" \
  --custom-tasks evaluation/lighteval_tasks.py \
  --output-dir "$OUTPUT_DIR" \
  --save-details \
  --push-to-hub \
  --results-org "$RESULTS_ORG" \
  "${EXTRA_ARGS[@]}"
