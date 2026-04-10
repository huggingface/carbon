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

# Example usage:
# sbatch evaluation/launch_inspect.slurm --model hf-carbon/my-model --revision main --tp 2 --dp 1
# sbatch evaluation/launch_inspect.slurm --model hf-carbon/my-model --task lab_bench_cloning_scenarios --tp 4 --dp 1 -- --limit 1 --max-samples 1

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sbatch evaluation/launch_inspect.slurm --model MODEL [--revision REV] [--task TASK_SUFFIX]
    [--tp N] [--dp N] [--reasoning-parser PARSER] [--results-org ORG] [--log-dir DIR]
    [-- ...extra inspect args]

Examples:
  sbatch evaluation/launch_inspect.slurm --model hf-carbon/my-model --revision main --tp 2 --dp 1
  sbatch evaluation/launch_inspect.slurm --model hf-carbon/my-model --task lab_bench_cloning_scenarios --tp 4 --dp 1 --reasoning-parser qwen3
  sbatch evaluation/launch_inspect.slurm --model hf-carbon/my-model --task lab_bench_seqqa -- --limit 1 --max-samples 1
USAGE
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "Missing value for ${option}." >&2
    usage
    exit 1
  fi
}

slugify() {
  local value="$1"
  value="${value//[^[:alnum:]]/_}"
  while [[ "$value" == *__* ]]; do
    value="${value//__/_}"
  done
  value="${value##_}"
  value="${value%%_}"
  if [[ -z "$value" ]]; then
    value="unknown"
  fi
  printf '%s\n' "$value"
}

normalize_task_suffix() {
  local task_suffix="$1"
  task_suffix="${task_suffix#inspect_evals/}"
  task_suffix="${task_suffix#/}"
  printf '%s\n' "$task_suffix"
}

is_hf_repo_id() {
  local model="$1"
  case "$model" in
    /*|./*|../*|~/*)
      return 1
      ;;
  esac

  if [[ "$model" == */* ]] && [[ "$model" != */*/* ]]; then
    return 0
  fi

  return 1
}

resolve_local_basename() {
  local model_path="$1"
  if [[ "$model_path" == ~/* ]]; then
    model_path="${HOME}/${model_path#~/}"
  fi
  model_path="${model_path%/}"
  basename "$model_path"
}

detect_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    awk -F',' 'NF { print NF }' <<<"${CUDA_VISIBLE_DEVICES}"
    return 0
  fi

  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    if [[ "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "${SLURM_GPUS_ON_NODE}"
      return 0
    fi
    if [[ "${SLURM_GPUS_ON_NODE}" =~ gpu:([0-9]+) ]]; then
      printf '%s\n' "${BASH_REMATCH[1]}"
      return 0
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    return 0
  fi

  return 1
}

print_command() {
  local label="$1"
  shift
  printf '%s' "${label}:"
  printf ' %q' "$@"
  printf '\n'
}

has_option_arg() {
  local option="$1"
  shift
  local arg
  for arg in "$@"; do
    case "$arg" in
      "${option}"|${option}=*)
        return 0
        ;;
    esac
  done
  return 1
}

strip_option_from_inspect_cmd() {
  local option="$1"
  local filtered=()
  local skip_next=0
  local arg

  for arg in "${INSPECT_CMD[@]}"; do
    if (( skip_next )); then
      skip_next=0
      continue
    fi
    if [[ "$arg" == "$option" ]]; then
      skip_next=1
      continue
    fi
    filtered+=("$arg")
  done

  INSPECT_CMD=("${filtered[@]}")
}

MODEL=""
MODEL_REVISION="main"
TASK_SUFFIX="lab_bench_seqqa"
TP=2
DP=1
REASONING_PARSER=""
RESULTS_ORG="hf-carbon"
LOG_DIR=""
DEFAULT_MAX_CONNECTIONS=128
DEFAULT_MAX_SAMPLES=128
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      require_value "$1" "${2:-}"
      MODEL="$2"
      shift 2
      ;;
    --revision)
      require_value "$1" "${2:-}"
      MODEL_REVISION="$2"
      shift 2
      ;;
    --task|--tasks)
      require_value "$1" "${2:-}"
      TASK_SUFFIX="$2"
      shift 2
      ;;
    --tp|--tensor-parallel-size)
      require_value "$1" "${2:-}"
      TP="$2"
      shift 2
      ;;
    --dp|--data-parallel-size)
      require_value "$1" "${2:-}"
      DP="$2"
      shift 2
      ;;
    --reasoning-parser)
      require_value "$1" "${2:-}"
      REASONING_PARSER="$2"
      shift 2
      ;;
    --results-org)
      require_value "$1" "${2:-}"
      RESULTS_ORG="$2"
      shift 2
      ;;
    --log-dir)
      require_value "$1" "${2:-}"
      LOG_DIR="$2"
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

if ! [[ "$TP" =~ ^[0-9]+$ ]] || (( TP < 1 )); then
  echo "--tp must be a positive integer." >&2
  exit 1
fi

if ! [[ "$DP" =~ ^[0-9]+$ ]] || (( DP < 1 )); then
  echo "--dp must be a positive integer." >&2
  exit 1
fi

TASK_SUFFIX="$(normalize_task_suffix "$TASK_SUFFIX")"
if [[ -z "$TASK_SUFFIX" ]]; then
  echo "--task must not be empty." >&2
  exit 1
fi
TASK_PATH="inspect_evals/${TASK_SUFFIX}"

if is_hf_repo_id "$MODEL"; then
  MODEL_ORG="${MODEL%%/*}"
  MODEL_NAME="${MODEL##*/}"
else
  MODEL_ORG="local"
  MODEL_NAME="$(resolve_local_basename "$MODEL")"
fi

MODEL_ORG_SLUG="$(slugify "$MODEL_ORG")"
MODEL_NAME_SLUG="$(slugify "$MODEL_NAME")"
MODEL_REVISION_SLUG="$(slugify "$MODEL_REVISION")"
SPACE_NAME="inspect-logs_${MODEL_ORG_SLUG}_${MODEL_NAME_SLUG}_${MODEL_REVISION_SLUG}"
MODEL_SHA=""
JOB_NAME="carbon-eval_${MODEL_NAME_SLUG}_${TASK_SUFFIX}"

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  if ! scontrol update JobId="${SLURM_JOB_ID}" JobName="${JOB_NAME}" >/dev/null; then
    echo "Warning: failed to update Slurm job name to ${JOB_NAME}." >&2
  fi
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${REPO_ROOT}/evaluation/.venv"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Expected virtual env at ${VENV_PATH}. Create it before running this job." >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/12.9
fi

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

source "${VENV_PATH}/bin/activate"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face auth missing. Run 'hf auth login' after activating the venv." >&2
  exit 1
fi

if is_hf_repo_id "$MODEL"; then
  if MODEL_SHA="$(
    MODEL="$MODEL" MODEL_REVISION="$MODEL_REVISION" uv run --project "${REPO_ROOT}/evaluation" python - <<'PY'
import os

from huggingface_hub import HfApi

model_info = HfApi().model_info(
    os.environ["MODEL"], revision=os.environ["MODEL_REVISION"]
)
print(model_info.sha or "")
PY
  )"; then
    MODEL_SHA="${MODEL_SHA//$'\r'/}"
    MODEL_SHA="${MODEL_SHA//$'\n'/}"
    if [[ -z "$MODEL_SHA" ]]; then
      echo "Warning: resolved an empty SHA for ${MODEL}@${MODEL_REVISION}." >&2
    fi
  else
    echo "Warning: failed to resolve Hub SHA for ${MODEL}@${MODEL_REVISION}; continuing with revision metadata only." >&2
    MODEL_SHA=""
  fi
fi

AVAILABLE_GPUS="$(detect_gpu_count || true)"
REQUIRED_GPUS=$((TP * DP))
if [[ -n "$AVAILABLE_GPUS" ]] && (( REQUIRED_GPUS > AVAILABLE_GPUS )); then
  echo "Requested TP*DP=${REQUIRED_GPUS}, but only detected ${AVAILABLE_GPUS} GPUs in this allocation." >&2
  echo "Resubmit with a larger GPU allocation or reduce --tp/--dp." >&2
  exit 1
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${REPO_ROOT}/results/inspect-logs/${MODEL_ORG_SLUG}_${MODEL_NAME_SLUG}_${MODEL_REVISION_SLUG}"
fi

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

METADATA_ARGS=(
  --metadata "model_repo=${MODEL}"
  --metadata "model_revision=${MODEL_REVISION}"
)

if [[ -n "$MODEL_SHA" ]]; then
  METADATA_ARGS+=(--metadata "model_sha=${MODEL_SHA}")
fi

INSPECT_CMD=(
  uv run --project "${REPO_ROOT}/evaluation" inspect eval "${TASK_PATH}"
  --model "vllm/${MODEL}"
  -M "tensor_parallel_size=${TP}"
  -M "data_parallel_size=${DP}"
  -M "revision=${MODEL_REVISION}"
  "${METADATA_ARGS[@]}"
  --max-connections "${DEFAULT_MAX_CONNECTIONS}"
  --max-samples "${DEFAULT_MAX_SAMPLES}"
  --log-dir "${LOG_DIR}"
  --log-level info
  --display plain
)

if [[ -n "$REASONING_PARSER" ]]; then
  INSPECT_CMD+=(-M "reasoning_parser=${REASONING_PARSER}")
fi

if has_option_arg --max-connections "${EXTRA_ARGS[@]}"; then
  strip_option_from_inspect_cmd --max-connections
fi

if has_option_arg --max-samples "${EXTRA_ARGS[@]}"; then
  strip_option_from_inspect_cmd --max-samples
fi

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  INSPECT_CMD+=("${EXTRA_ARGS[@]}")
fi

BUNDLE_CMD=(
  uv run --project "${REPO_ROOT}/evaluation" inspect view bundle
  --log-dir "${LOG_DIR}"
  --output-dir "hf/${RESULTS_ORG}/${SPACE_NAME}"
  --overwrite
)

print_command "Running eval" "${INSPECT_CMD[@]}"
"${INSPECT_CMD[@]}"

print_command "Pushing viewer" "${BUNDLE_CMD[@]}"
"${BUNDLE_CMD[@]}"
