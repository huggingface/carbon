#!/bin/bash
#SBATCH --job-name=carbon-eval
#SBATCH --qos=high
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=2-00:00:00

# Example usage:
# sbatch evaluation/launch_inspect_api.slurm --model openai/gpt-5.4 --task lab_bench_seqqa --reasoning-effort high
# sbatch evaluation/launch_inspect_api.slurm --model openai/gpt-5.4 -- --max-samples 128

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sbatch evaluation/launch_inspect_api.slurm --model PROVIDER/MODEL [--task TASK1 TASK2 ...]
    [--reasoning-effort LEVEL] [--results-org ORG] [--log-dir DIR]
    [-- ...extra inspect args]

Examples:
  sbatch evaluation/launch_inspect_api.slurm --model openai/gpt-5.4 --task lab_bench_seqqa --reasoning-effort high
  sbatch evaluation/launch_inspect_api.slurm --model openai/gpt-5.4 --task lab_bench_litqa lab_bench_suppqa lab_bench_figqa
  sbatch evaluation/launch_inspect_api.slurm --model openai/gpt-5.4 -- --max-samples 128
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
TASK_SUFFIXES=()
REASONING_EFFORT=""
RESULTS_ORG="hf-carbon"
LOG_DIR=""
DEFAULT_MAX_CONNECTIONS=32
DEFAULT_MAX_SAMPLES=64
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      require_value "$1" "${2:-}"
      MODEL="$2"
      shift 2
      ;;
    --task|--tasks)
      require_value "$1" "${2:-}"
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TASK_SUFFIXES+=("$1")
        shift
      done
      ;;
    --reasoning-effort)
      require_value "$1" "${2:-}"
      REASONING_EFFORT="$2"
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

if [[ "$MODEL" != */* ]]; then
  echo "--model must be an Inspect provider/model identifier such as openai/gpt-5.4." >&2
  exit 1
fi

if (( ${#TASK_SUFFIXES[@]} == 0 )); then
  TASK_SUFFIXES=("lab_bench_seqqa")
fi

TASK_PATHS=()
for i in "${!TASK_SUFFIXES[@]}"; do
  TASK_SUFFIXES[$i]="$(normalize_task_suffix "${TASK_SUFFIXES[$i]}")"
  if [[ -z "${TASK_SUFFIXES[$i]}" ]]; then
    echo "--task must not be empty." >&2
    exit 1
  fi
  TASK_PATHS+=("inspect_evals/${TASK_SUFFIXES[$i]}")
done

MODEL_PROVIDER="${MODEL%%/*}"
MODEL_NAME="${MODEL#*/}"
MODEL_PROVIDER_SLUG="$(slugify "$MODEL_PROVIDER")"
MODEL_NAME_SLUG="$(slugify "$MODEL_NAME")"
SPACE_NAME="inspect-logs_${MODEL_PROVIDER_SLUG}_${MODEL_NAME_SLUG}"
JOB_NAME="carbon-eval-${MODEL_NAME_SLUG}"

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

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

source "${VENV_PATH}/bin/activate"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face auth missing. Run 'hf auth login' after activating the venv." >&2
  exit 1
fi

case "$MODEL_PROVIDER" in
  openai)
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "OPENAI_API_KEY must be set to run ${MODEL} via Inspect." >&2
      exit 1
    fi
    ;;
esac

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${REPO_ROOT}/results/inspect-logs/${MODEL_PROVIDER_SLUG}_${MODEL_NAME_SLUG}"
fi

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

METADATA_ARGS=(
  --metadata "api_model=${MODEL}"
)

if [[ -n "$REASONING_EFFORT" ]]; then
  METADATA_ARGS+=(--metadata "reasoning_effort=${REASONING_EFFORT}")
fi

for TASK_PATH in "${TASK_PATHS[@]}"; do
  INSPECT_CMD=(
    uv run --project "${REPO_ROOT}/evaluation" inspect eval "${TASK_PATH}"
    --model "${MODEL}"
    "${METADATA_ARGS[@]}"
    --max-connections "${DEFAULT_MAX_CONNECTIONS}"
    --max-samples "${DEFAULT_MAX_SAMPLES}"
    --no-fail-on-error
    --log-dir "${LOG_DIR}"
    --display plain
  )

  if [[ -n "$REASONING_EFFORT" ]]; then
    INSPECT_CMD+=(--reasoning-effort "${REASONING_EFFORT}")
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

  print_command "Running eval" "${INSPECT_CMD[@]}"
  "${INSPECT_CMD[@]}"
done

BUNDLE_CMD=(
  uv run --project "${REPO_ROOT}/evaluation" inspect view bundle
  --log-dir "${LOG_DIR}"
  --output-dir "hf/${RESULTS_ORG}/${SPACE_NAME}"
  --overwrite
)

print_command "Pushing viewer" "${BUNDLE_CMD[@]}"
"${BUNDLE_CMD[@]}"
