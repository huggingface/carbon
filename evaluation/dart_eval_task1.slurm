#!/bin/bash
#SBATCH -J dart_eval_t1
#SBATCH -o slurm_logs/dart_eval_t1_%j.out
#SBATCH -e slurm_logs/dart_eval_t1_%j.err
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Example usage:
# sbatch --export=DART_WORK_DIR=/path/to/dart_work,MODEL=nucleotide_transformer,DO_DATASET_GEN=true,DO_ZERO_SHOT=true dart_eval_task1.slurm

set -euo pipefail

DART_WORK_DIR=${DART_WORK_DIR:-""}
DART_EVAL_DIR=${DART_EVAL_DIR:-""}
MODEL=${MODEL:-"nucleotide_transformer"}
DO_DATASET_GEN=${DO_DATASET_GEN:-"true"}
DO_ZERO_SHOT=${DO_ZERO_SHOT:-"true"}
CCRE_BED=${CCRE_BED:-""}
PROCESSED_TSV=${PROCESSED_TSV:-""}

if [[ -z "$DART_WORK_DIR" ]]; then
  echo "DART_WORK_DIR env var is required"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD=(python "$SCRIPT_DIR/dart_eval_task1.py" --model "$MODEL" --dart_work_dir "$DART_WORK_DIR")

if [[ -n "$DART_EVAL_DIR" ]]; then
  CMD+=(--dart_eval_dir "$DART_EVAL_DIR")
fi

if [[ "$DO_DATASET_GEN" == "true" ]]; then
  CMD+=(--run_dataset_gen)
fi

if [[ "$DO_ZERO_SHOT" == "true" ]]; then
  CMD+=(--run_zero_shot)
fi

if [[ -n "$CCRE_BED" ]]; then
  CMD+=(--ccre_bed "$CCRE_BED")
fi

if [[ -n "$PROCESSED_TSV" ]]; then
  CMD+=(--processed_tsv "$PROCESSED_TSV")
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
