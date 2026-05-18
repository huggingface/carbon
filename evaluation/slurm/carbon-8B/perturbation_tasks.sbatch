#!/bin/bash
#SBATCH --job-name=c8b-perturb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00

# Perturbation tasks chained on 8 GPUs (multi-GPU support).

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --no-sync --group evaluation)
RUN_SHA=${RUN_SHA:-$(git rev-parse --short=12 HEAD)}
OUT_ROOT=${OUT_ROOT:-results/$RUN_SHA}

OUT="$OUT_ROOT/perturbation_tasks"
MODEL=${MODEL:-HuggingFaceBio/Carbon-8B}

TASK_SPECS=(
    "motif_human:motif_human:test"
    "syn_human:syn_human:test"
    "syn_mouse:syn_mouse:test"
    "promoter_revcomp:promoter_revcomp:test"
)

for TASK_SPEC in "${TASK_SPECS[@]}"; do
    IFS=: read -r TASK_NAME OUT_SUBDIR SPLIT <<< "$TASK_SPEC"
    TASK_OUT="$OUT/$OUT_SUBDIR"
    mkdir -p "$TASK_OUT"

    "${UV_RUN[@]}" python "evaluation/perturbation_tasks.py" \
        --task "$TASK_NAME" \
        --model "$MODEL" \
        --split "$SPLIT" \
        --bf16 \
        --output_dir "$TASK_OUT"
done
