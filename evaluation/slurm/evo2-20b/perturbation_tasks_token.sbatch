#!/bin/bash
#SBATCH --job-name=evo2-20b-perturb-token
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=24:00:00

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --group evaluation)

OUT=${OUT:-results/perturbation_tasks_token}
MODEL=${MODEL:-arcinstitute/evo2_20b}

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

    "${UV_RUN[@]}" python "evaluation/perturbation_tasks_token.py" \
        --task "$TASK_NAME" \
        --model "$MODEL" --backend evo2 \
        --split "$SPLIT" \
        --bf16 \
        --output_dir "$TASK_OUT"
done
