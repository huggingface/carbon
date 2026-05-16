#!/bin/bash
#SBATCH --job-name=c3b-perturb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# Perturbation tasks chained on 1 GPU.

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --group evaluation)

OUT=${OUT:-results}
MODEL=${MODEL:-HuggingFaceBio/Carbon-3B}

TASK_SPECS=(
    "tata_perturbation:tata"
    "synonymous_codon_substitution:synonymous_codons"
    "motif_human:motif_human"
    "promoter_revcomp:promoter_revcomp"
    "syn_human:syn_human"
    "syn_mouse:syn_mouse"
)

for TASK_SPEC in "${TASK_SPECS[@]}"; do
    IFS=: read -r TASK_NAME OUT_SUBDIR <<< "$TASK_SPEC"
    TASK_OUT="$OUT/$OUT_SUBDIR"
    mkdir -p "$TASK_OUT"

    "${UV_RUN[@]}" python "evaluation/perturbation_tasks.py" \
        --task "$TASK_NAME" \
        --model "$MODEL" \
        --bf16 \
        --output_dir "$TASK_OUT"
done
