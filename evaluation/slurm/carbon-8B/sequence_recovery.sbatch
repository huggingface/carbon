#!/bin/bash
#SBATCH --job-name=c8b-sr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --group evaluation)

OUT=results/sequence_recovery
MODEL=HuggingFaceBio/Carbon-8B

mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/sequence_recovery.py" \
    --model "$MODEL" \
    --data_type eukaryote \
    --add_dna_tag --bf16 \
    --output_dir "$OUT"
