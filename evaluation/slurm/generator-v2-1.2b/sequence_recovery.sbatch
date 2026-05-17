#!/bin/bash
#SBATCH --job-name=gen1p2b-sr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --no-sync --group evaluation)
RUN_SHA=${RUN_SHA:-$(git rev-parse --short=12 HEAD)}
OUT_ROOT=${OUT_ROOT:-results/$RUN_SHA}

OUT="$OUT_ROOT/sequence_recovery"
MODEL=GenerTeam/GENERator-v2-eukaryote-1.2b-base

mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/sequence_recovery.py" \
    --model "$MODEL" \
    --data_type eukaryote --add_bos --bf16 \
    --output_dir "$OUT"
