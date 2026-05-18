#!/bin/bash
#SBATCH --job-name=c3b-brca2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00

# BRCA2 on 8 GPUs. ~5 min on Carbon 3B.

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --no-sync --group evaluation)
RUN_SHA=${RUN_SHA:-$(git rev-parse --short=12 HEAD)}
OUT_ROOT=${OUT_ROOT:-results/$RUN_SHA}

OUT="$OUT_ROOT/brca2"
MODEL=HuggingFaceBio/Carbon-3B

mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/vep_eval.py" \
    --model "$MODEL" \
    --data_path hf://datasets/HuggingFaceBio/brca2-vep/brca2_vep.parquet \
    --bf16 \
    --output_dir "$OUT"
