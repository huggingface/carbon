#!/bin/bash
#SBATCH --job-name=evo2-20b-brca2
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

OUT=${OUT:-results}
MODEL=${MODEL:-arcinstitute/evo2_20b}
mkdir -p "$OUT/brca2"

"${UV_RUN[@]}" python "evaluation/vep_eval.py" \
    --model "$MODEL" --backend evo2 \
    --data_path hf://datasets/HuggingFaceBio/brca2-vep/brca2_vep.parquet \
    --bf16 \
    --output_dir "$OUT/brca2"
