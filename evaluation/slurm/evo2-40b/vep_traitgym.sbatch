#!/bin/bash
#SBATCH --job-name=evo2-40b-traitgym
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=72:00:00

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --group evaluation)

OUT=${OUT:-results/traitgym}
MODEL=${MODEL:-arcinstitute/evo2_40b}
mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/vep_eval.py" \
    --model "$MODEL" --backend evo2 \
    --data_path hf://datasets/HuggingFaceBio/traitgym/mendelian_traits_vep.parquet \
    --bf16 --rev_comp_avg \
    --output_dir "$OUT"
