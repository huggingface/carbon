#!/bin/bash
#SBATCH --job-name=c500m-traitgym
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00

# TraitGym Mendelian with --rev_comp_avg (strand-symmetric scoring; ~2x compute).

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --group evaluation)

OUT=results/traitgym
MODEL=HuggingFaceBio/Carbon-500M

mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/vep_eval.py" \
    --model "$MODEL" \
    --data_path hf://datasets/HuggingFaceBio/traitgym/mendelian_traits_vep.parquet \
    --bf16 --rev_comp_avg \
    --output_dir "$OUT"
