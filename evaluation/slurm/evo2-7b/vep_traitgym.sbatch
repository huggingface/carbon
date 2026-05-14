#!/bin/bash
#SBATCH --job-name=evo2-7b-traitgym
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=30:00:00

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --group evaluation)


OUT=results/traitgym
mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/vep_eval.py" \
    --model evo2_7b_base --backend evo2 \
    --data_path hf://datasets/HuggingFaceBio/traitgym/mendelian_traits_vep.parquet \
    --bf16 --rev_comp_avg \
    --output_dir "$OUT"
