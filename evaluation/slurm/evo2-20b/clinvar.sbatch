#!/bin/bash
#SBATCH --job-name=evo2-20b-clinvar
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=48:00:00

# ClinVar coding + non_coding. Default 24 kb context; CONTEXT_LENGTH=48000 for 48 kb.
# Evo2-20B fits per H100 in the ClinVar shard path, so all allocated GPUs are
# used as independent shards.

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --group evaluation)

CONTEXT_LENGTH=${CONTEXT_LENGTH:-24000}
OUT=${OUT:-results/clinvar_${CONTEXT_LENGTH}}
MODEL=${MODEL:-arcinstitute/evo2_20b}
mkdir -p "$OUT/coding" "$OUT/non_coding"

"${UV_RUN[@]}" python "evaluation/clinvar_vep_eval.py" \
    --model "$MODEL" --backend evo2 \
    --clinvar_path hf://datasets/HuggingFaceBio/clinvar-vep-final/coding/clinvar_coding.parquet \
    --bf16 --context_length "$CONTEXT_LENGTH" \
    --output_dir "$OUT/coding"

"${UV_RUN[@]}" python "evaluation/clinvar_vep_eval.py" \
    --model "$MODEL" --backend evo2 \
    --clinvar_path hf://datasets/HuggingFaceBio/clinvar-vep-final/non_coding/clinvar_non_coding.parquet \
    --bf16 --context_length "$CONTEXT_LENGTH" \
    --output_dir "$OUT/non_coding"
