#!/bin/bash
#SBATCH --job-name=c3b-clinvar
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

# ClinVar coding + non_coding chained. Default 24 kb left context;
# override with CONTEXT_LENGTH=48000 for the full Carbon-paper recipe.
#   sbatch evaluation/slurm/carbon-3B/clinvar.sbatch  # 24 kb
#   sbatch --export=CONTEXT_LENGTH=48000 evaluation/slurm/carbon-3B/clinvar.sbatch  # 48 kb

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --group evaluation)

CONTEXT_LENGTH=${CONTEXT_LENGTH:-24000}
OUT=results/clinvar_${CONTEXT_LENGTH}
MODEL=HuggingFaceBio/Carbon-3B

mkdir -p "$OUT/coding" "$OUT/non_coding"

"${UV_RUN[@]}" python "evaluation/clinvar_vep_eval.py" \
    --model "$MODEL" \
    --clinvar_path hf://datasets/HuggingFaceBio/clinvar-vep-final/coding/clinvar_coding.parquet \
    --add_dna_tag --bf16 --context_length "$CONTEXT_LENGTH" \
    --output_dir "$OUT/coding"

"${UV_RUN[@]}" python "evaluation/clinvar_vep_eval.py" \
    --model "$MODEL" \
    --clinvar_path hf://datasets/HuggingFaceBio/clinvar-vep-final/non_coding/clinvar_non_coding.parquet \
    --add_dna_tag --bf16 --context_length "$CONTEXT_LENGTH" \
    --output_dir "$OUT/non_coding"
