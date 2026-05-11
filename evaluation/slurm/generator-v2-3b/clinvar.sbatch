#!/bin/bash
#SBATCH --job-name=gen3b-clinvar
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

# ClinVar coding + non_coding. Default 24 kb context; CONTEXT_LENGTH=48000 for 48 kb.

set -euo pipefail
source ~/.bashrc

HF_VENV=${HF_VENV:-/fsx/loubna/.venv-2-6-cu124/bin/activate}
source "$HF_VENV"

CONTEXT_LENGTH=${CONTEXT_LENGTH:-24000}
EVAL=${EVAL_DIR:-/fsx/loubna/projects_v2/carbon/loubna-workspace/release_repos/carbon/evaluation}
OUT=$SLURM_SUBMIT_DIR/results/clinvar_${CONTEXT_LENGTH}
MODEL=GenerTeam/GENERator-v2-eukaryote-3b-base

mkdir -p "$OUT/coding" "$OUT/non_coding"

python "$EVAL/clinvar_vep_eval.py" \
    --model "$MODEL" \
    --clinvar_path hf://datasets/hf-carbon/clinvar-vep-final/coding/clinvar_coding.parquet \
    --bf16 --context_length "$CONTEXT_LENGTH" \
    --output_dir "$OUT/coding"

python "$EVAL/clinvar_vep_eval.py" \
    --model "$MODEL" \
    --clinvar_path hf://datasets/hf-carbon/clinvar-vep-final/non_coding/clinvar_non_coding.parquet \
    --bf16 --context_length "$CONTEXT_LENGTH" \
    --output_dir "$OUT/non_coding"
