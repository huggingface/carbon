#!/bin/bash
#SBATCH --job-name=gen3b-brca
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00

set -euo pipefail
source ~/.bashrc

HF_VENV=${HF_VENV:-/fsx/loubna/.venv-2-6-cu124/bin/activate}
source "$HF_VENV"

EVAL=${EVAL_DIR:-/fsx/loubna/projects_v2/carbon/loubna-workspace/release_repos/carbon/evaluation}
OUT=$SLURM_SUBMIT_DIR/results
MODEL=GenerTeam/GENERator-v2-eukaryote-3b-base

mkdir -p "$OUT/brca1" "$OUT/brca2"

python "$EVAL/vep_eval.py" \
    --model "$MODEL" \
    --data_path hf://datasets/hf-carbon/brca1-vep/brca1_vep.parquet \
    --bf16 \
    --output_dir "$OUT/brca1"

python "$EVAL/vep_eval.py" \
    --model "$MODEL" \
    --data_path hf://datasets/hf-carbon/brca2-vep/brca2_vep.parquet \
    --bf16 \
    --output_dir "$OUT/brca2"
