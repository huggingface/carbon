#!/bin/bash
#SBATCH --job-name=c3b-clinvar
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc
source /fsx/loubna/.venv-2-6-cu124/bin/activate

EVAL=/fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/evaluation
OUT=$SLURM_SUBMIT_DIR/results/clinvar
MODEL=hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1

mkdir -p $OUT

# ClinVar — default 24 kb left context, right-end / next-token scoring
python $EVAL/clinvar_vep_eval.py \
    --model $MODEL \
    --add_dna_tag --bf16 \
    --context_length 24000 \
    --output_dir $OUT
