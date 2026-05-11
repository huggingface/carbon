#!/bin/bash
#SBATCH --job-name=gen3b-clinvar
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
MODEL=GenerTeam/GENERator-v2-eukaryote-3b-base

mkdir -p $OUT

python $EVAL/clinvar_vep_eval.py \
    --model $MODEL \
    --bf16 --context_length 24000 \
    --output_dir $OUT
