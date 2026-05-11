#!/bin/bash
#SBATCH --job-name=gen3b-perturb
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00

set -euo pipefail
source ~/.bashrc
source /fsx/loubna/.venv-2-6-cu124/bin/activate

EVAL=/fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/evaluation
OUT=$SLURM_SUBMIT_DIR/results
MODEL=GenerTeam/GENERator-v2-eukaryote-3b-base

mkdir -p $OUT/tata $OUT/synonymous_codons

python $EVAL/perturbation_tasks.py \
    --task tata_perturbation \
    --model $MODEL \
    --bf16 \
    --output_dir $OUT/tata

python $EVAL/perturbation_tasks.py \
    --task synonymous_codon_substitution \
    --model $MODEL \
    --bf16 \
    --output_dir $OUT/synonymous_codons
