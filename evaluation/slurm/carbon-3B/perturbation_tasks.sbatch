#!/bin/bash
#SBATCH --job-name=c3b-perturb
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
MODEL=hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1

mkdir -p $OUT/tata $OUT/synonymous_codons

# TATA perturbation
python $EVAL/perturbation_tasks.py \
    --task tata_perturbation \
    --model $MODEL \
    --add_dna_tag --bf16 \
    --output_dir $OUT/tata

# Synonymous codon substitution
python $EVAL/perturbation_tasks.py \
    --task synonymous_codon_substitution \
    --model $MODEL \
    --add_dna_tag --bf16 \
    --output_dir $OUT/synonymous_codons
