#!/bin/bash
#SBATCH --job-name=c3b-traitgym
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
OUT=$SLURM_SUBMIT_DIR/results/traitgym
MODEL=hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1

mkdir -p $OUT

# TraitGym Mendelian (with --rev_comp_avg, strand-symmetric scoring)
python $EVAL/vep_eval.py \
    --model $MODEL \
    --data_path hf://datasets/hf-carbon/traitgym/mendelian_traits_vep.parquet \
    --add_dna_tag --bf16 --rev_comp_avg \
    --output_dir $OUT
