#!/bin/bash
#SBATCH --job-name=c3b-traitgym
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00

# TraitGym Mendelian with --rev_comp_avg (strand-symmetric scoring; ~2x compute).

set -euo pipefail
source ~/.bashrc

HF_VENV=${HF_VENV:-/fsx/loubna/.venv-2-6-cu124/bin/activate}
source "$HF_VENV"

EVAL=${EVAL_DIR:-/fsx/loubna/projects_v2/carbon/loubna-workspace/release_repos/carbon/evaluation}
OUT=$SLURM_SUBMIT_DIR/results/traitgym
MODEL=HuggingFaceBio/Carbon-3B

mkdir -p "$OUT"

python "$EVAL/vep_eval.py" \
    --model "$MODEL" \
    --data_path hf://datasets/HuggingFaceBio/traitgym/mendelian_traits_vep.parquet \
    --add_dna_tag --bf16 --rev_comp_avg \
    --output_dir "$OUT"
