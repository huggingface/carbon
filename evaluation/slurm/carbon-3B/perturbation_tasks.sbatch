#!/bin/bash
#SBATCH --job-name=c3b-perturb
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# TATA + synonymous-codon substitution chained on 1 GPU.

set -euo pipefail
source ~/.bashrc

HF_VENV=${HF_VENV:-/fsx/loubna/.venv-2-6-cu124/bin/activate}
source "$HF_VENV"

EVAL=${EVAL_DIR:-/fsx/loubna/projects_v2/carbon/loubna-workspace/release_repos/carbon/evaluation}
OUT=$SLURM_SUBMIT_DIR/results
MODEL=HuggingFaceBio/Carbon-3B

mkdir -p "$OUT/tata" "$OUT/synonymous_codons"

python "$EVAL/perturbation_tasks.py" \
    --task tata_perturbation \
    --model "$MODEL" \
    --add_dna_tag --bf16 \
    --output_dir "$OUT/tata"

python "$EVAL/perturbation_tasks.py" \
    --task synonymous_codon_substitution \
    --model "$MODEL" \
    --add_dna_tag --bf16 \
    --output_dir "$OUT/synonymous_codons"
