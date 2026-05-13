#!/bin/bash
#SBATCH --job-name=c3b-sr
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00

set -euo pipefail
source ~/.bashrc

HF_VENV=${HF_VENV:-/fsx/loubna/.venv-2-6-cu124/bin/activate}
source "$HF_VENV"

EVAL=${EVAL_DIR:-/fsx/loubna/projects_v2/carbon/loubna-workspace/release_repos/carbon/evaluation}
OUT=$SLURM_SUBMIT_DIR/results/sequence_recovery
MODEL=HuggingFaceBio/carbon-3B-hybrid-loss-1T-mix2-v1

mkdir -p "$OUT"

python "$EVAL/sequence_recovery.py" \
    --model "$MODEL" \
    --data_type eukaryote \
    --add_dna_tag --bf16 \
    --output_dir "$OUT"
