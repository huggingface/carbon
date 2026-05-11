#!/bin/bash
#SBATCH --job-name=genv2-niah
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

# Genome-NIAH retrieval, GENERator-v2 (HF backend).
# Native is 16k; only ctx ≤ 16k unless you wrap with a YaRN config patch.
# Env overrides:
#   MODEL=...   default GenerTeam/GENERator-v2-eukaryote-3b-base
#   TASK=...    niah | neardup_d4 | neardup_d2 | neardup_d1
#   CTX=...     4096 | 8192 | 16384
set -euo pipefail
source ~/.bashrc

HF_VENV=${HF_VENV:-/fsx/loubna/.venv-2-6-cu124/bin/activate}
source "$HF_VENV"

EVAL=${EVAL_DIR:-/fsx/loubna/projects_v2/carbon/loubna-workspace/release_repos/carbon/evaluation}
OUT=$SLURM_SUBMIT_DIR/results/genome_niah
MODEL=${MODEL:-GenerTeam/GENERator-v2-eukaryote-3b-base}
TASK=${TASK:-niah}
CTX=${CTX:-16384}

mkdir -p "$OUT"

python "$EVAL/genome_niah_eval.py" \
    --model "$MODEL" \
    --task "$TASK" --ctx "$CTX" \
    --bf16 \
    --output_dir "$OUT"
