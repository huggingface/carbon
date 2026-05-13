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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
cd "$REPO_ROOT"

UV_RUN=(uv run --group evaluation)

OUT=$SLURM_SUBMIT_DIR/results/genome_niah
MODEL=${MODEL:-GenerTeam/GENERator-v2-eukaryote-3b-base}
TASK=${TASK:-niah}
CTX=${CTX:-16384}

mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/genome_niah_eval.py" \
    --model "$MODEL" \
    --task "$TASK" --ctx "$CTX" \
    --bf16 \
    --output_dir "$OUT"
