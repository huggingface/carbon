#!/bin/bash
#SBATCH --job-name=c3b-niah
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

# Genome-NIAH retrieval, Carbon (HF backend).
# Override defaults via env vars:
#   MODEL=...     HF repo or path (default lc32k-rope5M native)
#   TASK=...      niah | neardup_d4 | neardup_d2 | neardup_d1   (default niah)
#   CTX=...       4096 | 8192 | 16384 | 32768 | 65536 | 131072  (default 32768)
#
# Run all (task, ctx) combos:
#   for TASK in niah neardup_d4 neardup_d2 neardup_d1; do
#     for CTX in 4096 8192 16384 32768; do
#       TASK=$TASK CTX=$CTX sbatch genome_niah.sbatch
#     done
#   done
set -euo pipefail
source ~/.bashrc

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
cd "$REPO_ROOT"

UV_RUN=(uv run --group evaluation)

OUT=$SLURM_SUBMIT_DIR/results/genome_niah
MODEL=${MODEL:-HuggingFaceBio/carbon-3B-longctx-32k-rope5M}
TASK=${TASK:-niah}
CTX=${CTX:-32768}

mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/genome_niah_eval.py" \
    --model "$MODEL" \
    --task "$TASK" --ctx "$CTX" \
    --add_dna_tag --bf16 \
    --output_dir "$OUT"
