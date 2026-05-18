#!/bin/bash
#SBATCH --job-name=c8b-niah
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

# Genome-NIAH retrieval, Carbon (HF backend).
# Override defaults via env vars:
#   MODEL=...     HF repo or path (default Carbon-8B)
#   TASK=...      niah | neardup_d4 | neardup_d2 | neardup_d1   (default niah)
#   CTX=...       4096 | 8192 | 16384 | 32768 | 65536 | 131072  (default 65536)
#
# Run all (task, ctx) combos:
#   for TASK in niah neardup_d4 neardup_d2 neardup_d1; do
#     for CTX in 4096 8192 16384 32768 65536 131072; do
#       TASK=$TASK CTX=$CTX sbatch evaluation/slurm/carbon-8B/genome_niah.sbatch
#     done
#   done
source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --no-sync --group evaluation)
RUN_SHA=${RUN_SHA:-$(git rev-parse --short=12 HEAD)}
OUT_ROOT=${OUT_ROOT:-results/$RUN_SHA}

OUT="$OUT_ROOT/genome_niah"
MODEL=${MODEL:-HuggingFaceBio/Carbon-8B}
MODEL_SHORT_NAME=${MODEL_SHORT_NAME:-Carbon-8B-yarn}
TASK=${TASK:-niah}
CTX=${CTX:-65536}

mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/genome_niah_eval.py" \
    --model "$MODEL" \
    --task "$TASK" --ctx "$CTX" \
    --add_dna_tag --bf16 \
    --with_yarn --model_short_name "$MODEL_SHORT_NAME" \
    --output_dir "$OUT"
