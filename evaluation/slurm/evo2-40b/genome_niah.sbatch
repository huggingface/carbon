#!/bin/bash
#SBATCH --job-name=evo2-40b-niah
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=72:00:00

# Genome-NIAH retrieval, Evo2-40B (vortex backend on 8 GPUs).
# Long-ctx is slow — use sharding to split a (task, ctx) cell across jobs.
#
# Env overrides:
#   MODEL=...     default arcinstitute/evo2_40b
#   TASK=...      niah | neardup_d4 | neardup_d2 | neardup_d1
#   CTX=...       4096 | 8192 | 16384 | 32768 | 65536 | 131072
#   SHARD=...     0..N-1   (default 0)
#   NSHARDS=...   N        (default 1; recommend many shards for ctx >= 32k)
#   PREFILL=...   prefill_chars (default 4096)
#   POOL=...      max_samples cap (default: use all 500/cell)
source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --no-sync --group evaluation --group evo2)
RUN_SHA=${RUN_SHA:-$(git rev-parse --short=12 HEAD)}
OUT_ROOT=${OUT_ROOT:-results/$RUN_SHA}

OUT="$OUT_ROOT/genome_niah"
MODEL=${MODEL:-arcinstitute/evo2_40b}
TASK=${TASK:-niah}
CTX=${CTX:-32768}
SHARD=${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}
NSHARDS=${NSHARDS:-1}
PREFILL=${PREFILL:-4096}

mkdir -p "$OUT"

EXTRA=()
if [ -n "${POOL:-}" ]; then EXTRA+=(--max_samples "$POOL"); fi

"${UV_RUN[@]}" python "evaluation/genome_niah_eval.py" \
    --model "$MODEL" --backend evo2 \
    --task "$TASK" --ctx "$CTX" \
    --shard_idx "$SHARD" --n_shards "$NSHARDS" \
    --prefill_chars "$PREFILL" \
    --output_dir "$OUT" \
    "${EXTRA[@]}"
