#!/bin/bash
#SBATCH --job-name=evo2-1b-sr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --no-sync --group evaluation --group evo2)
RUN_SHA=${RUN_SHA:-$(git rev-parse --short=12 HEAD)}
OUT_ROOT=${OUT_ROOT:-results/$RUN_SHA}
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}


OUT="$OUT_ROOT/sequence_recovery"
MODEL=${MODEL:-arcinstitute/evo2_1b_base}
mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/sequence_recovery.py" \
    --model "$MODEL" --backend evo2 \
    --data_type eukaryote --gen_len_bp 30 --bf16 \
    --output_dir "$OUT" \
    --shard_idx "$SHARD" --n_shards "$NSHARDS"
