#!/bin/bash
#SBATCH --job-name=evo2-7b-niah
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00

# Genome-NIAH retrieval, Evo2-7B (vortex backend on 8 GPUs).
# Long-ctx is slow — per-row time grows from ~3 min (8k) to ~14h (128k).
# Use sharding to split a (task, ctx) cell across multiple jobs.
#
# Env overrides:
#   MODEL=...     default evo2_7b
#   TASK=...      niah | neardup_d4 | neardup_d2 | neardup_d1
#   CTX=...       4096 | 8192 | 16384 | 32768 | 65536 | 131072
#   SHARD=...     0..N-1   (default 0)
#   NSHARDS=...   N        (default 1; recommend 6 for ctx ≥ 32k)
#   PREFILL=...   prefill_chars (default 4096)
#   POOL=...      max_samples cap (default: use all 500/cell)
#
# Example — full sweep at ctx=32k using 6 shards × 8-GPU nodes:
#   for SHARD in 0 1 2 3 4 5; do
#     SHARD=$SHARD NSHARDS=6 CTX=32768 sbatch genome_niah.sbatch
#   done
set -euo pipefail
source ~/.bashrc

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
cd "$REPO_ROOT"

UV_RUN=(uv run --group evaluation)

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export HF_HUB_TRUST_REMOTE_CODE=1
NVIDIA_LIBS=$("${UV_RUN[@]}" python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')")
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}

OUT=$SLURM_SUBMIT_DIR/results/genome_niah

MODEL=${MODEL:-evo2_7b}
TASK=${TASK:-niah}
CTX=${CTX:-32768}
SHARD=${SHARD:-0}
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
