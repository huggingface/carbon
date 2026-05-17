#!/bin/bash
#SBATCH --job-name=evo2-7b-sr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00

source ~/.bashrc
set -euo pipefail
UV_RUN=(uv run --group evaluation)


OUT=${OUT:-results/sequence_recovery}
MODEL=${MODEL:-arcinstitute/evo2_7b}
mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/sequence_recovery.py" \
    --model "$MODEL" --backend evo2 \
    --data_type eukaryote --gen_len_bp 30 --bf16 \
    --output_dir "$OUT"
