#!/bin/bash
#SBATCH --job-name=c3b-brca2
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00

# BRCA2 on 8 GPUs. ~5 min on Carbon 3B.

set -euo pipefail
source ~/.bashrc

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
cd "$REPO_ROOT"

UV_RUN=(uv run --group evaluation)

OUT=$SLURM_SUBMIT_DIR/results
MODEL=HuggingFaceBio/Carbon-3B

mkdir -p "$OUT/brca2"

"${UV_RUN[@]}" python "evaluation/vep_eval.py" \
    --model "$MODEL" \
    --data_path hf://datasets/HuggingFaceBio/brca2-vep/brca2_vep.parquet \
    --add_dna_tag --bf16 \
    --output_dir "$OUT/brca2"
