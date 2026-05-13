#!/bin/bash
#SBATCH --job-name=c3b-traitgym
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00

# TraitGym Mendelian with --rev_comp_avg (strand-symmetric scoring; ~2x compute).

set -euo pipefail
source ~/.bashrc

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
cd "$REPO_ROOT"

UV_RUN=(uv run --group evaluation)

OUT=$SLURM_SUBMIT_DIR/results/traitgym
MODEL=HuggingFaceBio/Carbon-3B

mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/vep_eval.py" \
    --model "$MODEL" \
    --data_path hf://datasets/HuggingFaceBio/traitgym/mendelian_traits_vep.parquet \
    --add_dna_tag --bf16 --rev_comp_avg \
    --output_dir "$OUT"
