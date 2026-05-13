#!/bin/bash
#SBATCH --job-name=gen3b-perturb
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
source ~/.bashrc

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
cd "$REPO_ROOT"

UV_RUN=(uv run --group evaluation)

OUT=$SLURM_SUBMIT_DIR/results
MODEL=GenerTeam/GENERator-v2-eukaryote-3b-base

mkdir -p "$OUT/tata" "$OUT/synonymous_codons"

"${UV_RUN[@]}" python "evaluation/perturbation_tasks.py" \
    --task tata_perturbation \
    --model "$MODEL" --bf16 \
    --output_dir "$OUT/tata"

"${UV_RUN[@]}" python "evaluation/perturbation_tasks.py" \
    --task synonymous_codon_substitution \
    --model "$MODEL" --bf16 \
    --output_dir "$OUT/synonymous_codons"
