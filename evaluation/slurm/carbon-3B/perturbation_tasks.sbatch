#!/bin/bash
#SBATCH --job-name=c3b-perturb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# TATA + synonymous-codon substitution chained on 1 GPU.

set -euo pipefail
source ~/.bashrc

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
cd "$REPO_ROOT"

UV_RUN=(uv run --group evaluation)

OUT=$SLURM_SUBMIT_DIR/results
MODEL=HuggingFaceBio/Carbon-3B

mkdir -p "$OUT/tata" "$OUT/synonymous_codons"

"${UV_RUN[@]}" python "evaluation/perturbation_tasks.py" \
    --task tata_perturbation \
    --model "$MODEL" \
    --add_dna_tag --bf16 \
    --output_dir "$OUT/tata"

"${UV_RUN[@]}" python "evaluation/perturbation_tasks.py" \
    --task synonymous_codon_substitution \
    --model "$MODEL" \
    --add_dna_tag --bf16 \
    --output_dir "$OUT/synonymous_codons"
