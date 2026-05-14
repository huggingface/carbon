#!/bin/bash
#SBATCH --job-name=gen3b-clinvar
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

# ClinVar coding + non_coding. Default 24 kb context; CONTEXT_LENGTH=48000 for 48 kb.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

if [[ -z "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
    while [[ "$REPO_ROOT" != "/" && ! -f "$REPO_ROOT/pyproject.toml" ]]; do
        REPO_ROOT=$(dirname "$REPO_ROOT")
    done
fi
if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "Could not locate repository root from REPO_ROOT=${REPO_ROOT}" >&2
    exit 1
fi
cd "$REPO_ROOT"

UV_RUN=(uv run --group evaluation)

CONTEXT_LENGTH=${CONTEXT_LENGTH:-24000}
OUT=$SLURM_SUBMIT_DIR/results/clinvar_${CONTEXT_LENGTH}
MODEL=GenerTeam/GENERator-v2-eukaryote-3b-base

mkdir -p "$OUT/coding" "$OUT/non_coding"

"${UV_RUN[@]}" python "evaluation/clinvar_vep_eval.py" \
    --model "$MODEL" \
    --clinvar_path hf://datasets/HuggingFaceBio/clinvar-vep-final/coding/clinvar_coding.parquet \
    --bf16 --context_length "$CONTEXT_LENGTH" \
    --output_dir "$OUT/coding"

"${UV_RUN[@]}" python "evaluation/clinvar_vep_eval.py" \
    --model "$MODEL" \
    --clinvar_path hf://datasets/HuggingFaceBio/clinvar-vep-final/non_coding/clinvar_non_coding.parquet \
    --bf16 --context_length "$CONTEXT_LENGTH" \
    --output_dir "$OUT/non_coding"
