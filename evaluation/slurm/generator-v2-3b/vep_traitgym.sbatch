#!/bin/bash
#SBATCH --job-name=gen3b-traitgym
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00

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

OUT=$SLURM_SUBMIT_DIR/results/traitgym
MODEL=GenerTeam/GENERator-v2-eukaryote-3b-base

mkdir -p "$OUT"

"${UV_RUN[@]}" python "evaluation/vep_eval.py" \
    --model "$MODEL" \
    --data_path hf://datasets/HuggingFaceBio/traitgym/mendelian_traits_vep.parquet \
    --bf16 --rev_comp_avg \
    --output_dir "$OUT"
