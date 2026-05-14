#!/bin/bash
#SBATCH --job-name=evo2-1b-brca2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=06:00:00

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


OUT=$SLURM_SUBMIT_DIR/results
mkdir -p "$OUT/brca2"

"${UV_RUN[@]}" python "evaluation/vep_eval.py" \
    --model evo2_1b_base --backend evo2 \
    --data_path hf://datasets/HuggingFaceBio/brca2-vep/brca2_vep.parquet \
    --bf16 \
    --output_dir "$OUT/brca2"
