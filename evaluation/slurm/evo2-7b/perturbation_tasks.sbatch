#!/bin/bash
#SBATCH --job-name=evo2-7b-perturb
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00

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

OUT=$SLURM_SUBMIT_DIR/results
mkdir -p "$OUT/tata" "$OUT/synonymous_codons"

"${UV_RUN[@]}" python "evaluation/perturbation_tasks.py" \
    --task tata_perturbation \
    --model evo2_7b_base --backend evo2 --bf16 \
    --output_dir "$OUT/tata"

"${UV_RUN[@]}" python "evaluation/perturbation_tasks.py" \
    --task synonymous_codon_substitution \
    --model evo2_7b_base --backend evo2 --bf16 \
    --output_dir "$OUT/synonymous_codons"
