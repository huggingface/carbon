#!/bin/bash
#SBATCH --job-name=evo2-1b-brca2
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=06:00:00

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
mkdir -p "$OUT/brca2"

"${UV_RUN[@]}" python "evaluation/vep_eval.py" \
    --model evo2_1b_base --backend evo2 \
    --data_path hf://datasets/HuggingFaceBio/brca2-vep/brca2_vep.parquet \
    --bf16 \
    --output_dir "$OUT/brca2"
