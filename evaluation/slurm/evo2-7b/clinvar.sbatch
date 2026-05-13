#!/bin/bash
#SBATCH --job-name=evo2-7b-clinvar
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=36:00:00

# ClinVar coding + non_coding. Default 24 kb context; CONTEXT_LENGTH=48000 for 48 kb.
# 2 GPUs for headroom on Evo2 7B at long context.

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

CONTEXT_LENGTH=${CONTEXT_LENGTH:-24000}
OUT=$SLURM_SUBMIT_DIR/results/clinvar_${CONTEXT_LENGTH}
mkdir -p "$OUT/coding" "$OUT/non_coding"

"${UV_RUN[@]}" python "evaluation/clinvar_vep_eval.py" \
    --model evo2_7b_base --backend evo2 \
    --clinvar_path hf://datasets/HuggingFaceBio/clinvar-vep-final/coding/clinvar_coding.parquet \
    --bf16 --context_length "$CONTEXT_LENGTH" \
    --output_dir "$OUT/coding"

"${UV_RUN[@]}" python "evaluation/clinvar_vep_eval.py" \
    --model evo2_7b_base --backend evo2 \
    --clinvar_path hf://datasets/HuggingFaceBio/clinvar-vep-final/non_coding/clinvar_non_coding.parquet \
    --bf16 --context_length "$CONTEXT_LENGTH" \
    --output_dir "$OUT/non_coding"
