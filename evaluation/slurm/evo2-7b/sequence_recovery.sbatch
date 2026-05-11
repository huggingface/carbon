#!/bin/bash
#SBATCH --job-name=evo2-7b-sr
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00

set -euo pipefail
source ~/.bashrc

EVO2_VENV=${EVO2_VENV:-/fsx/loubna/projects_v2/.evo2-cu124/bin/activate}
source "$EVO2_VENV"

export CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_TRUST_REMOTE_CODE=1
NVIDIA_LIBS=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')")
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}

EVAL=${EVAL_DIR:-/fsx/loubna/projects_v2/carbon/loubna-workspace/release_repos/carbon/evaluation}
OUT=$SLURM_SUBMIT_DIR/results/sequence_recovery
mkdir -p "$OUT"

python "$EVAL/sequence_recovery.py" \
    --model evo2_7b_base --backend evo2 \
    --data_type eukaryote --gen_len_bp 30 --bf16 \
    --output_dir "$OUT"
