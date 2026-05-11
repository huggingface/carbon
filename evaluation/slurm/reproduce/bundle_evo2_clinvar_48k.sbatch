#!/bin/bash
#SBATCH --job-name=repro-evo2-clinvar48k
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=36:00:00

# Evo2 1B · ClinVar at 48 kb left context (single GPU, longer wall time).

set -euo pipefail
source ~/.bashrc
source /fsx/loubna/projects_v2/.evo2-cu124/bin/activate

export CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_TRUST_REMOTE_CODE=1
NVIDIA_LIBS=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')")
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}

EVAL=/fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/evaluation
OUT=$SLURM_SUBMIT_DIR/results/evo2-1b/clinvar_48k
mkdir -p $OUT/coding $OUT/non_coding

MODEL=evo2_1b_base
CODING=hf://datasets/hf-carbon/clinvar-vep-final/coding/clinvar_coding.parquet
NONCODING=hf://datasets/hf-carbon/clinvar-vep-final/non_coding/clinvar_non_coding.parquet

echo "=== Evo2 1B · ClinVar CODING @ 48 kb ==="
python $EVAL/clinvar_vep_eval.py \
    --model $MODEL --backend evo2 --clinvar_path $CODING \
    --bf16 --context_length 48000 \
    --output_dir $OUT/coding

echo "=== Evo2 1B · ClinVar NON-CODING @ 48 kb ==="
python $EVAL/clinvar_vep_eval.py \
    --model $MODEL --backend evo2 --clinvar_path $NONCODING \
    --bf16 --context_length 48000 \
    --output_dir $OUT/non_coding

echo "=== DONE ==="
