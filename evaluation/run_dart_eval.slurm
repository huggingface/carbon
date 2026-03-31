#!/bin/bash
#SBATCH --output=/fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation/slurm_logs/dart_eval_%A.out
#SBATCH --error=/fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation/slurm_logs/dart_eval_%A.err
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --qos=high

# Usage: sbatch run_dart_eval.sbatch <model_path> <prefix_mode> <model_name>
# prefix_mode: "dna_tags" (hybrid), "no_prefix" (6-mer raw), or "default" (uses <s>)
source /fsx/loubna/.venv-2-6-cu124/bin/activate
export HF_HUB_TRUST_REMOTE_CODE=1

MODEL_PATH=$1
PREFIX_MODE=$2
MODEL_NAME=$3

DART_WORK_DIR="/fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/dart_work"
OUTPUT_DIR="./eval_results/dart_task1/${MODEL_NAME}"
mkdir -p ${OUTPUT_DIR}

echo "Running DART Task 1: Model=${MODEL_PATH}, prefix_mode=${PREFIX_MODE}, model_name=${MODEL_NAME}"

cd /fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation

# Build the prefix flag based on mode
if [ "$PREFIX_MODE" = "dna_tags" ]; then
    PREFIX_FLAG="--use_dna_tags"
elif [ "$PREFIX_MODE" = "no_prefix" ]; then
    PREFIX_FLAG="--no_prefix"
else
    PREFIX_FLAG=""
fi

python dart_eval_task1.py \
    --model ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --dart_work_dir ${DART_WORK_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 16 \
    --bf16 \
    ${PREFIX_FLAG}

echo "DART eval completed!"
