#!/bin/bash
#SBATCH --output=/fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation/slurm_logs/tata_v2_eval_%A.out
#SBATCH --error=/fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation/slurm_logs/tata_v2_eval_%A.err
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --qos=normal

# Usage: sbatch run_tata_v2_eval.sbatch <model_path> <prefix_mode> <model_name>
# prefix_mode: "dna_tags" (hybrid), "no_prefix" (6-mer raw), or "default"
source /fsx/loubna/.venv-2-6-cu124/bin/activate
export HF_HUB_TRUST_REMOTE_CODE=1

MODEL_PATH=$1
PREFIX_MODE=$2
MODEL_NAME=$3

DATASET="hf-carbon/test_datasets"
SPLIT="tata_perturbed_v2"
TASK="tata_perturbation"
OUTPUT_DIR="./eval_results/tata_perturbation_v2"
MAX_LENGTH=2048
BATCH_SIZE=16

cd /fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation

# Build prefix flag
if [ "$PREFIX_MODE" = "dna_tags" ]; then
    PREFIX_FLAG="--use_dna_tags"
elif [ "$PREFIX_MODE" = "no_prefix" ]; then
    PREFIX_FLAG="--no_prefix"
else
    PREFIX_FLAG=""
fi

python cds_half_shuffle_eval.py \
    --model ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --dataset ${DATASET} \
    --split ${SPLIT} \
    --task ${TASK} \
    --output_dir ${OUTPUT_DIR} \
    --max_length ${MAX_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --bf16 \
    ${PREFIX_FLAG}

echo "Job completed!"
