#!/bin/bash
#SBATCH --output=/fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation/slurm_logs/seq_recovery_eval_%A.out
#SBATCH --error=/fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation/slurm_logs/seq_recovery_eval_%A.err
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --qos=high

# Usage: sbatch run_3_evals.sbatch <model_path> <prefix_mode> <model_short_name> <checkpoint>
# prefix_mode: "dna_tags" (hybrid), "no_prefix" (6-mer raw), or "default" (uses <s>)
source /fsx/loubna/.venv-2-6-cu124/bin/activate
export HF_HUB_TRUST_REMOTE_CODE=1

MODEL_PATH=$1
PREFIX_MODE=$2
MODEL_SHORT_NAME=$3
CHECKPOINT=$4

DATA_TYPE="eukaryote"
# Output structure: {base_dir}/{data_type}/ with files {model_name}_{step}.parquet
BASE_OUTPUT_DIR="./eval_results/sequence_recovery_hybrid"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATA_TYPE}"
mkdir -p ${OUTPUT_DIR}

# Model name for output file: {short_name}_{checkpoint}
FULL_MODEL_NAME="${MODEL_SHORT_NAME}_${CHECKPOINT}"

echo "Running: Model=${MODEL_PATH}, prefix_mode=${PREFIX_MODE}, output_name=${FULL_MODEL_NAME}"

cd /fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation

# Build the prefix flag based on mode
if [ "$PREFIX_MODE" = "dna_tags" ]; then
    PREFIX_FLAG="--use_dna_tags"
elif [ "$PREFIX_MODE" = "no_prefix" ]; then
    PREFIX_FLAG="--no_prefix"
else
    PREFIX_FLAG=""
fi

python sequence_recovery_eval.py \
    --model ${MODEL_PATH} \
    --model_name ${FULL_MODEL_NAME} \
    --data_type ${DATA_TYPE} \
    --data_path hf://datasets/GenerTeam/sequence-recovery \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 64 \
    --max_seq_len 6144 \
    --gen_len 5 \
    --gen_len_bp 30 \
    --bf16 \
    ${PREFIX_FLAG}

echo "Job completed!"
