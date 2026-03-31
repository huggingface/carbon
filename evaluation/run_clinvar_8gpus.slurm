#!/bin/bash
#SBATCH --output=/fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation/slurm_logs/clinvar_eval_%A.out
#SBATCH --error=/fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation/slurm_logs/clinvar_eval_%A.err
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --qos=high

# Usage: sbatch run_clinvar_eval.sbatch <model_path> <prefix_mode> <model_name> [context_length]
# prefix_mode: "dna_tags" (hybrid), "no_prefix" (6-mer raw), or "default" (uses <s>)
# context_length: optional, in bp (default: 96000)
source /fsx/loubna/.venv-2-6-cu124/bin/activate
export HF_HUB_TRUST_REMOTE_CODE=1

MODEL_PATH=$1
PREFIX_MODE=$2
MODEL_NAME=$3
CONTEXT_LENGTH=${4:-96000}

OUTPUT_DIR="./eval_results/clinvar_vep"
mkdir -p ${OUTPUT_DIR}

echo "Running ClinVar VEP: Model=${MODEL_PATH}, prefix_mode=${PREFIX_MODE}, model_name=${MODEL_NAME}"

cd /fsx/loubna/projects_v2/carbon/loubna-workspace/evaluation/carbon/evaluation

# Build the prefix flag based on mode
if [ "$PREFIX_MODE" = "dna_tags" ]; then
    PREFIX_FLAG="--use_dna_tags"
elif [ "$PREFIX_MODE" = "no_prefix" ]; then
    PREFIX_FLAG="--no_prefix"
else
    PREFIX_FLAG=""
fi

python clinvar_vep_eval.py \
    --model ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --context_length ${CONTEXT_LENGTH} \
    --batch_size 4 \
    --bf16 \
    ${PREFIX_FLAG}

echo "ClinVar eval completed!"
