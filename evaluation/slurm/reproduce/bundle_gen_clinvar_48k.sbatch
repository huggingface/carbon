#!/bin/bash
#SBATCH --job-name=repro-gen-clinvar48k
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

# GENERator v2 3B · ClinVar at 48 kb left context.

set -euo pipefail
source ~/.bashrc
source /fsx/loubna/.venv-2-6-cu124/bin/activate

EVAL=/fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/evaluation
OUT=$SLURM_SUBMIT_DIR/results/gen-3b/clinvar_48k
mkdir -p $OUT/coding $OUT/non_coding

MODEL=GenerTeam/GENERator-v2-eukaryote-3b-base
CODING=hf://datasets/hf-carbon/clinvar-vep-final/coding/clinvar_coding.parquet
NONCODING=hf://datasets/hf-carbon/clinvar-vep-final/non_coding/clinvar_non_coding.parquet

echo "=== GENERator 3B · ClinVar CODING @ 48 kb ==="
python $EVAL/clinvar_vep_eval.py \
    --model $MODEL --clinvar_path $CODING \
    --bf16 --context_length 48000 \
    --output_dir $OUT/coding

echo "=== GENERator 3B · ClinVar NON-CODING @ 48 kb ==="
python $EVAL/clinvar_vep_eval.py \
    --model $MODEL --clinvar_path $NONCODING \
    --bf16 --context_length 48000 \
    --output_dir $OUT/non_coding

echo "=== DONE ==="
