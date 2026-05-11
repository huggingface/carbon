#!/bin/bash
#SBATCH --job-name=repro-hf-sr
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00

# Sequence recovery (eukaryote split) for Carbon 3B and GENERator v2 3B,
# 8-GPU sharded.

set -euo pipefail
source ~/.bashrc
source /fsx/loubna/.venv-2-6-cu124/bin/activate

EVAL=/fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/evaluation
OUT=$SLURM_SUBMIT_DIR/results

mkdir -p $OUT/{carbon-3B,gen-3b}/sr

echo "=== Carbon 3B · SR eukaryote ==="
python $EVAL/sequence_recovery.py \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --data_type eukaryote --add_dna_tag --bf16 \
    --output_dir $OUT/carbon-3B/sr

echo "=== GENERator 3B · SR eukaryote ==="
python $EVAL/sequence_recovery.py \
    --model GenerTeam/GENERator-v2-eukaryote-3b-base \
    --data_type eukaryote --bf16 \
    --output_dir $OUT/gen-3b/sr

echo "=== DONE ==="
