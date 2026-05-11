#!/bin/bash
#SBATCH --job-name=repro-carbon-clinvar
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

# Carbon 3B 1T-mix2 · ClinVar coding + non_coding. Two separate parquets on
# the Hub, run sequentially. 24 kb left context.

set -euo pipefail
source ~/.bashrc
source /fsx/loubna/.venv-2-6-cu124/bin/activate

EVAL=/fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/evaluation
OUT=$SLURM_SUBMIT_DIR/results/carbon-3B/clinvar
mkdir -p $OUT/coding $OUT/non_coding

MODEL=hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1
CODING=hf://datasets/hf-carbon/clinvar-vep-final/coding/clinvar_coding.parquet
NONCODING=hf://datasets/hf-carbon/clinvar-vep-final/non_coding/clinvar_non_coding.parquet

echo "=== Carbon 3B · ClinVar CODING ==="
python $EVAL/clinvar_vep_eval.py \
    --model $MODEL --clinvar_path $CODING \
    --add_dna_tag --bf16 --context_length 24000 \
    --output_dir $OUT/coding

echo "=== Carbon 3B · ClinVar NON-CODING ==="
python $EVAL/clinvar_vep_eval.py \
    --model $MODEL --clinvar_path $NONCODING \
    --add_dna_tag --bf16 --context_length 24000 \
    --output_dir $OUT/non_coding

echo "=== DONE ==="
