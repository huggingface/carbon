#!/bin/bash
#SBATCH --job-name=repro-hf-vep
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

# BRCA1 + BRCA2 + TraitGym Mendelian on Carbon 3B and GENERator v2 3B.
# 6 vep_eval.py runs total, multi-GPU sharded across 8 GPUs each.

set -euo pipefail
source ~/.bashrc
source /fsx/loubna/.venv-2-6-cu124/bin/activate

EVAL=/fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/evaluation
OUT=$SLURM_SUBMIT_DIR/results

mkdir -p $OUT/{carbon-3B,gen-3b}/{brca1,brca2,traitgym}

CARBON=hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1
GEN=GenerTeam/GENERator-v2-eukaryote-3b-base
BRCA1=hf://datasets/hf-carbon/brca1-vep/brca1_vep.parquet
BRCA2=hf://datasets/hf-carbon/brca2-vep/brca2_vep.parquet
TRAITGYM=hf://datasets/hf-carbon/traitgym/mendelian_traits_vep.parquet

echo "=== Carbon 3B · BRCA1 ==="
python $EVAL/vep_eval.py --model $CARBON --data_path $BRCA1 --add_dna_tag --bf16 --output_dir $OUT/carbon-3B/brca1

echo "=== Carbon 3B · BRCA2 ==="
python $EVAL/vep_eval.py --model $CARBON --data_path $BRCA2 --add_dna_tag --bf16 --output_dir $OUT/carbon-3B/brca2

echo "=== Carbon 3B · TraitGym Mendelian ==="
python $EVAL/vep_eval.py --model $CARBON --data_path $TRAITGYM --add_dna_tag --bf16 --rev_comp_avg --output_dir $OUT/carbon-3B/traitgym

echo "=== GENERator 3B · BRCA1 ==="
python $EVAL/vep_eval.py --model $GEN --data_path $BRCA1 --bf16 --output_dir $OUT/gen-3b/brca1

echo "=== GENERator 3B · BRCA2 ==="
python $EVAL/vep_eval.py --model $GEN --data_path $BRCA2 --bf16 --output_dir $OUT/gen-3b/brca2

echo "=== GENERator 3B · TraitGym Mendelian ==="
python $EVAL/vep_eval.py --model $GEN --data_path $TRAITGYM --bf16 --rev_comp_avg --output_dir $OUT/gen-3b/traitgym

echo "=== DONE ==="
