#!/bin/bash
#SBATCH --job-name=repro-evo2-vep
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# BRCA1 + BRCA2 + TraitGym Mendelian on Evo2 1B (single GPU).

set -euo pipefail
source ~/.bashrc
source /fsx/loubna/projects_v2/.evo2-cu124/bin/activate

export CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_TRUST_REMOTE_CODE=1
NVIDIA_LIBS=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')")
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}

EVAL=/fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/evaluation
OUT=$SLURM_SUBMIT_DIR/results/evo2-1b
mkdir -p $OUT/brca1 $OUT/brca2 $OUT/traitgym

MODEL=evo2_1b_base
BRCA1=hf://datasets/hf-carbon/brca1-vep/brca1_vep.parquet
BRCA2=hf://datasets/hf-carbon/brca2-vep/brca2_vep.parquet
TRAITGYM=hf://datasets/hf-carbon/traitgym/mendelian_traits_vep.parquet

echo "=== Evo2 1B · BRCA1 ==="
python $EVAL/vep_eval.py --model $MODEL --backend evo2 --data_path $BRCA1 --bf16 --output_dir $OUT/brca1

echo "=== Evo2 1B · BRCA2 ==="
python $EVAL/vep_eval.py --model $MODEL --backend evo2 --data_path $BRCA2 --bf16 --output_dir $OUT/brca2

echo "=== Evo2 1B · TraitGym Mendelian (--rev_comp_avg) ==="
python $EVAL/vep_eval.py --model $MODEL --backend evo2 --data_path $TRAITGYM --bf16 --rev_comp_avg --output_dir $OUT/traitgym

echo "=== DONE ==="
