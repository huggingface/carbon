#!/bin/bash
#SBATCH --job-name=repro-perturb-all
#SBATCH --output=/fsx/loubna/logs/%x-%j.out
#SBATCH --error=/fsx/loubna/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00

# TATA + SYN on Carbon 3B, GENERator 3B, Evo2 1B — all chained on 1 GPU.
# Both tasks are extremely fast (small datasets, full-sequence likelihood
# scoring), so the 6 runs together fit in one GPU-hour.

set -euo pipefail
source ~/.bashrc

EVAL=/fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/evaluation
OUT=$SLURM_SUBMIT_DIR/results

mkdir -p $OUT/{carbon-3B,gen-3b,evo2-1b}/{tata,syn}

# -------- HF backend (Carbon 3B + GENERator v2 3B) --------
HF_PY=/fsx/loubna/.venv-2-6-cu124/bin/python

echo "=== Carbon 3B · TATA ==="
$HF_PY $EVAL/perturbation_tasks.py \
    --task tata_perturbation \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --add_dna_tag --bf16 \
    --output_dir $OUT/carbon-3B/tata

echo "=== Carbon 3B · SYN ==="
$HF_PY $EVAL/perturbation_tasks.py \
    --task synonymous_codon_substitution \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --add_dna_tag --bf16 \
    --output_dir $OUT/carbon-3B/syn

echo "=== GENERator v2 3B · TATA ==="
$HF_PY $EVAL/perturbation_tasks.py \
    --task tata_perturbation \
    --model GenerTeam/GENERator-v2-eukaryote-3b-base \
    --bf16 \
    --output_dir $OUT/gen-3b/tata

echo "=== GENERator v2 3B · SYN ==="
$HF_PY $EVAL/perturbation_tasks.py \
    --task synonymous_codon_substitution \
    --model GenerTeam/GENERator-v2-eukaryote-3b-base \
    --bf16 \
    --output_dir $OUT/gen-3b/syn

# -------- Evo2 backend (Evo2 1B) --------
source /fsx/loubna/projects_v2/.evo2-cu124/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_TRUST_REMOTE_CODE=1
NVIDIA_LIBS=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')")
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}

echo "=== Evo2 1B · TATA ==="
python $EVAL/perturbation_tasks.py \
    --task tata_perturbation \
    --model evo2_1b_base --backend evo2 --bf16 \
    --output_dir $OUT/evo2-1b/tata

echo "=== Evo2 1B · SYN ==="
python $EVAL/perturbation_tasks.py \
    --task synonymous_codon_substitution \
    --model evo2_1b_base --backend evo2 --bf16 \
    --output_dir $OUT/evo2-1b/syn

echo "=== DONE ==="
