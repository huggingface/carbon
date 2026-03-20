#!/bin/bash
#SBATCH --job-name=cluster-%x
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Required env vars (pass via --export):
#   DATASET_DIR   – path to directory with subfolders of .jsonl.gz shards
#   DATASET_NAME  – name used for the output subdirectory
#
# Optional env vars (with defaults):
#   OUTPUT_ROOT          – where to write results (default: ./output)
#   N_PER_SUBFOLDER      – texts sampled per subfolder (default: 50000)
#   DBSCAN_EPS           – DBSCAN neighbourhood radius (default: 0.15)
#   DBSCAN_MIN_SAMPLES   – DBSCAN minimum core points (default: 50)
#   MIN_TOKENS           – minimum token length to keep a doc (default: 64)
#   MAX_TOKENS           – maximum token length to keep a doc (default: 4096)
#   TEXT_CLUSTERING_SRC  – path to text-clustering library
#   HF_DATASETS_CACHE    – HuggingFace datasets cache (for benchmark data)

set -euo pipefail

PYTHON=${PYTHON:-python}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/cluster_local.py"

echo "========================================"
echo "Job:          $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Dataset dir:  ${DATASET_DIR}"
echo "Dataset name: ${DATASET_NAME}"
echo "Started:      $(date)"
echo "========================================"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

$PYTHON "$SCRIPT" \
    --dataset_dir  "${DATASET_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --output_root  "${OUTPUT_ROOT:-${SCRIPT_DIR}/output}" \
    --n_per_subfolder    "${N_PER_SUBFOLDER:-50000}" \
    --dbscan_eps         "${DBSCAN_EPS:-0.15}" \
    --dbscan_min_samples "${DBSCAN_MIN_SAMPLES:-50}" \
    --min_tokens         "${MIN_TOKENS:-64}" \
    --max_tokens         "${MAX_TOKENS:-4096}"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
