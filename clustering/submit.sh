#!/bin/bash
# Submit clustering jobs for two datasets on separate nodes.
#
# Usage:
#   BASE=/path/to/clean/data bash submit.sh
#
# Environment:
#   BASE         – parent directory containing dataset subfolders (required)
#   OUTPUT_ROOT  – where to write results (default: ./output)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM="${SCRIPT_DIR}/run_clustering.slurm"
BASE="${BASE:?BASE env var must be set to the parent directory of processed_regex / processed_snowflake}"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

JOB1=$(sbatch --job-name=cluster-regex \
    --export=ALL,DATASET_DIR="${BASE}/processed_regex",DATASET_NAME="processed_regex" \
    "$SLURM" | awk '{print $4}')
echo "Submitted regex job: $JOB1"

# Wait briefly, read the assigned node, then exclude it for the second job
# so the two memory-heavy jobs don't compete on the same machine.
sleep 10
NODE1=$(squeue -j "$JOB1" -h -o "%N" 2>/dev/null || echo "")

if [ -n "$NODE1" ] && [ "$NODE1" != "None" ]; then
    echo "Regex job on node: $NODE1 — excluding for snowflake job"
    EXTRA="--exclude=${NODE1}"
else
    EXTRA=""
fi

JOB2=$(sbatch --job-name=cluster-snowflake \
    $EXTRA \
    --export=ALL,DATASET_DIR="${BASE}/processed_snowflake",DATASET_NAME="processed_snowflake" \
    "$SLURM" | awk '{print $4}')

echo "Submitted snowflake job: $JOB2"
echo "Both jobs submitted: regex=${JOB1}  snowflake=${JOB2}"
