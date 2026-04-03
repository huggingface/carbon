#!/bin/bash
# Polls for regex scan completion and submits enrichment jobs automatically.
# Run with: nohup bash watch_and_enrich.sh > /fsx/dana_aubakirova/carbon_project/logs/watch_and_enrich.log 2>&1 &

set -euo pipefail

PYTHON=/fsx/dana_aubakirova/miniconda/bin/python
ENRICH_SCRIPT=/fsx/dana_aubakirova/carbon_project/carbon/data/coord-enrichment/enrich_pipeline.py
LOGS=/fsx/dana_aubakirova/carbon_project/logs

declare -A SCAN_LOG_DIRS=(
    [regex_biorxiv]="genomic_coords_biorxiv_meca_regex_20260225"
    [regex_biostars]="genomic_coords_biostars_regex_20260225"
    [regex_finepdfs]="genomic_coords_finepdfs_regex_outliers_removed_20260303"
    [regex_fineweb]="genomic_coords_fineweb_regex_snowflake_filtered_20260303"
    [regex_pubmed]="genomic_coords_pubmed_regex_snowflake_filtered_20260303"
)

declare -A TASKS=(
    [regex_biorxiv]=20
    [regex_biostars]=20
    [regex_finepdfs]=20
    [regex_fineweb]=20
    [regex_pubmed]=20
)

submitted=()

echo "[$(date)] Watching for regex scan completions..."

while [ ${#submitted[@]} -lt 5 ]; do
    for key in regex_biorxiv regex_biostars regex_finepdfs regex_fineweb regex_pubmed; do
        # Skip already submitted
        if [[ " ${submitted[*]} " =~ " ${key} " ]]; then continue; fi

        log_dir="$LOGS/${SCAN_LOG_DIRS[$key]}"
        n_tasks=${TASKS[$key]}
        n_complete=$(ls "$log_dir/completions/" 2>/dev/null | wc -l)

        echo "[$(date)] $key: $n_complete/$n_tasks completions"

        if [ "$n_complete" -ge "$n_tasks" ]; then
            echo "[$(date)] ✓ $key scan done — submitting enrichment..."
            cd /fsx/dana_aubakirova/carbon_project/carbon/data/coord-enrichment
            $PYTHON "$ENRICH_SCRIPT" --run "$key" 2>&1
            submitted+=("$key")
            echo "[$(date)] ✓ $key enrichment submitted"
        fi
    done

    if [ ${#submitted[@]} -lt 5 ]; then
        sleep 120
    fi
done

echo "[$(date)] All regex enrichment jobs submitted: ${submitted[*]}"
