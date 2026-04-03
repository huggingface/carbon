#!/bin/bash
#
# Full-scale FineWeb-Edu DNA/RNA Filtering Pipeline
# Dataset: HuggingFaceFW/fineweb-edu (7.2TB, 15,642 arrow files)
# Date: 2026-01-27
#
# This script runs a two-stage filtering pipeline on the full FineWeb-Edu dataset:
# - Stage 1: Strict regex filter (fast, high-confidence DNA/RNA sequences)
# - Stage 2: ML classifier (0.9 threshold) on remaining documents
#
# Results are saved to: s3://hf-carbon/fineweb-edu-filtered-20260127/
# Logs are saved to: /fsx/dana_aubakirova/carbon_project/logs/fineweb-edu-filtered-20260127/
#
# Estimated completion: <24 hours with 1000 parallel tasks
#

set -e

# Activate conda environment
conda activate carbon-env

# Change to the pipeline directory
cd /fsx/dana_aubakirova/carbon_project/carbon/data-retrieval/two-stage-pipeline

echo "=========================================="
echo "FineWeb-Edu Full Dataset Filtering"
echo "=========================================="
echo "Dataset: HuggingFaceFW/fineweb-edu"
echo "Size: 7.2TB (15,642 arrow files)"
echo "Tasks: 200 parallel tasks (~78 files per task)"
echo "Output: s3://hf-carbon/fineweb-edu-filtered-20260127/"
echo "=========================================="
echo ""

# Create logging directory
mkdir -p /fsx/dana_aubakirova/carbon_project/logs/fineweb-edu-filtered-20260127

# Run Stage 1: Strict Regex Filter
echo "Submitting Stage 1: Strict Regex Filter"
echo "- Partition: hopper-cpu"
echo "- Time: 23 hours"
echo "- Resources: 200 tasks × 8 workers × 4 CPUs × 8GB RAM"
echo ""
python two_stage_pipeline.py --run stage1

echo ""
echo "✓ Stage 1 submitted successfully!"
echo ""
echo "To monitor progress:"
echo "  tail -f /fsx/dana_aubakirova/carbon_project/logs/fineweb-edu-filtered-20260127/stage1_strict_regex/*/log.log"
echo ""
echo "To check job status:"
echo "  squeue -u dana_aubakirova"
echo ""
echo "To view statistics after completion:"
echo "  python two_stage_pipeline.py --stats"
echo ""

# Uncomment to also run Stage 2 immediately
# echo "Submitting Stage 2: ML Classifier"
# echo "- Partition: hopper-prod (GPU)"
# echo "- Time: 23 hours"
# echo "- Resources: 200 tasks × 8 workers × 4 CPUs × 16GB RAM × 1 GPU"
# echo ""
# python two_stage_pipeline.py --run stage2
# echo ""
# echo "✓ Stage 2 submitted successfully!"

