# Data Retrieval Pipeline

This directory contains DNA/RNA sequence filters for processing the FineWeb-EDU dataset to identify biological content.

## Directory Structure

```
data-retrieval/
├── two-stage-pipeline/
│   └── two_stage_pipeline.py       # DNA/RNA filter implementations (V4, V5, LLM)
├── classifier-finetuning/           # Classifier training code
│   ├── finetune_distilbert.py      # Main DistilBERT training script
│   ├── prepare_training_data.py    # Training data preparation
│   ├── expand_false_positives.py   # False positive augmentation
│   ├── run_finetuning.sh           # Training execution script
│   └── training_data/              # Training and test datasets
└── samples/                         # Sample outputs
```

## Filter Implementations

### DNAFilterV4Strict
**High-precision regex filter for DNA/RNA sequences**

- **Requirements**:
  - 15+ base sequences (`ATCGUN`)
  - 3+ unique bases
  - 1+ biological keyword (dna, rna, gene, etc.)
  - Excludes LaTeX sequences (in braces)
- **Use case**: High-confidence biological sequence detection
- **Performance**: ~0.01-0.05% pass rate on FineWeb-EDU
- **Speed**: Very fast (CPU-only regex matching)

### DNAFilterV5Expanded
**Extended filter with relaxed requirements and categorization**

- **Features**:
  - Relaxed to 12+ bases with 3+ unique (was 15+)
  - Expanded keyword lists (molecular biology, lab techniques)
  - Exclusion filters (clinical medicine, SEO, false positives)
  - Document categorization (A/B/C)
  
- **Categories**:
  - **Category A (bio_text)**: Molecular biology text without explicit sequences
  - **Category B (bio_sequence)**: Sequences with minimal explanation
  - **Category C (interleaved)**: Sequences + natural language explanation (TARGET)

- **Current configuration (V5d)**: Only returns Category C documents
  - Requires both sequences AND substantial explanation
  - Optimal for training language models on biological protocols
- **Speed**: Fast (CPU-only, more complex regex patterns than V4)

### LLMClassifier
**Transformer-based classification using fine-tuned DistilBERT**

- **Model**: DistilBERT fine-tuned on DNA/RNA sequences
- **Threshold**: 0.9 (configurable, high confidence)
- **Method**: Deep learning classifier trained on biological content
- **Features**:
  - Understands context and semantic meaning
  - Catches sequences that don't match strict regex patterns but still have general biology content
  - GPU-accelerated inference
- **Use case**: Higher recall than regex while maintaining precision
- **Speed**: ~200 docs/second on GPU
- **Trained model**: `/fsx/dana_aubakirova/carbon_project/models/finetuned_distilbert_1405`

## Running the Filters

### Test Runs (Small-scale testing)

```bash
# V5 test on 100K documents
python two-stage-pipeline/two_stage_pipeline.py --run v5_test

# V4 test on 400K documents
python two-stage-pipeline/two_stage_pipeline.py --run v4_test

# LLM classifier (DistilBERT) test on 100K documents on hopper prod (use GPU for faster processing)
python two-stage-pipeline/two_stage_pipeline.py --run llm_test

# Analyze V5 results (category breakdown, samples)
python two-stage-pipeline/two_stage_pipeline.py --analyze-v5
```

### Full Dataset Runs

```bash
# V5 on full 10BT dataset (~9.7M documents)
python two-stage-pipeline/two_stage_pipeline.py --run v5_full

# V4 on full FineWeb-EDU (1.5B documents, 2410 parquet files)
python two-stage-pipeline/two_stage_pipeline.py --run v4_full
```

**Note**: The LLM classifier requires GPU resources but offers higher recall than regex-only approaches. For fastest processing, use V4 or V5 regex filters (CPU-only).

## Dataset Paths

- **10BT subset**: `/fsx/leandro/data/fineweb-edu-10bt/sample/10BT` (14 parquet files, ~9.7M docs)
- **Full FineWeb-EDU**: `/fsx/dana_aubakirova/.cache/datasets--HuggingFaceFW--fineweb-edu/...` (2410 parquet files, 1.5B docs)

## Output Locations

### Test Runs
- **V5 test (100K)**: `/fsx/dana_aubakirova/carbon_project/data/v5_final_100k`
- **V4 test (400K)**: `/fsx/dana_aubakirova/carbon_project/data/v4_strict_400k`
- **LLM test (100K)**: `/fsx/dana_aubakirova/carbon_project/data/llm_classifier_test`

### Full Runs
- **V5 full (10BT)**: `s3://hf-carbon/v5_final_10BT_20260127`
- **V4 full (FineWeb-EDU)**: `s3://hf-carbon/fineweb-edu-v4-filtered-full-20260127`

## Performance Characteristics

### DNAFilterV4Strict
- **Speed**: Very fast (CPU-only)
- **Precision**: High
- **Recall**: Moderate (strict requirements)
- **Throughput**: ~1000-2000 docs/second
- **Cluster config**: 
  - 20 tasks for full dataset
  - 8 workers per task
  - 4 CPUs per task, 8GB RAM per CPU
  - ~48 hours for 1.5B documents

### DNAFilterV5Expanded
- **Speed**: Fast (CPU-only, more regex patterns)
- **Precision**: High (with exclusion filters)
- **Recall**: Higher than V4 (relaxed requirements)
- **Throughput**: ~500-1000 docs/second
- **Cluster config**:
  - 14 tasks for 10BT (1 per parquet file)
  - 8 workers per task
  - 8 CPUs per task, 8GB RAM per CPU
  - ~4 hours for 9.7M documents

### LLMClassifier (DistilBERT)
- **Speed**: Fast (GPU-accelerated)
- **Precision**: High (trained on biological data)
- **Recall**: Higher than regex (semantic understanding)
- **Throughput**: ~200 docs/second on GPU
- **Model**: Fine-tuned DistilBERT transformer
- **Cluster config**:
  - 4 tasks for 100K test
  - 4 workers per task
  - 4 CPUs per task, 16GB RAM per CPU
  - 1 GPU per task (hopper-prod partition)
  - ~4 hours for 100K documents
- **Best for**: Higher recall than regex while maintaining precision

## Logging

Logs are stored in `/fsx/dana_aubakirova/carbon_project/logs/`:
- `v5_final_100k/` - V5 test logs
- `v4_strict_400k/` - V4 test logs
- `llm_classifier_test/` - LLM classifier test logs
- `v5_final_10BT/` - V5 full 10BT logs
- `fineweb-edu-v4-full/` - V4 full FineWeb-EDU logs

## Filter Statistics

The filters track and report:
- Total documents processed
- Documents passed/rejected
- Category breakdown (V5 only)
- Exclusion reasons (clinical, generic, false positives)
- Keyword counts and sequence statistics

Statistics are printed every 10,000 documents and saved to `stats.json` in the logging directory.
