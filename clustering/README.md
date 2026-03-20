# Dataset clustering

Clusters local `.jsonl.gz` datasets using sentence-transformer embeddings
(all-MiniLM-L6-v2), UMAP dimensionality reduction, and DBSCAN.

Biology benchmarks (LabBench, MMLU-Bio) are embedded **jointly** with the
training texts so they land in the same UMAP space. Cluster assignments are
computed on training texts only; benchmark positions are saved separately for
overlay visualisation.

---

## Dependencies

```bash
pip install sentence-transformers umap-learn hdbscan faiss-cpu \
            pyarrow pandas matplotlib tqdm
```

The clustering library from
[huggingface/text-clustering](https://github.com/huggingface/text-clustering)
must be on `PYTHONPATH` (or set `TEXT_CLUSTERING_SRC`):

```bash
git clone https://github.com/huggingface/text-clustering
export TEXT_CLUSTERING_SRC=/path/to/text-clustering
```

---

## 1 — Download the datasets from HuggingFace Hub

### Regex-filtered data
Collection: https://huggingface.co/collections/hf-carbon/clean-midtraining-engdna-data-regex

```bash
huggingface-cli download hf-carbon/fineweb-edu-regex     --repo-type dataset --local-dir data/processed_regex/fineweb_edu
huggingface-cli download hf-carbon/fineweb-pdf-en-regex  --repo-type dataset --local-dir data/processed_regex/fineweb_pdf_en
huggingface-cli download hf-carbon/pubmed-regex          --repo-type dataset --local-dir data/processed_regex/pubmed
huggingface-cli download hf-carbon/biorxiv-regex         --repo-type dataset --local-dir data/processed_regex/biorxiv
huggingface-cli download hf-carbon/biostars-regex        --repo-type dataset --local-dir data/processed_regex/biostars
```

### Bio-classifier–filtered data
Collection: https://huggingface.co/collections/hf-carbon/clean-midtraining-engdna-data-bio-classifier

```bash
huggingface-cli download hf-carbon/fineweb-edu-bio-threshold    --repo-type dataset --local-dir data/processed_snowflake/fineweb_edu
huggingface-cli download hf-carbon/fineweb-pdf-en-bio-threshold --repo-type dataset --local-dir data/processed_snowflake/fineweb_pdf_en
huggingface-cli download hf-carbon/pubmed-bio-threshold         --repo-type dataset --local-dir data/processed_snowflake/pubmed
huggingface-cli download hf-carbon/biorxiv-bio-threshold        --repo-type dataset --local-dir data/processed_snowflake/biorxiv
huggingface-cli download hf-carbon/biostars-bio-threshold       --repo-type dataset --local-dir data/processed_snowflake/biostars
```

Each subdirectory should contain `.jsonl.gz` shards with a `text` field and
`metadata.token_count`.

---

## 2 — Download benchmark data (optional, for overlay plots)

```python
from datasets import load_dataset

# LabBench
load_dataset("hf-carbon/lab-bench", "CloningScenarios")
load_dataset("hf-carbon/lab-bench", "SeqQA")

# MMLU-Bio
load_dataset("cais/mmlu", "college_biology")
load_dataset("cais/mmlu", "high_school_biology")
load_dataset("hf-carbon/mmlu-pro-biology")
load_dataset("hf-carbon/mmlu-redux-2.0-biology", "college_biology")
load_dataset("hf-carbon/mmlu-redux-2.0-biology", "medical_genetics")
load_dataset("hf-carbon/mmlu-redux-2.0-biology", "high_school_biology")
```

Set `HF_DATASETS_CACHE` to control where the cache is written:

```bash
export HF_DATASETS_CACHE=/path/to/cache
```

---

## 3 — Run clustering

### Local (single dataset)

```bash
export TEXT_CLUSTERING_SRC=/path/to/text-clustering
export HF_DATASETS_CACHE=/path/to/cache

python cluster_local.py \
    --dataset_dir  data/processed_regex \
    --dataset_name processed_regex \
    --n_per_subfolder 50000 \
    --output_root  output
```

### SLURM (two datasets in parallel on separate nodes)

```bash
export BASE=/path/to/clean/data   # parent of processed_regex / processed_snowflake
bash submit.sh
```

Or submit a single job manually:

```bash
sbatch \
  --export=ALL,DATASET_DIR=/path/to/processed_regex,DATASET_NAME=processed_regex \
  run_clustering.slurm
```

Key environment variables for the SLURM job:

| Variable | Default | Description |
|---|---|---|
| `DATASET_DIR` | — (required) | Path to dataset root |
| `DATASET_NAME` | — (required) | Name for the output subdirectory |
| `OUTPUT_ROOT` | `./output` | Where to write results |
| `N_PER_SUBFOLDER` | `50000` | Texts sampled per subfolder |
| `DBSCAN_EPS` | `0.15` | DBSCAN neighbourhood radius |
| `DBSCAN_MIN_SAMPLES` | `50` | DBSCAN minimum core points |
| `MIN_TOKENS` | `64` | Minimum token count to keep a document |
| `MAX_TOKENS` | `4096` | Maximum token count to keep a document |
| `TEXT_CLUSTERING_SRC` | `/fsx/.../text-clustering` | Path to clustering library |
| `HF_DATASETS_CACHE` | `/fsx/.../.cache` | HuggingFace datasets cache |

---

## 4 — Outputs

Results are written to `output/{dataset_name}/clusters_eps{eps}/`:

```
cluster_labels.npy          # per-text cluster IDs (-1 = noise)
cluster_summaries.json      # {cluster_id: "3-word topic label"}
cluster_stats.json          # n_clusters, noise_pct, sizes, …
cluster_plot.png            # UMAP scatter (training data only)
cluster_plot_labbench.png   # UMAP + LabBench overlay
cluster_plot_mmlu_bio.png   # UMAP + MMLU-Bio overlay
embeddings.npy              # sentence-transformer embeddings
projections.npy             # 2-D UMAP coordinates
faiss.index                 # FAISS nearest-neighbour index
texts.json                  # sampled texts
samples_clustered.jsonl     # full docs with cluster_id + cluster_name
labbench_projections.npy    # LabBench 2-D UMAP coordinates
mmlu_bio_projections.npy    # MMLU-Bio 2-D UMAP coordinates
```
