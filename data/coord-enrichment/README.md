# Genomic Coordinate Enrichment

Inserts compact gene + sequence annotations inline after every genomic coordinate reference found in a document.

**Before:**
```
...the region chr17:43,094,000-43,094,500 harbours a critical exon...
```
**After:**
```
...the region chr17:43,094,000-43,094,500 [gene=BRCA1 (protein_coding) | strand=- | seq=CTGATGTAGGT...(...) | assembly=GRCh38] harbours a critical exon...
```

Annotations are fetched from the Ensembl REST API — no LLMs involved.

## Usage

**Step 1 — Scan for coordinates:**
```bash
cd data-retrieval/two-stage-pipeline
python scan_genomic_coords.py --run biorxiv   # or: biostars | finepdfs | pipeline_t2 | pubmed | regex_* | all
```
Output: `processed_snowflake_genomic_coords/<subset>/`

**Step 2 — Enrich:**
```bash
cd data/coord-enrichment
python enrich_pipeline.py --run biorxiv   # or: biostars | finepdfs | pipeline_t2 | pubmed | regex_* | all
```
Output: `processed_snowflake_enriched/<subset>/` or `processed_regex_enriched/<subset>/`

## Files

| File | Description |
|------|-------------|
| `fetcher.py` | Parses coordinates, queries Ensembl, builds annotations |
| `enrich_pipeline.py` | Slurm pipeline wrapping `fetcher.py` |
| `watch_and_enrich.sh` | Auto-submits enrichment once scans complete |
| `ENRICHMENT_STATS.md` | Result stats across all processed subsets |

## Notes

- Ensembl enforces **15 req/s per IP** — task counts are kept low accordingly
- ~46% of coordinates get annotated (the rest fall in intergenic regions)
- Requires `data/synth-data/common.py` for sequence fetching
