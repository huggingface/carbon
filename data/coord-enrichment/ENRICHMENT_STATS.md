# Genomic Coordinate Enrichment — Dataset Summary

## Pipeline Overview

Documents from `processed_snowflake` and `processed_regex` were scanned for explicit genomic coordinate references in two formats:

- **UCSC**: `chr17:43,044,295-43,170,245`
- **Ensembl**: `17:43044295-43170245` (≥6 digits to exclude antibody dilutions and journal citations)

For each coordinate found, the [Ensembl REST API](https://rest.ensembl.org) was queried to identify the overlapping gene and fetch the local DNA sequence. A compact annotation was inserted **inline immediately after** the coordinate in the document text:

```
chr3:1395000-1400000 [gene=CNTN6 (protein_coding) | strand=+ | seq=TTAACTAAATCAAACAAGAAA...(...) | assembly=GRCh38]
```

## Results

| Dataset | Type | Docs enriched | Docs w/ ≥1 annotation | Coordinate refs found | Annotations inserted | Unique genes |
|---|---|---:|---:|---:|---:|---:|
| biorxiv | snowflake | 2,471 | 2,198 (89.0%) | 13,195 | 6,187 (46.9%) | 2,484 |
| biostars | snowflake | 1,022 | 899 (88.0%) | 5,774 | 2,680 (46.4%) | 954 |
| finepdfs | snowflake | 42 | 37 (88.1%) | 204 | 98 (48.0%) | 57 |
| pipeline_t2 | snowflake | 3,122 | 2,924 (93.7%) | 12,578 | 5,906 (47.0%) | 679 |
| pubmed | snowflake | 15 | 15 (100.0%) | 34 | 17 (50.0%) | 4 |
| biorxiv | regex | 683 | 630 (92.2%) | 3,271 | 1,544 (47.2%) | 657 |
| biostars | regex | 159 | 153 (96.2%) | 903 | 431 (47.7%) | 149 |
| finepdfs | regex | 2,087 | 1,936 (92.8%) | 56,380 | 25,235 (44.8%) | 8,116 |
| fineweb | regex | 532 | 499 (93.8%) | 2,540 | 1,229 (48.4%) | 140 |
| pubmed | regex | 4,234 | 4,047 (95.6%) | 42,752 | 20,432 (47.8%) | 7,234 |
| **TOTAL** | | **14,367** | **13,338 (92.8%)** | **137,631** | **63,759 (46.3%)** | **~19,000+** |

## Notes

- **~46% annotation rate** on coordinates is expected — roughly half of genomic coordinates fall in intergenic regions with no overlapping gene returned by Ensembl.
- **92.8% doc coverage** — nearly every document that passed the coordinate filter received at least one annotation.
- **regex finepdfs** and **regex pubmed** dominate by volume, contributing ~99k of the 137k total coordinate references.
- `biostars` has the highest hit rate among source datasets (~1.6–4.5%) as it is a genomics Q&A forum where users frequently paste raw coordinates.

## Output Locations

| Type | Path |
|---|---|
| snowflake enriched | `/fsx/dana_aubakirova/carbon_data/clean/processed_snowflake_enriched/` |
| regex enriched | `/fsx/dana_aubakirova/carbon_data/clean/processed_regex_enriched/` |
| scan outputs (coords only) | `/fsx/dana_aubakirova/carbon_data/clean/processed_snowflake_genomic_coords/` |

## Key Files

| File | Description |
|---|---|
| `carbon/data-retrieval/two-stage-pipeline/scan_genomic_coords.py` | Scans all subsets for coordinate patterns |
| `carbon/data/coord-enrichment/fetcher.py` | Calls Ensembl API, builds annotations |
| `carbon/data/coord-enrichment/enrich_pipeline.py` | Datatrove pipeline that runs enrichment on Slurm |
| `carbon/data-retrieval/two-stage-pipeline/two_stage_pipeline.py` | Contains `GenomicCoordFilter` class |
