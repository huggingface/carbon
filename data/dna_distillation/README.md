# DNA Distillation Data Pipeline

This directory contains the dataset generation pipeline for DNA distillation data built from `GenerTeam/pretrain_data_eukaryote`.

## What the script does

`generate_data.py` builds a balanced sample across the six top-level species folders:

- `fungi`
- `invertebrate`
- `plant`
- `protozoa`
- `vertebrate_mammalian`
- `vertebrate_other`

For each species, the pipeline:

1. Lists all `.parq` source files from the Hub dataset.
2. Scans every parquet file independently with `pyarrow`, instead of downloading the full dataset eagerly.
3. Filters out rows that cannot produce a valid prompt/completion pair.
4. Assigns each eligible row a deterministic sampling priority derived from the seed plus stable row identity.
5. Keeps the exact global top `--num-samples-per-species` rows per species by merging file-local candidates.
6. Materializes only the selected rows in a second pass.

After sampling all species, the pipeline:

1. Concatenates the sampled rows.
2. Optionally shuffles the full dataset with `--shuffle`.
3. Adds prompt-completion fields with `datasets.Dataset.map`.
4. Optionally pushes the final dataset to the Hub with `Dataset.push_to_hub`.

## Prompt and completion format

The script preserves the original source columns and adds:

- `source`: source dataset ID, currently `GenerTeam/pretrain_data_eukaryote`
- `type`: normalized species bucket
- `mapped_species_tag`: tokenizer tag derived from raw `species_type`
- `mapped_gene_tag`: tokenizer tag derived from raw `gene_type`
- `tag_mode`: one of `no_tags`, `both`, `species_only`, or `gene_only`
- `prompt`: metadata prefix plus `<dna>` followed by the rightmost usable DNA context
- `completion`: the last `completion_len` DNA base pairs from the source sequence
- `prompt_len`: DNA context length in base pairs, excluding the prefix text
- `completion_len`: per-row DNA completion length in base pairs

Prompt generation uses a mixed metadata ablation by default:

- `0.5`: `no_tags` → `<dna>SEQUENCE`
- `1/6`: `both` → `<species_tag><gene_tag><dna>SEQUENCE`
- `1/6`: `species_only` → `<species_tag><dna>SEQUENCE`
- `1/6`: `gene_only` → `<gene_tag><dna>SEQUENCE`

These fractions can be changed with:

- `--no-tags-frac`
- `--both-tags-frac`
- `--species-only-frac`
- `--gene-only-frac`

The four values are interpreted on a `0-1` scale and must sum to `1.0`.

For each row:

- the DNA context is truncated from the left, keeping the rightmost context window
- the kept context is rounded down to a multiple of 6 base pairs
- `completion_len` is sampled deterministically and uniformly from the inclusive range
  `[--min-completion-len, --max-completion-len]`
- if `--min-completion-len == --max-completion-len`, all rows use that constant length
- the metadata mode is assigned deterministically from `--seed` plus stable row identity, so results are reproducible across runs and process counts
- sampled row membership is exact and deterministic for a fixed seed, but it may differ from older reservoir-sampling runs with the same seed

Metadata tag mappings are:

- species:
  - `<mam>` → `<mammalian_species>`
  - `<vrt>` → `<vertebrate_non_mammalian_species>`
  - `<fng>` → `<fungi_species>`
  - `<pln>` → `<plant_species>`
  - `<prt>` → `<protozoa_species>`
  - `<inv>` → `<invertebrate_species>`
- gene:
  - `<cds>` → `<protein_coding_region>`
  - `<pseudo>` → `<pseudo_gene>`
  - `<tRNA>` → `<transfer_rna>`
  - `<tmRNA>` → `<transfer_messenger_rna>`
  - `<ncRNA>` → `<non_coding_rna>`
  - `<misc_RNA>` → `<miscellaneous_rna>`
  - `<rRNA>` → `<ribosomal_rna>`

## Eligibility rules

A row is eligible only if:

- `sequence` is present and is a string
- `len(sequence) >= sampled_completion_len + 6`
- at least one 6-bp chunk remains after reserving the completion suffix

Rows that fail these checks are skipped before exact sampling so the final per-species counts stay exact.

## How to run it

Dry run without pushing:

```sh
uv run --project data --script data/dna_distillation/generate_data.py \
    --num-samples-per-species 1024 \
    --prompt-len 6144 \
    --min-completion-len 30 \
    --max-completion-len 960 \
    --no-tags-frac 0.5 \
    --both-tags-frac 0.16666666666666666 \
    --species-only-frac 0.16666666666666666 \
    --gene-only-frac 0.16666666666666666 \
    --shuffle
```

Push a dataset to the Hub:

```sh
uv run --project data --script data/dna_distillation/generate_data.py \
    --num-samples-per-species 1024 \
    --prompt-len 6144 \
    --min-completion-len 30 \
    --max-completion-len 960 \
    --no-tags-frac 0.5 \
    --both-tags-frac 0.16666666666666666 \
    --species-only-frac 0.16666666666666666 \
    --gene-only-frac 0.16666666666666666 \
    --shuffle \
    --dataset-id hf-carbon/dna-distillation \
    --dataset-config default
```

## Notes

- `--prompt-len`, `--min-completion-len`, and `--max-completion-len` are measured in DNA characters / base pairs, not tokenizer tokens.
- Metadata fractions use the `-frac` suffix and are expressed on a `0-1` scale.
- `--num-proc` is used for both file-level sampling and the final `Dataset.map` post-processing step. The default is `16`.
- The sampler parallelizes across source parquet files, so values above `6` can improve throughput.
- If the script cannot find enough eligible rows for a species, it fails explicitly instead of silently returning fewer rows.
