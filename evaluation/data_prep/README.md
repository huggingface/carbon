# Dataset construction

Scripts to rebuild the VEP parquets from primary sources. **You don't need to
run these to use the evals** — every script in the parent folder defaults to
the prebuilt parquets on the Hub. These scripts are here for reproducibility:
they document the exact filters, windowing, and column schema we used.

| Script | Sources | Hub destination |
|---|---|---|
| [`prep_brca2.py`](prep_brca2.py) | [Huang et al. 2025, Nature](https://www.nature.com/articles/s41586-024-08388-8) Table S3 + chr13 hg19 (UCSC) | [`HuggingFaceBio/brca2-vep`](https://huggingface.co/datasets/HuggingFaceBio/brca2-vep) |
| [`prep_traitgym.py`](prep_traitgym.py) | [`songlab/TraitGym`](https://huggingface.co/datasets/songlab/TraitGym) + hg38 (UCSC) | [`HuggingFaceBio/traitgym`](https://huggingface.co/datasets/HuggingFaceBio/traitgym) |

Both produce the same schema (`chrom, pos, ref, alt, score, class,
ref_seq, var_seq`) so the parent [`vep_eval.py`](../vep_eval.py) reads
either without changes.

```bash
python prep_brca2.py --push_to_hub
python prep_traitgym.py --config mendelian_traits --push_to_hub
```

ClinVar has no prep script here because the production eval
([`clinvar_vep_eval.py`](../clinvar_vep_eval.py)) pulls
[`HuggingFaceBio/clinvar-vep-final`](https://huggingface.co/datasets/HuggingFaceBio/clinvar-vep-final)
directly — that dataset is GenerTeam's ClinVar release (mostly coding)
augmented with a Carbon-curated noncoding split.
