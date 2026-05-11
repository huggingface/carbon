# SLURM scripts

Reference SLURM scripts the Carbon team uses to run the eval suite on the
**internal HF cluster**. Paths (`/fsx/loubna/...`, partition `hopper-prod`,
etc.) are HF-specific — adapt them for your setup if needed.

One folder per model family, each with the same five scripts:

```
carbon-3B/          → hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1
generator-v2-3b/    → GenerTeam/GENERator-v2-eukaryote-3b-base
evo2-1b/            → evo2_1b_base
evo2-7b/            → evo2_7b_base
```

| Script | Eval |
|---|---|
| `sequence_recovery.sbatch` | Sequence recovery, eukaryote split |
| `vep_brca.sbatch` | BRCA1 then BRCA2 |
| `vep_traitgym.sbatch` | TraitGym Mendelian (with `--rev_comp_avg`) |
| `clinvar.sbatch` | ClinVar coding + non_coding (default 24 kb, `CONTEXT_LENGTH=48000` for 48 kb) |
| `perturbation_tasks.sbatch` | TATA + synonymous codons |

Launch one model:
```bash
cd evaluation/slurm/carbon-3B
for f in *.sbatch; do sbatch "$f"; done
```
