# SLURM scripts

Reference SLURM scripts the Carbon team uses to run the eval suite on the
**internal HF cluster**. Paths (`/fsx/loubna/...`, partition `hopper-prod`,
etc.) are HF-specific — adapt them for your setup if needed.

One folder per model family, each with the same six scripts:

```
carbon-3B/          → HuggingFaceBio/Carbon-3B   (also lc32k variants)
generator-v2-3b/    → GenerTeam/GENERator-v2-eukaryote-3b-base
evo2-1b/            → evo2_1b_base
evo2-7b/            → evo2_7b_base
```

| Script | Eval |
|---|---|
| `sequence_recovery.sbatch` | Sequence recovery, eukaryote split |
| `vep_brca2.sbatch` | BRCA2 |
| `vep_traitgym.sbatch` | TraitGym Mendelian (with `--rev_comp_avg`) |
| `clinvar.sbatch` | ClinVar coding + non_coding (default 24 kb, `CONTEXT_LENGTH=48000` for 48 kb) |
| `perturbation_tasks.sbatch` | TATA + synonymous codons |
| `genome_niah.sbatch` | Long-context retrieval. Defaults to `TASK=niah CTX=32768`. Override via env: see script header for a sweep example. |

Launch one model:
```bash
cd evaluation/slurm/carbon-3B
for f in *.sbatch; do sbatch "$f"; done
```

Sweep Genome-NIAH across all 4 tasks × 4 short contexts (Carbon, ~3h total):
```bash
cd evaluation/slurm/carbon-3B
for TASK in niah neardup_d4 neardup_d2 neardup_d1; do
  for CTX in 4096 8192 16384 32768; do
    TASK=$TASK CTX=$CTX sbatch genome_niah.sbatch
  done
done
```
