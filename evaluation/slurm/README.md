# SLURM scripts (private, not for the public release)

Reproducibility check: rerun the four model families on the new release
scripts and compare to the production numbers.

## Layout

```
slurm/
├── carbon-3B/          → hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 · HF · 8 GPUs · --add_dna_tag
├── generator-v2-3b/    → GenerTeam/GENERator-v2-eukaryote-3b-base   · HF · 8 GPUs · no tag
├── evo2-1b/            → evo2_1b_base                                · evo2 · 1 GPU
└── evo2-7b/            → evo2_7b_base                                · evo2 · 1 GPU (2 GPUs for ClinVar 24 kb)
```

Each folder has the same 5 scripts:

| Script | Eval | Defaults |
|---|---|---|
| `sequence_recovery.sbatch` | Sequence recovery, eukaryote split | gen_len_bp 30 |
| `vep_brca.sbatch` | BRCA1 then BRCA2 (`vep_eval.py`) | bf16 |
| `vep_traitgym.sbatch` | TraitGym Mendelian (`vep_eval.py`) | `--rev_comp_avg` |
| `clinvar.sbatch` | ClinVar (`clinvar_vep_eval.py`) | `--context_length 24000` |
| `perturbation_tasks.sbatch` | TATA + synonymous codons | back-to-back |

Results go to `<model>/results/<eval>/`. Logs to `/fsx/loubna/logs/`.

## Launch a single model

```bash
cd /fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/slurm/carbon-3B
for f in *.sbatch; do sbatch $f; done
```

## Launch everything

```bash
cd /fsx/loubna/projects_v2/carbon/loubna-workspace/carbon-release/slurm
for m in carbon-3B generator-v2-3b evo2-1b evo2-7b; do
  ( cd $m && for f in *.sbatch; do sbatch $f; done )
done
```

## Check progress

```bash
squeue -u $USER
```

Jobs are named `<model>-<eval>` (e.g. `c3b-brca`, `evo2-7b-clinvar`) so it's
easy to see at a glance which ones are running.
