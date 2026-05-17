# SLURM scripts

Reference SLURM scripts the Carbon team uses to run the eval suite on the
**internal HF cluster**. Cluster settings such as partition `hopper-prod`,
GPU counts, and walltimes are HF-specific — adapt them for your setup if needed.

Submit these scripts from the repository root. Log and result paths are relative
to that directory. Each script sources `~/.bashrc` before enabling strict mode,
so put local shell setup such as `uv` there instead of adding hard-coded PATH or
library exports to the scripts.

One folder per model family, each with the same core scripts. Carbon and Evo2
folders also include a token-level perturbation launcher.

```
carbon-3B/          → HuggingFaceBio/Carbon-3B   (also lc32k variants)
generator-v2-3b/    → GenerTeam/GENERator-v2-eukaryote-3b-base
evo2-1b/            → evo2_1b_base
evo2-7b/            → evo2_7b
evo2-20b/           → evo2_20b
evo2-40b/           → evo2_40b
```

| Script | Eval |
|---|---|
| `sequence_recovery.sbatch` | Sequence recovery, eukaryote split |
| `vep_brca2.sbatch` | BRCA2 |
| `vep_traitgym.sbatch` | TraitGym Mendelian (with `--rev_comp_avg`) |
| `clinvar.sbatch` | ClinVar coding + non_coding (default 24 kb, `CONTEXT_LENGTH=48000` for 48 kb) |
| `perturbation_tasks.sbatch` | Sequence-level perturbation tasks: motif, synonymous-codon, and promoter reverse-complement |
| `perturbation_tasks_token.sbatch` | Token-level variant of the sequence-level perturbation tasks |
| `genome_niah.sbatch` | Long-context retrieval. Defaults to `TASK=niah CTX=32768`. Override via env: see script header for a sweep example. |

The Evo2 20B and 40B scripts request 8 GPUs by default. The larger checkpoints
use vortex model-parallel loading in the eval scripts, and the 40B ClinVar path
requires a single 8-GPU model-parallel instance rather than one model per GPU.

Launch one model:
```bash
for f in evaluation/slurm/carbon-3B/*.sbatch; do sbatch "$f"; done
```

Sweep Genome-NIAH across all 4 tasks × 4 short contexts (Carbon, ~3h total):
```bash
for TASK in niah neardup_d4 neardup_d2 neardup_d1; do
  for CTX in 4096 8192 16384 32768; do
    TASK=$TASK CTX=$CTX sbatch evaluation/slurm/carbon-3B/genome_niah.sbatch
  done
done
```
