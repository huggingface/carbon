# Carbon evaluation

Six **zero-shot** DNA evals: sequence recovery, BRCA1, BRCA2, TraitGym
Mendelian, ClinVar, TATA perturbation, and synonymous codon substitution.
Every eval supports three model families through one flag, so the same
script runs on Carbon, GENERator, or Evo2.

## Contents

1. [Sequence recovery](#1-sequence-recovery) — next-30-bp generation, exact-base accuracy
2. [BRCA1, BRCA2, TraitGym Mendelian VEP](#2-brca1-brca2-traitgym-mendelian-vep) — centered 8 kb window, full-LL delta
3. [ClinVar VEP](#3-clinvar-vep) — right-end / next-token scoring (GENERator recipe)
4. [Sequence-level perturbation tasks](#4-sequence-level-perturbation-tasks) — TATA + synonymous codons, new tasks we built

## Scripts

| Script | Eval | Metric |
|---|---|---|
| [`sequence_recovery.py`](sequence_recovery.py) | Generate the next 30 bp of a DNA context and compare to the held-out continuation | Per-base accuracy |
| [`vep_eval.py`](vep_eval.py) | BRCA1 / BRCA2 / TraitGym VEP — **centered 8 kb window**, full-LL delta scoring | AUROC, AUPRC, Spearman ρ |
| [`clinvar_vep_eval.py`](clinvar_vep_eval.py) | ClinVar VEP — **right-end / next-token** scoring | AUROC, AUPRC |
| [`perturbation_tasks.py`](perturbation_tasks.py) | TATA perturbation and synonymous-codon substitution (one script, `--task`) | Pairwise discrimination accuracy `mean(LL(real) > LL(perturbed))` |

> Note: prep scripts for rebuilding the BRCA1 / BRCA2 / TraitGym parquets
> from primary sources live in [`data_prep/`](data_prep). Not needed for
> normal eval runs — every command below defaults to the prebuilt Hub parquets.

The two VEP scripts use different scoring recipes. BRCA / TraitGym use
**centered + full-LL delta** (the Evo2 / TraitGym convention recent papers
compare against). For ClinVar we use the GENERator setup — **right-end /
next-token** scoring — see [GENERator/variant_effect_prediction.py](https://github.com/GenerTeam/GENERator/blob/main/src/tasks/downstream/variant_effect_prediction.py).

## Backends

Every eval takes `--backend {hf, evo2}`:

- **`hf`** (default) — `transformers.AutoModelForCausalLM`. Works for Carbon
  (3B / 8B), GENERator, and any HF causal LM. Multi-GPU shards
  automatically.
- **`evo2`** — the official [`evo2`](https://github.com/ArcInstitute/evo2)
  inference library. Required for Arc Institute's Evo2 checkpoints (their
  weights aren't AutoModel-compatible). Pass the Evo2 model name (e.g.
  `evo2_1b_base`, `evo2_7b_base`, `evo2_40b_base`) as `--model`.

## DNA tags — why and when to use them

Carbon's **hybrid tokenizer** lets one model handle both English and DNA. To
do that it has a special `<dna>` tag that flips tokenization from BPE
(English) to 6-mer (DNA) mid-sequence. Anything between `<dna>` and the next
text is read as DNA bases, six at a time.

For zero-shot evals you have to opt in to DNA mode with the `--add_dna_tag`
flag. The script then prepends `<dna>` to every input so the hybrid model
tokenizes it as DNA. **You only pass this flag for Carbon hybrid models**.
For GENERator and Evo2 (both pure-DNA), leave it off — they tokenize DNA
natively.

| Model | Flag |
|---|---|
| Carbon hybrid (`carbon-3B-hybrid-loss-1T-mix2-v1`, `carbon-8B-hybrid-loss-1T-v1`) | `--add_dna_tag` |
| Carbon pure-DNA (`carbon-3B-pure-dna-*`) | `--add_bos` (sequence-recovery only; uses `<s>`) |
| GENERator (`GenerTeam/GENERator-*`) | _(no flag — raw DNA)_ |
| Evo2 (`evo2_1b_base`, `evo2_7b_base`, ...) | `--backend evo2` |

## 1. Sequence recovery

Training-free generative eval from the GENERator paper. Given a DNA context,
generate the next 30 bp and score per-base accuracy against the held-out
continuation. Three splits: `eukaryote`, `bacteria`, `others`.

Dataset: [`GenerTeam/sequence-recovery`](https://huggingface.co/datasets/GenerTeam/sequence-recovery)

```bash
# Carbon 3B hybrid (flagship)
python sequence_recovery.py \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --data_type eukaryote --add_dna_tag --bf16

# GENERator
python sequence_recovery.py \
    --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
    --data_type eukaryote --bf16

# Evo2 7B (1 GPU)
python sequence_recovery.py \
    --model evo2_7b_base --backend evo2 \
    --data_type eukaryote --gen_len_bp 30 --bf16
```

## 2. BRCA1, BRCA2, TraitGym Mendelian VEP

We use the [Evo2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1)
and [TraitGym](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v1)
evaluation recipe: for each variant, score the full log-likelihood of the
**8,192 bp window centered on the variant** for both the reference and
variant sequences. Score the variant by `delta = LL(var) - LL(ref)` and
report AUROC of `-delta` vs the binary label (LOF), plus Spearman ρ vs the
continuous functional score where available.

We follow each benchmark's own metric convention:

- **TraitGym** — headline is **by-chromosome-weighted AUPRC**, following the
  [TraitGym](https://github.com/songlab-cal/TraitGym/tree/main) leaderboard.
  Per-chromosome AUPRC uses sklearn's `average_precision_score`, weighted
  by chromosome size.
- **BRCA1 / BRCA2** — headline is **global AUROC + AUPRC** computed with
  `sklearn.metrics.auc(recall, precision)`, following the Evo2 §4.3.15
  recipe used by the original BRCA reproductions.

For single-locus datasets (BRCA1, BRCA2) the by-chrom number collapses to
global, so only global is reported. All four numbers (global AUROC/AUPRC,
by-chrom-weighted AUROC/AUPRC) are saved in the summary JSON regardless of
dataset.

Single eval script ([`vep_eval.py`](vep_eval.py)) handles all three datasets
— they share the same parquet schema (`chrom, pos, ref, alt, score, class,
ref_seq, var_seq`). Build a dataset with the matching `prep_*.py` once, then
point `--data_path` at it.

| Dataset | Source | Hub | n | Window |
|---|---|---|---|---|
| **BRCA1** | [Findlay et al. 2018, Nature](https://www.nature.com/articles/s41586-018-0461-z) (SGE on BRCA1) | `hf-carbon/brca1-vep` | 3,893 SNVs (823 LOF + 3,070 FUNC/INT) | chr17 hg19 |
| **BRCA2** | [Huang et al. 2025, Nature](https://www.nature.com/articles/s41586-024-08388-8) (DBD ACMG classes) | `hf-carbon/brca2-vep` | 6,836 SNVs (1,156 LOF + 5,680 FUNC/INT) | chr13 hg19 |
| **TraitGym Mendelian** | [Benegas, Eraslan & Song 2025](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v1) — **non-coding regulatory variants** for 113 Mendelian diseases | `hf-carbon/traitgym` | 3,380 variants (338 causal + 3,042 matched controls) | hg38, all chromosomes |

```bash
# Carbon 3B hybrid · BRCA1 (8 GPUs)
python vep_eval.py \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --data_path hf://datasets/hf-carbon/brca1-vep/brca1_vep.parquet \
    --add_dna_tag --bf16 \
    --output_dir ./results/brca1_vep

# Same script, BRCA2 — just swap the parquet
python vep_eval.py \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --data_path hf://datasets/hf-carbon/brca2-vep/brca2_vep.parquet \
    --add_dna_tag --bf16 \
    --output_dir ./results/brca2_vep

# TraitGym Mendelian — pass --rev_comp_avg, variants can sit on either strand
python vep_eval.py \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --data_path hf://datasets/hf-carbon/traitgym/mendelian_traits_vep.parquet \
    --add_dna_tag --bf16 --rev_comp_avg \
    --output_dir ./results/traitgym_mendelian

# Evo2 7B (1 GPU)
python vep_eval.py \
    --model evo2_7b_base --backend evo2 \
    --data_path hf://datasets/hf-carbon/brca1-vep/brca1_vep.parquet \
    --bf16 --output_dir ./results/brca1_vep_evo2
```

## 3. ClinVar VEP

ClinVar uses the [GENERator VEP recipe](https://github.com/GenerTeam/GENERator/blob/main/src/tasks/downstream/variant_effect_prediction.py):
for each variant we build a long **left-context window** with the variant
position at the right end (default 24 kb), do one forward pass, take the
softmax of the last token, then marginalise over all DNA tokens whose first
base is the ref / alt nucleotide to get `P(ref)` and `P(alt)`. Score is
`log(P(ref) / P(alt))` — higher when the alt is more surprising. AUROC of
score vs `label == 1`.

Dataset: [`hf-carbon/clinvar-vep-final`](https://huggingface.co/datasets/hf-carbon/clinvar-vep-final).
This is the GenerTeam ClinVar release (mostly coding) augmented with a
Carbon-curated noncoding split, since GenerTeam's release is ~99 % coding and
we wanted balanced coverage. Schema: `chrom, pos, ref, alt, label, region,
variant_type, …`. If `region` or `variant_type` is present, the script prints
per-breakdown AUROC / AUPRC automatically.

```bash
# Carbon 3B hybrid (flagship, 8 GPUs, 24 kbp context)
python clinvar_vep_eval.py \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --add_dna_tag --bf16 --context_length 24000 \
    --output_dir ./results/clinvar

# Evo2 7B
python clinvar_vep_eval.py \
    --model evo2_7b_base --backend evo2 --bf16 \
    --context_length 24000 --output_dir ./results/clinvar_evo2
```

## 4. Sequence-level perturbation tasks

TATA perturbation and synonymous codon substitution — **new tasks we built
for Carbon**, not ported from prior work. Each applies a structural
perturbation to a real biological sequence (motif disruption or codon swap)
and asks whether the model assigns higher log-likelihood to the unperturbed
version. Distinct from VEP, which probes single-nucleotide changes.

- **TATA perturbation** — disrupt the TATA-box motif inside a promoter with
  random substitutions. A model that has internalized eukaryotic promoter
  architecture should prefer the intact promoter.
- **Synonymous codon substitution** — replace codons in a CDS with synonyms
  encoding the same amino acid. A model that has learned codon-usage bias
  should prefer the native codon usage over the synonymous variant.

Dataset: [`hf-carbon/carbon_tasks`](https://huggingface.co/datasets/hf-carbon/carbon_tasks)
(columns `original_sequence` = real, `sequence` = perturbed)

```bash
# Carbon 3B hybrid · TATA
python perturbation_tasks.py \
    --task tata_perturbation \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --add_dna_tag --bf16

# Carbon 3B hybrid · synonymous codons
python perturbation_tasks.py \
    --task synonymous_codon_substitution \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --add_dna_tag --bf16

# Evo2 7B
python perturbation_tasks.py \
    --task tata_perturbation \
    --model evo2_7b_base --backend evo2 --bf16
```

## Environment

Two separate Python environments are needed because the Evo2 library has
incompatible CUDA/PyTorch pins:

- **HF backend** — any recent `transformers` + `torch` install works.
- **Evo2 backend** — follow the [evo2 install guide](https://github.com/ArcInstitute/evo2).
  Sequence-recovery on Evo2 needs FlashAttention; the other evals do not.

Common deps (HF backend): `pip install transformers torch pandas scikit-learn tqdm datasets huggingface_hub`.
