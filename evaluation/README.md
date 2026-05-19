# Carbon evaluation

Six **zero-shot** DNA evals: sequence recovery, BRCA2, TraitGym Mendelian,
ClinVar, nucleotide triplet-expansion, synonymous codon substitution, and Genome-NIAH
long-context retrieval. Every eval supports three model families through
one flag, so the same script runs on Carbon, GENERator, or Evo2.

## Contents

1. [Sequence recovery](#1-sequence-recovery) — next-30-bp generation, exact-base accuracy
2. [BRCA2, TraitGym Mendelian VEP](#2-brca2-traitgym-mendelian-vep) — centered 8 kb window, full-LL delta
3. [ClinVar VEP](#3-clinvar-vep) — right-end / next-token scoring (GENERator recipe)
4. [Sequence-level perturbation tasks](#4-sequence-level-perturbation-tasks) — nucleotide triplet-expansion + synonymous codon substitution, new tasks we built
5. [Genome-NIAH long-context retrieval](#5-genome-niah-long-context-retrieval) — long-context needle-in-a-haystack for DNA (4 tasks × 6 context lengths up to 786 kbp)

## Scripts

| Script | Eval | Metric |
|---|---|---|
| [`sequence_recovery.py`](sequence_recovery.py) | Generate the next 30 bp of a DNA context and compare to the held-out continuation | Per-base accuracy |
| [`vep_eval.py`](vep_eval.py) | BRCA2 / TraitGym VEP — **centered 8 kb window**, full-LL delta scoring | AUROC, AUPRC, Spearman ρ |
| [`clinvar_vep_eval.py`](clinvar_vep_eval.py) | ClinVar VEP — **right-end / next-token** scoring | AUROC, AUPRC |
| [`perturbation_tasks.py`](perturbation_tasks.py) | nucleotide triplet-expansion and synonymous codon substitution (one script, `--task`) | Pairwise discrimination accuracy `mean(LL(real) > LL(perturbed))` |
| [`genome_niah_eval.py`](genome_niah_eval.py) | Long-context retrieval: insert (key, value) in a real-genome haystack, greedy-decode the value | `gen_exact_match`, `gen_base_accuracy`, `ll_correct` |

> Note: prep scripts for rebuilding the BRCA2 / TraitGym parquets from
> primary sources live in [`data_prep/`](data_prep). Not needed for normal
> eval runs — every command below defaults to the prebuilt Hub parquets.

Run the commands below from the repository root. `uv run --group evaluation`
uses the root project environment plus evaluation-only dependencies.

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
  `evo2_1b_base`, `evo2_7b`, `evo2_40b`) as `--model`.

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
| Carbon hybrid (`Carbon-3B`, `carbon-8B-hybrid-loss-1T-v1`) | `--add_dna_tag` |
| Carbon pure-DNA (`carbon-3B-pure-dna-*`) | `--add_bos` (sequence-recovery only; uses `<s>`) |
| GENERator (`GenerTeam/GENERator-*`) | _(no flag — raw DNA)_ |
| Evo2 (`evo2_1b_base`, `evo2_7b`, ...) | `--backend evo2` |

## 1. Sequence recovery

Training-free generative eval from the GENERator paper. Given a DNA context,
generate the next 30 bp and score per-base accuracy against the held-out
continuation. Three splits: `eukaryote`, `bacteria`, `others`.

Dataset: [`GenerTeam/sequence-recovery`](https://huggingface.co/datasets/GenerTeam/sequence-recovery)

```bash
# Carbon 3B hybrid (flagship)
uv run --group evaluation python evaluation/sequence_recovery.py \
    --model HuggingFaceBio/Carbon-3B \
    --data_type eukaryote --add_dna_tag --bf16

# GENERator
uv run --group evaluation python evaluation/sequence_recovery.py \
    --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
    --data_type eukaryote --bf16

# Evo2 7B (1 GPU)
uv run --group evaluation python evaluation/sequence_recovery.py \
    --model evo2_7b --backend evo2 \
    --data_type eukaryote --gen_len_bp 30 --bf16
```

### Long-rollout Sequence Recovery sweeps

For base-pair rollouts across Carbon/Evo2 backends, use the SR sweep wrapper.
When `USE_EVO2=true`, the launcher maps each HF-style `GEN_LEN` point to
`GEN_LEN * BP_PER_TOKEN` generated base pairs unless `GEN_LEN_BP` is explicitly
set.

```sh
MODEL=evo2_7b \
MODEL_NAME=Evo2-7B \
USE_EVO2=true \
BATCH_SIZE=1 \
GEN_LENS="5 10 20 40 80 160 320 640 1280 2560" \
ACCURACY_MODE=prediction_length \
BASE_OUTPUT_DIR=./eval_results/sequence_recovery_long_rollouts_pow2 \
evaluation/submit_sequence_recovery_gen_len_sweep.sh
```

Plot a completed sweep with Carbon and Evo2 curves:

```sh
uv run --group evaluation \
  --with matplotlib \
  --with numpy \
  python evaluation/scripts/plot_sequence_recovery_sweep.py \
  --base_dir ./eval_results/sequence_recovery_long_rollouts_pow2 \
  --data_type eukaryote \
  --model "3B hybrid (HF)=Carbon-3B-600B-dna-generv2-fp32-lmhead" \
  --model "8B hybrid (HF)=Carbon-8B-600B-dna-fp32-lmhead" \
  --model "Evo2 7B=Evo2-7B" \
  --out scratch/plots/sequence_recovery_sweep_overall.png \
  --type_panels scratch/plots/sequence_recovery_sweep_types.png
```

## 2. BRCA2, TraitGym Mendelian VEP

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
- **BRCA2** — headline is **global AUROC + AUPRC** computed with
  `sklearn.metrics.auc(recall, precision)`, following the Evo2 §4.3.15
  recipe used by the original BRCA reproductions.

For single-locus datasets (BRCA2) the by-chrom number collapses to global,
so only global is reported. All four numbers (global AUROC/AUPRC,
by-chrom-weighted AUROC/AUPRC) are saved in the summary JSON regardless of
dataset.

Single eval script ([`vep_eval.py`](vep_eval.py)) handles both datasets —
they share the same parquet schema (`chrom, pos, ref, alt, score, class,
ref_seq, var_seq`). Build a dataset with the matching `prep_*.py` once,
then point `--data_path` at it.

| Dataset | Source | Hub | n | Window |
|---|---|---|---|---|
| **BRCA2** | [Huang et al. 2025, Nature](https://www.nature.com/articles/s41586-024-08388-8) (DBD ACMG classes) | `HuggingFaceBio/brca2-vep` | 6,836 SNVs (1,156 LOF + 5,680 FUNC/INT) | chr13 hg19 |
| **TraitGym Mendelian** | [Benegas, Eraslan & Song 2025](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v1) — **non-coding regulatory variants** for 113 Mendelian diseases | `HuggingFaceBio/traitgym` | 3,380 variants (338 causal + 3,042 matched controls) | hg38, all chromosomes |

```bash
# Carbon 3B hybrid · BRCA2 (8 GPUs)
uv run --group evaluation python evaluation/vep_eval.py \
    --model HuggingFaceBio/Carbon-3B \
    --data_path hf://datasets/HuggingFaceBio/brca2-vep/brca2_vep.parquet \
    --add_dna_tag --bf16 \
    --output_dir ./results/brca2_vep

# TraitGym Mendelian — pass --rev_comp_avg, variants can sit on either strand
uv run --group evaluation python evaluation/vep_eval.py \
    --model HuggingFaceBio/Carbon-3B \
    --data_path hf://datasets/HuggingFaceBio/traitgym/mendelian_traits_vep.parquet \
    --add_dna_tag --bf16 --rev_comp_avg \
    --output_dir ./results/traitgym_mendelian

# Evo2 7B (1 GPU)
uv run --group evaluation python evaluation/vep_eval.py \
    --model evo2_7b --backend evo2 \
    --data_path hf://datasets/HuggingFaceBio/brca2-vep/brca2_vep.parquet \
    --bf16 --output_dir ./results/brca2_vep_evo2
```

## 3. ClinVar VEP

ClinVar uses the [GENERator VEP recipe](https://github.com/GenerTeam/GENERator/blob/main/src/tasks/downstream/variant_effect_prediction.py):
for each variant we build a long **left-context window** with the variant
position at the right end (default 24 kb), do one forward pass, take the
softmax of the last token, then marginalise over all DNA tokens whose first
base is the ref / alt nucleotide to get `P(ref)` and `P(alt)`. Score is
`log(P(ref) / P(alt))` — higher when the alt is more surprising. AUROC of
score vs `label == 1`.

Dataset: [`HuggingFaceBio/clinvar-vep-final`](https://huggingface.co/datasets/HuggingFaceBio/clinvar-vep-final).
This is the GenerTeam ClinVar release (mostly coding) augmented with a
Carbon-curated noncoding split, since GenerTeam's release is ~99 % coding and
we wanted balanced coverage. Schema: `chrom, pos, ref, alt, label, region,
variant_type, …`. If `region` or `variant_type` is present, the script prints
per-breakdown AUROC / AUPRC automatically.

```bash
# Carbon 3B hybrid (flagship, 8 GPUs, 24 kbp context)
uv run --group evaluation python evaluation/clinvar_vep_eval.py \
    --model HuggingFaceBio/Carbon-3B \
    --add_dna_tag --bf16 --context_length 24000 \
    --output_dir ./results/clinvar

# Evo2 7B
uv run --group evaluation python evaluation/clinvar_vep_eval.py \
    --model evo2_7b --backend evo2 --bf16 \
    --context_length 24000 --output_dir ./results/clinvar_evo2
```

## 4. Sequence-level perturbation tasks

nucleotide triplet-expansion and synonymous codon substitution — **new tasks we built
for Carbon**, not ported from prior work. Each applies a structural
perturbation to a real biological sequence and asks whether the model assigns higher log-likelihood to the unperturbed
version. Distinct from VEP, which probes single-nucleotide changes.

- **nucleotide triplet-expansion** — A 30 bp codon-aligned region beginning 60 bp downstream of the first complete codon of the CDS exon is replaced with 10 consecutive CAG triplets (CAGCAGCAGCAGCAGCAGCAGCAGCAGCAG), mimicking the pathological trinucleotide repeat expansions underlying polyglutamine disorders (Huntington's disease, SCAs, DRPLA).
- **Synonymous codon substitution** — Codons within a real CDS are replaced with the highest-frequency synonym for the target species, while the upstream and downstream flanking sequence is left unchanged. Amino acid identity is preserved by construction. The model should prefer the natural codon usage over the artificially optimised variant.

Dataset: [`HuggingFaceBio/carbon_tasks`](https://huggingface.co/datasets/HuggingFaceBio/carbon_tasks)
(columns `original_sequence` = real, `sequence` = perturbed)

```bash
# Carbon 3B hybrid · triplet-expansion
uv run --group evaluation python evaluation/perturbation_tasks.py \
    --task motif_human \
    --model HuggingFaceBio/Carbon-3B \
    --bf16

# Carbon 3B hybrid · human synonymous codons
uv run --group evaluation python evaluation/perturbation_tasks.py \
    --task syn_human \
    --model HuggingFaceBio/Carbon-3B \
    --bf16

# Carbon 3B hybrid · mouse synonymous codons
uv run --group evaluation python evaluation/perturbation_tasks.py \
    --task syn_mouse \
    --model HuggingFaceBio/Carbon-3B \
    --bf16

# Evo2 7B
uv run --group evaluation python evaluation/perturbation_tasks.py \
    --task motif_human \
    --model evo2_7b --backend evo2 --bf16
```

## 5. Genome-NIAH long-context retrieval

A long-context needle-in-a-haystack for DNA. We insert a 24 bp (key, value) pair
at a controlled depth in a real-genome haystack (OpenGenome2), then ask the
model to greedy-decode the value when prompted with `haystack + key`.

Four tasks of increasing difficulty:

| `--task` | Distractors | Key identity to distractor |
|---|---|---|
| `niah` | 0 | — |
| `neardup_d4` | 8 | 83% (Δ=4 bp) |
| `neardup_d2` | 8 | 92% (Δ=2 bp) |
| `neardup_d1` | 8 | 96% (Δ=1 bp) — discriminator |

Six context lengths (`--ctx`, given in **6-mer tokens** — Carbon's native tokenization unit, where 1 token = 6 bp). The corresponding bp counts below are derived (= ctx × 6):

| `--ctx` (6-mer tokens) | bp | haystack mode |
|---|---|---|
| 4096   | 24 kbp  | contiguous |
| 8192   | 49 kbp  | contiguous |
| 16384  | 98 kbp  | contiguous |
| 32768  | 197 kbp | stitched (ACGT runs concatenated) |
| 65536  | 393 kbp | stitched |
| 131072 | 786 kbp | stitched |

Dataset is `HuggingFaceBio/genome-niah` (auto-loaded). Each (task, ctx) cell has
n=500 rows by default. Pass `--max_samples N` for quick smoke tests.

**Metrics.** **The default / headline metric is `gen_exact_match`** — that's the number we report in tables, the one users should compare against. The other two are auxiliary diagnostics, written to the same JSON for convenience but not the primary score.

| metric | role | what it measures |
|---|---|---|
| **`gen_exact_match`** | **default / headline** | binary, 1 iff the entire 24 bp greedy-decoded value byte-matches the label |
| `gen_base_accuracy` | diagnostic | continuous, fraction of the 24 bp matching (chance for ACGT = 0.25) — distinguishes "close but slipped" from "random" |
| `ll_correct` | sanity check | binary, `LL(positive) > LL(negative)`. **Inflated** for this task — negative has 24 bp wrong so likelihood gap is easy. We see `ll_correct ≥ 0.95` even when `gen_em ≤ 0.10`. Do **not** report as headline. |

Each model is evaluated up to its native context (with optional yarn4× extension for Carbon, documented separately on the yarn variant repos).

```bash
# Carbon-3B-lc32k at native 32 k (4 k / 8 k / 16 k / 32 k)
uv run --group evaluation python evaluation/genome_niah_eval.py \
    --model HuggingFaceBio/carbon-3B-longctx-32k-rope5M \
    --task niah --ctx 32768 --add_dna_tag --bf16

# GENERator-v2 3B at native 16 k (4 k / 8 k / 16 k)
uv run --group evaluation python evaluation/genome_niah_eval.py \
    --model GenerTeam/GENERator-v2-eukaryote-3b-base \
    --task niah --ctx 16384 --bf16

# Evo2-7B at 32 k, single 8-GPU node
uv run --group evaluation python evaluation/genome_niah_eval.py \
    --model evo2_7b --backend evo2 \
    --task niah --ctx 32768 --prefill_chars 4096
```

Sample wallclock on **one 8-GPU H100 node**:

| backend | model | unit | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|---|---|
| hf | Carbon 3B (lc32k) | full sweep, n=500 rows | ~5 min | ~6 min | ~12 min | ~30 min | ~1.5 h |
| hf | GENERator-v2 3B | full sweep, n=500 rows | ~5 min | ~6 min | ~12 min | — | — |
| evo2 | Evo2-7B | per row | ~10 min | ~20 min | ~50 min | ~1.5 h | ~4 h |

**Note on Evo2 numbers.** Evo2's reference inference (`vortex`) requires multi-GPU
model-parallel inference for ctx ≥ ~32 k — a single-GPU run OOMs. We use vortex's
default 8-GPU pipeline-parallel layer-splitting, and
add row-level sharding (`--shard_idx --n_shards`) at the eval script so a `(task, ctx)`
cell can be split across multiple SLURM jobs. Full n=500 on a single
node is impractical at ctx ≥ 16 k — shard across nodes and/or use `--max_samples`.

For Evo2 at long ctx, subset (`--max_samples N`) and shard (`--shard_idx --n_shards`).
We used n=100 at 16-32 k and n=20 at 64 k (smaller n = noisier estimates).
Aggregate across shards by concatenating per-shard parquets and taking a sample-weighted mean of `gen_exact_match`.

```bash
# From the repository root: Evo2 at 32k, n=100 split across 6 shards (1 node each)
for SHARD in 0 1 2 3 4 5; do
  POOL=100 SHARD=$SHARD NSHARDS=6 TASK=niah CTX=32768 \
    sbatch evaluation/slurm/evo2-7b/genome_niah.sbatch
done
```
## Environment

Install the root project environment with uv:

```bash
# Core dependencies only
uv sync

# Core plus evaluation dependencies
uv sync --group evaluation
```

**Evo2 backend** — the official [evo2](https://github.com/ArcInstitute/evo2)
install needs CUDA 12.1+, Python 3.11/3.12, and matching `flash-attn` +
`transformer-engine` builds for your PyTorch. Follow their install guide
first, then layer any additional pins on top with uv:

```bash
uv pip install -r requirements-evo2.txt
```

We use a separate venv for Evo2 in practice because the `flash-attn` and
`transformer-engine` wheels are tightly tied to a specific torch + CUDA
build.
