# SeqQA Data Pipeline

Programmatic generator for synthetic sequence-reasoning training data, inspired by the SeqQA tasks in [LAB-Bench](https://arxiv.org/abs/2407.10362) but redesigned for training at scale rather than benchmarking.

## Overview

`generate_data.py` produces sequence-reasoning questions that test DNA/RNA sequence comprehension: GC content, restriction digests, ORF analysis, translation efficiency, primer design, and cloning workflows. Each example contains a question and an ideal answer. The output is JSONL, optionally pushed directly to the Hub.

```sh
# Generate all subtasks (1000 examples each by default)
uv run --directory data python seqqa/generate_data.py \
    --output ../scratch/seqqa_training.jsonl

# Use a normal distribution over sequence lengths (centred on midpoint)
uv run --directory data python seqqa/generate_data.py \
    --output ../scratch/seqqa_training.jsonl --length-distribution normal

# Push to Hub
uv run --directory data python seqqa/generate_data.py \
    --push-to-hub --hub-repo hf-carbon/seqqa-synth --hub-config v1
```

## Subtasks

The pipeline generates **15 subtasks** across four categories:

### Sequence Properties
| Subtask | Question | Answer |
|---|---|---|
| `seq_gc_pct` | Estimate the GC% of a DNA fragment | Rounded integer |

### Restriction Enzymes
| Subtask | Question | Answer |
|---|---|---|
| `restriction_fragment_count` | How many bands after digesting with enzyme(s)? | Integer |
| `restriction_fragment_lengths` | List fragment lengths from a complete digest | Comma-separated integers |

### Open Reading Frames
| Subtask | Question | Answer |
|---|---|---|
| `orf_aa_position` | Which amino acid sits at position N of the longest ORF? | Three-letter AA name |
| `orf_aa_sequence` | Translate the longest ORF | Full AA sequence |
| `orf_count_over_threshold` | How many ORFs encode proteins longer than N amino acids? | Integer |

### Translation
| Subtask | Question | Answer |
|---|---|---|
| `translation_upstream_aug_count` | Count upstream AUG codons in a 5' leader | Integer |
| `translation_efficiency` | Which transcript has the strongest Kozak context? | Transcript label (e.g. Transcript A) |

### PCR & Primer Design
| Subtask | Question | Answer |
|---|---|---|
| `amplicon_target_primers` | Which primer pair amplifies a given target region? | Fwd, Rev primer pair |
| `amplicon_length_primers` | Which primer pair produces an N bp amplicon? | Fwd, Rev primer pair |
| `primer_pair_amplicon_length` | What amplicon length from these primers on this template? | Integer (bp) |
| `amplicon_sequence` | What exact DNA sequence is amplified by this primer pair? | Full DNA amplicon sequence |

### Cloning Workflows
| Subtask | Question | Answer |
|---|---|---|
| `restriction_clone_primer_design` | Design RE-cloning primers (pad + site + binding) for a vector/insert pair | Fwd, Rev primer pair |
| `vector_insert_compatibility` | Which enzyme pair opens the vector while leaving the insert intact? | Enzyme pair |
| `gibson_primer_design` | Design Gibson assembly primers (homology + binding) for a vector/insert pair | Fwd, Rev primer pair |

## Sequence Sources

The pipeline draws from **real biological sequences**, not synthetic random DNA:

- **AddGene plasmid sequences** (`carbon-internal/AddGene`, `sequences.jsonl`): random windows from cached plasmid sequences. Most subtasks use 50-5000 bp windows; `amplicon_sequence` and `orf_aa_sequence` use longer natural windows so the ideal answer can span many kilobases or encode much longer proteins. Lengths can be sampled uniformly or from a truncated normal distribution via the `--length-distribution` flag.
- **AddGene vectors** (`carbon-internal/AddGene`, `plasmids.jsonl`): real lab vectors with known single-cut restriction sites. Used as recipient vectors in cloning subtasks. Vectors with <2 single-cut enzymes are filtered out.
- **Insert catalog**: ORFs extracted from AddGene plasmid windows (240-1800 bp coding sequences) serve as insert sequences for cloning tasks.
- **Procedural sequences**: synthetically generated 5' leaders with controlled upstream AUG counts for translation subtasks, and full-length RNA sequences with Kozak context variants for translation efficiency comparisons.

## Comparison with LAB-Bench SeqQA

### What LAB-Bench does

[LAB-Bench](https://arxiv.org/abs/2407.10362) defines **15 SeqQA subtasks** (750 questions total) as a benchmark for evaluating LLM sequence reasoning. Questions are multiple-choice with one correct answer and three distractors. The tasks span:

- **8 PCR subtasks**: primer selection given gene name or sequence + enzyme pair, Gibson assembly primers (HindIII/SmaI linearization), amplicon length prediction
- **2 Restriction enzyme subtasks**: fragment count and fragment lengths after digestion
- **4 ORF subtasks**: translation efficiency, AA sequence of longest ORF, ORF count above threshold, AA at specific position
- **1 Property subtask**: GC content percentage

The public benchmark families use fixed named-gene cloning contexts: `futurehouse/lab-bench` SeqQA repeatedly uses **pUC19**, named **E. coli** genes, and specific enzymes such as **HindIII** and **SmaI** for Gibson linearization, while `EdisonScientific/labbench2` `seqqa2` adds named **M. genitalium** genes. Human evaluators were given APE (A Plasmid Editor) software; models were evaluated without tools.

### Similarities

- **Programmatic generation**: both pipelines generate questions and ground truth computationally using sequence analysis (restriction digests, ORF finding, primer binding, GC calculation).
- **Shared task families**: GC content, restriction fragment count/lengths, ORF AA position/sequence/count, translation efficiency, primer design, amplicon length, and Gibson assembly all appear in both.
- **Molecular biology scope**: both test practical lab-relevant sequence reasoning rather than abstract pattern matching.

### Key Differences

| Aspect | LAB-Bench SeqQA | This Pipeline |
|---|---|---|
| **Purpose** | Evaluation benchmark (fixed, small) | Training data generation (scalable) |
| **Scale** | 750 questions, 15 subtasks, fixed | Configurable; default 1000/subtask across 15 subtasks, arbitrarily scalable |
| **Sequence sources** | pUC19 + named benchmark genes | Diverse AddGene plasmid sequences + hundreds of AddGene vectors |
| **Organisms/vectors** | Single canonical vector (pUC19) | ~200 different real lab vectors, no single canonical choice |
| **Benchmark contamination** | N/A (is the benchmark) | Partially mitigated: no gene-name lookups avoids named-gene overlap. No active sequence-level or enzyme-level filtering is currently applied |
| **Gibson linearization** | Fixed to HindIII or SmaI | Any single-cut enzyme on the chosen vector |
| **Gene references** | Questions reference genes by name (e.g. "aceE") | No gene-name lookups; all sequences provided inline |
| **PCR subtask naming** | Encodes input/output in name (e.g. `PCR-gene-enzprimers`) | Descriptive names (e.g. `restriction_clone_primer_design`) |
| **Enzyme pool** | Full set including HindIII, SmaI | 20 common cloning enzymes |
| **Translation tasks** | Translation efficiency comparison across sequences | Upstream AUG counting in synthetic 5' leaders; translation efficiency comparison via Kozak context variants |
| **Insert sequences** | E. coli gene CDS | ORFs mined from AddGene plasmids (insert catalog) |
| **Answer format** | Multiple-choice with distractors | Open-ended ideal answers only |
| **Reproducibility** | Fixed dataset | Deterministic via seed; SHA-256-based per-subtask seed derivation |
