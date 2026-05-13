# Carbon

Genomic foundation models from Hugging Face. Carbon is a family of causal
language models trained on **1T tokens of DNA / 6T DNA base pairs** from the
[Carbon Pretraining Corpus](https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus),
a curated mix of DNA & RNA sequences.

This repo contains the eval code for Carbon tasks: sequence recovery, variant
effect prediction, TATA promoter perturbation, and synonymous codon
substitution. We put this together because the zero-shot DNA eval landscape is
currently scattered — useful tasks live in different repos, often buried
alongside evals that need finetuning or that are already saturated, which
makes reproducibility harder.

## Contents

- [Models](#models)
- [Inference](#inference)
- [Pretraining](#pretraining)
- [Evaluation](#evaluation)
- [Finetuning](#finetuning)

## Models

Current Carbon checkpoints are available in the
[Carbon checkpoints collection](https://huggingface.co/collections/HuggingFaceBio/carbon-checkpoints).

| Model | Params | Notes |
|---|---|---|
| [`HuggingFaceBio/Carbon-500M`](https://huggingface.co/HuggingFaceBio/Carbon-500M) | 500M | Draft model for speculative decoding. |
| **[`HuggingFaceBio/Carbon-3B`](https://huggingface.co/HuggingFaceBio/Carbon-3B)** | 3B | **Flagship.** Matches or beats Evo2 7B. |
| [`HuggingFaceBio/Carbon-8B`](https://huggingface.co/HuggingFaceBio/Carbon-8B) | 8B | Larger model for more performance. |

The Carbon checkpoints use a **hybrid tokenizer**: BPE for English text and 6-mer
for DNA, switched by a `<dna>` tag mid-sequence. That's why every inference
or eval snippet below wraps DNA inputs with `<dna>` — see
[evaluation/README.md](evaluation/README.md) for the full DNA-tag explanation.

TODO: add this behavior in tokenizer by default?

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceBio/Carbon-3B"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
                                             torch_dtype="bfloat16").to("cuda")

# DNA generation: wrap the prompt with <dna> so the tokenizer routes to 6-mer mode.
context = "ATGGCCTCGAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG"
prompt = f"<dna>{context}"
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tok.decode(out[0]))
```

For zero-shot variant scoring, just feed the model the full sequence and read
the log-likelihood — see [`evaluation/vep_eval.py`](evaluation/vep_eval.py).

## Pretraining

### Training data

Carbon was trained on **1 T tokens (≈ 6 T DNA base pairs)** drawn from the
[Carbon Pretraining Corpus](https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus) mix of:

- **Eukaryote genes** (animals, plants, fungi, protists) — functional genomic regions, extracted from refSeq from Generator training mix.
- **mRNA transcripts** — processed, spliced mRNA from OpenGenome2.
- **Prokaryote genomes** — long chromosomal chunks from bacteria and archaea
  (GTDB v220 + IMG/PR), included as a smaller fraction (~10 % of the
  training mixture).

The mixture is **eukaryote-heavy by design**. Carbon's target use case is
eukaryote. The
prokaryote share is 10% of the pretraining mixture, so the model can be continually pretrained on prokaryote species.

### Pretraining code

Carbon was trained with our Megatron-LM fork:
[**huggingface/Megatron-LM-Carbon**](https://github.com/huggingface/Megatron-LM-Carbon).
The fork adds:

- Hybrid loss: the loss for bridging coarse 6-mer tokenization and single-nucleotide resolution.
- Carbon training scripts

## Evaluation

This repo ships a **suite of seven zero-shot DNA evaluations** with
reproducible code. The benchmark datasets are available in this [collection](https://huggingface.co/collections/HuggingFaceBio/dna-benchmarks).

The suite covers four modes of zero-shot evaluation:

- **Variant effect prediction**, with three established benchmarks spanning
  both coding (BRCA2) and non-coding regulatory variants (TraitGym
  Mendelian), plus ClinVar for broad pathogenic-vs-benign coverage.
- **A generative task** — sequence recovery, ported from the GENERator paper.
- **Two perturbation tasks** we built — TATA-box perturbation and
  synonymous-codon substitution — to probe regulatory-motif awareness and
  codon-usage structure.
- **Long-context retrieval**  we built — Genome-NIAH, a needle-in-a-haystack eval
  adapted to DNA (four tasks × six context lengths up to 786 kbp).

All eval scripts live in [`evaluation/`](evaluation). Each one runs on Carbon,
GENERator, or Evo2 via a single backend flag, so numbers are directly
comparable across model families.

| Benchmark | What it measures | Script |
|---|---|---|
| **Sequence recovery** | Given a DNA context, generate the next 30 bp; score per-base accuracy against the held-out continuation. Training-free generative eval from the GENERator paper. | [`sequence_recovery.py`](evaluation/sequence_recovery.py) |
| **TATA perturbation** | Disrupt the TATA-box motif in a promoter; the model should assign higher likelihood to the intact promoter. Probes regulatory-motif awareness. | [`perturbation_tasks.py`](evaluation/perturbation_tasks.py) `--task tata_perturbation` |
| **Synonymous codon substitution** | Replace codons with synonyms encoding the same amino acid; the model should prefer native codon usage. Probes coding-region structure. | [`perturbation_tasks.py`](evaluation/perturbation_tasks.py) `--task synonymous_codon_substitution` |
| **BRCA2 VEP** | Zero-shot VEP on saturation-mutagenesis BRCA2 ([Huang 2025](https://www.nature.com/articles/s41586-024-08388-8)). Centered 8 kb window + full-LL delta. | [`vep_eval.py`](evaluation/vep_eval.py) |
| **TraitGym Mendelian** | 3,380 fine-mapped non-coding regulatory variants for 113 Mendelian diseases ([Benegas et al. 2025](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v1)). Centered 8 kb window + full-LL delta. | [`vep_eval.py`](evaluation/vep_eval.py) |
| **ClinVar** | Pathogenic vs benign on curated coding + noncoding ClinVar variants. Right-end / next-token scoring with 24 kb left context. | [`clinvar_vep_eval.py`](evaluation/clinvar_vep_eval.py) (uses [`HuggingFaceBio/clinvar-vep-final`](https://huggingface.co/datasets/HuggingFaceBio/clinvar-vep-final) directly) |
| **Genome-NIAH** | Long-context retrieval: insert a (key, value) pair in a real-genome haystack, ask the model to retrieve the value. Four tasks × six context lengths (up to 786 kbp). | [`genome_niah_eval.py`](evaluation/genome_niah_eval.py) |

See [`evaluation/README.md`](evaluation/README.md) for run commands, DNA-tag
flags, and per-benchmark details.

## Finetuning

A minimal end-to-end finetuning example (promoter detection from the
Nucleotide Transformer downstream benchmark) lives in
[`finetuning/`](finetuning). It uses the standard 🤗 Transformers `Trainer`
with `AutoModelForSequenceClassification` on top of the Carbon backbone — swap
in any other classification dataset by changing one flag.

To specialise Carbon on a new clade (e.g. a specific bacterium or protist
that wasn't well represented in the pretraining mix), the same scaffolding
works for **continual pretraining**: load the model with
`AutoModelForCausalLM`, feed it sequences with the `<dna>` tag, and continue
training on next-token loss. The ~10 % prokaryote slice in the pretraining
data means the model already has a reasonable starting point even for
bacterial sequences.

## Citation

```bibtex
@misc{carbon2026,
  title  = {Carbon: Genomic foundation models},
  author = {Hugging Face},
  year   = {2026},
  url    = {https://huggingface.co/HuggingFaceBio}
}
```

## License

Apache 2.0.
