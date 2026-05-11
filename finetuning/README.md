# Carbon finetuning 

**⚠️ PLACEHOLDER, REPLACE WITH LEWIS & ED's SETUP**

A minimal example of finetuning Carbon on a downstream DNA task using the
standard 🤗 Transformers `Trainer`. We pick **promoter detection** because
it's the canonical short-sequence DNA classification task — used by both
[Evo2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1) and
[GENERator](https://arxiv.org/abs/2502.07272) in their downstream benchmarks
via the [GUE](https://arxiv.org/abs/2306.15006) /
[Nucleotide Transformer downstream](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks)
suites.

## Task

Binary classification: is a 300 bp human DNA segment a transcription start
site (promoter)?

Dataset: [`InstaDeepAI/nucleotide_transformer_downstream_tasks`](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks),
config `promoter_all` (~53K train / 5.9K test, balanced).

## Run

Single GPU:

```bash
python finetune_promoter.py \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --add_dna_tag \
    --output_dir ./outputs/promoter-carbon-3B
```

Multi-GPU with `torchrun`:

```bash
torchrun --nproc_per_node=8 finetune_promoter.py \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --add_dna_tag \
    --batch_size 4 --grad_accum 4 \
    --output_dir ./outputs/promoter-carbon-3B
```

## DNA tags

Same rule as the eval scripts: pass `--add_dna_tag` for Carbon **hybrid**
models (`carbon-3B-hybrid-*`, `carbon-8B-hybrid-*`) so the tokenizer routes
to 6-mer DNA mode. Omit it for pure-DNA models and for non-Carbon DNA LMs.

## Adapting to other tasks

The script uses `AutoModelForSequenceClassification`, so you can swap in any
other `nucleotide_transformer_downstream_tasks` config (`enhancers`,
`H3K4me3`, `splice_sites_all`, …) by changing `--config`. For longer
sequences, raise `--max_length`. For multi-class tasks, the head adapts
automatically — Transformers infers `num_labels` from the dataset labels.

For tasks beyond NT downstream — DART-eval enhancer activity, ClinVar
classification heads, etc. — the same scaffolding works; you only need to
write a dataset loader that yields `(sequence, label)` and to keep
`AutoModelForSequenceClassification` as the model class.

## Dependencies

```bash
pip install transformers datasets torch scikit-learn
```
