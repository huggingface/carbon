# Carbon Fine-Tuning

This directory contains task-specific fine-tuning recipes for Carbon models.

| Recipe | Task | Metric |
|---|---|---|
| [`deepstarr/`](deepstarr/) | DeepSTARR enhancer activity regression | PCC / log PCC |
| [`finetune_promoter.py`](finetune_promoter.py) | GUE promoter detection | accuracy, F1, MCC, AUROC |

## DeepSTARR

The DeepSTARR recipe includes a minimal best-recipe script, the full
experimental trainer, FSDP2 config, and Slurm launch template used for Carbon
3B/8B enhancer activity fine-tuning.

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/deepstarr/deepstarr_best_recipe.py \
  --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
  --output_dir scratch/deepstarr/carbon-3b-best-recipe
```

See [`deepstarr/README.md`](deepstarr/README.md) for the full launch notes.

## Promoter Detection

A minimal example of fine-tuning Carbon on a downstream DNA task using the
standard Transformers `Trainer`. We pick promoter detection because it is the
canonical short-sequence DNA classification task used by Evo2 and GENERator in
their downstream benchmarks via the GUE / Nucleotide Transformer downstream
suites.

### Task

Binary classification: is a 300 bp human DNA segment a transcription start
site promoter?

Dataset: `InstaDeepAI/nucleotide_transformer_downstream_tasks`, config
`promoter_all` (about 53K train / 5.9K test, balanced).

### Run

Single GPU:

```sh
python finetuning/finetune_promoter.py \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --add_dna_tag \
    --output_dir ./outputs/promoter-carbon-3B
```

Multi-GPU with `torchrun`:

```sh
torchrun --nproc_per_node=8 finetuning/finetune_promoter.py \
    --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
    --add_dna_tag \
    --batch_size 4 --grad_accum 4 \
    --output_dir ./outputs/promoter-carbon-3B
```

### DNA Tags

Same rule as the eval scripts: pass `--add_dna_tag` for Carbon hybrid models
(`carbon-3B-hybrid-*`, `carbon-8B-hybrid-*`) so the tokenizer routes to 6-mer
DNA mode. Omit it for pure-DNA models and for non-Carbon DNA LMs.

### Adapting To Other Tasks

The promoter script uses `AutoModelForSequenceClassification`, so you can swap
in any other `nucleotide_transformer_downstream_tasks` config (`enhancers`,
`H3K4me3`, `splice_sites_all`, etc.) by changing `--config`. For longer
sequences, raise `--max_length`. For multi-class tasks, the head adapts
automatically because Transformers infers `num_labels` from the dataset labels.

For tasks beyond NT downstream, the same scaffolding works. Write a dataset
loader that yields `(sequence, label)` and keep
`AutoModelForSequenceClassification` as the model class.

### Dependencies

```sh
pip install transformers datasets torch scikit-learn
```
