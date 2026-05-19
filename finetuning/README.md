# Carbon Fine-Tuning

This directory contains task-specific fine-tuning recipes for Carbon models.

| Recipe | Task | Metric |
|---|---|---|
| [`deepstarr/`](deepstarr/) | DeepSTARR enhancer activity regression | PCC / log PCC |
| [`promoter_activity/`](promoter_activity/) | Random Promoter DREAM activity regression | PCC / Spearman |
| [`finetune_promoter.py`](finetune_promoter.py) | GUE promoter detection | accuracy, F1, MCC, AUROC |

## DeepSTARR

The DeepSTARR recipe includes a minimal regression training script, FSDP2
config, and Slurm launch template used for Carbon 3B enhancer activity
fine-tuning.

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/deepstarr/deepstarr_train.py \
  --model HuggingFaceBio/Carbon-3B \
  --output_dir scratch/deepstarr/carbon-3b-regression-train
```

See [`deepstarr/README.md`](deepstarr/README.md) for the full launch notes.

## Random Promoter DREAM Activity

The promoter activity recipe fine-tunes `HuggingFaceBio/Carbon-500M-remote` on
`HuggingFaceBio/random-promoter-dream-2022` with the validated 200k-example
Pearson-Huber setup. The loss gathers predictions and labels across ranks for
multi-device launches, so Pearson correlation is computed on the global
microbatch rather than independently on each rank.

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/promoter_activity/promoter_activity_train.py \
  --output_dir scratch/promoter_activity/carbon-500m-pearson-huber-200k
```

See [`promoter_activity/README.md`](promoter_activity/) for the full launch
notes.

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
python finetune_promoter.py \
    --model HuggingFaceBio/Carbon-3B \
    --add_dna_tag \
    --output_dir ./outputs/promoter-carbon-3B
```

Multi-GPU with `torchrun`:

```sh
torchrun --nproc_per_node=8 finetune_promoter.py \
    --model HuggingFaceBio/Carbon-3B \
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
