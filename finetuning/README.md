# Carbon Fine-Tuning

This directory contains task-specific fine-tuning recipes for Carbon models.

| Recipe | Task | Metric |
|---|---|---|
| [`deepstarr/`](deepstarr/) | DeepSTARR enhancer activity regression | PCC / log PCC |
| [`promoter_activity/`](promoter_activity/) | Random Promoter DREAM activity regression | PCC / Spearman |
| [`malinois/`](malinois/) | Malinois MPRA activity regression | PCC / Spearman |
| [`finetune_promoter.py`](finetune_promoter.py) | GUE promoter detection | accuracy, F1, MCC, AUROC |
| [`finetune_sft.py`](finetune_sft.py) | Supervised fine-tuning with FNS | perplexity / loss |

## Environment

Use the repository environment before running the recipe scripts:

```sh
uv sync --frozen
source .venv/bin/activate
hf auth whoami
```

The Slurm wrappers in this directory are single-node launch templates. For
multi-node training, use a matching Accelerate config and pass the appropriate
machine-rank/main-process settings explicitly.

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

## Malinois MPRA

The Malinois recipe fine-tunes Carbon on the Gosai et al. MPRA regression table
for three cell-type activity targets: K562, HepG2, and SK-N-SH. It uses the
benchmark chromosome holdout split and an MSE-only full fine-tuning recipe.

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/malinois/malinois_train.py \
  --model HuggingFaceBio/Carbon-3B \
  --output_dir scratch/malinois/carbon-3b-mse-smoke \
  --max_train_samples 512 \
  --max_eval_samples 128 \
  --max_steps 10 \
  --eval_steps 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --skip_test
```

See [`malinois/README.md`](malinois/README.md) for the full launch notes.

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

## Supervised Fine-Tuning with FNSTrainer

The `finetune_sft.py` script performs autoregressive language modeling on DNA
sequences using **Factorized Nucleotide Supervision (FNS)**. FNSTrainer
applies base-pair level loss for DNA k-mer tokens and token-level loss for BPE
tokens, providing finer-grained supervision than standard causal language
modeling.

### FNS Loss

For DNA k-mer tokens, FNS marginalizes token-level predictions to
nucleotide-level predictions. For each position `i` in a k-mer:

1. Compute token probabilities: `P(token | context)`
2. Marginalize to nucleotide probabilities: `P(nucleotide_i | context) = Σ P(token | context)` for all tokens with `nucleotide_i` at position `i`
3. Apply cross-entropy loss at nucleotide level

This provides k× more supervision signal per token compared to standard
token-level loss, particularly useful for DNA sequence modeling where
individual nucleotides matter.

### Usage

Single GPU:

```sh
python finetune_sft.py \
    --model HuggingFaceBio/Carbon-3B \
    --dataset your/dataset \
    --output_dir ./outputs/sft-carbon-3B
```

Multi-GPU with `torchrun`:

```sh
torchrun --nproc_per_node=8 finetune_sft.py \
    --model HuggingFaceBio/Carbon-3B \
    --dataset your/dataset \
    --batch_size 4 --grad_accum 4 \
    --output_dir ./outputs/sft-carbon-3B
```

DNA-only loss (ignore BPE tokens):

```sh
python finetune_sft.py \
    --model HuggingFaceBio/Carbon-3B \
    --dataset your/dataset \
    --dna_loss_only \
    --output_dir ./outputs/sft-carbon-3B-dna-only
```

### Key Arguments

- `--dna_loss_only`: Only compute loss on DNA k-mer tokens, ignore BPE tokens
- `--add_dna_tag`: Wrap sequences with `<dna>...</dna>` tags (default: enabled)
- `--sequence_column`: Column name for sequences in dataset (default: "sequence")
- `--max_length`: Maximum sequence length (default: 2048)

### Dataset Format

Your dataset should have a `sequence` column (or specify via `--sequence_column`)
containing DNA sequences. The script automatically wraps sequences with
`<dna>...</dna>` tags for hybrid tokenizers and creates labels for causal LM.

Example dataset structure:
```python
{
    "sequence": ["ATCGATCG...", "GCTAGCTA...", ...]
}
```

### Implementation

The FNS loss is implemented in [`fns_trainer.py`](fns_trainer.py) as a custom
`Trainer` subclass. It automatically classifies tokens and applies appropriate
loss:

- **DNA k-mer tokens**: Base-pair level loss (marginalizes over k-mer vocabulary)
- **BPE tokens + DNA special tokens** (`<dna>`, `</dna>`, `<oov>`): Token-level cross-entropy loss

The trainer efficiently caches nucleotide mappings and DNA k-mer masks at the
first forward pass, and is compatible with DDP multi-GPU training.

### Dependencies

```sh
uv sync --frozen
source .venv/bin/activate
```
