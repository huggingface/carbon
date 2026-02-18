# Evaluation Setup

This folder contains standalone evaluation scripts.

## Create the virtual env
```sh
uv venv --python 3.12
```

## Install dependencies
Install evaluation dependencies with `uv` (from `pyproject.toml`):

```sh
uv sync
```

Some evals support `--use_evo2` and require Evo2 plus YAML support:

```sh
uv sync --extra evo2
```

## LightEval
For pretrained models we use log-likelihood metrics (logit-based scoring). Task specs follow the `task|few_shot` format (or just `task` to default to 0). The following example is a sanity check against `SmolLM3-3B-Base` (should yield 19.7%):

```sh
uv run accelerate launch --config_file ddp.yaml -m lighteval accelerate \
  "model_name=HuggingFaceTB/SmolLM3-3B-Base,dtype=bfloat16,batch_size=8" \
  "mmlu_pro_cf|0" \
  --custom-tasks mmlu_pro.py \
  --output-dir .
```

Biology-only subset:

```sh
uv run accelerate launch --config_file ddp.yaml -m lighteval accelerate \
  "model_name=HuggingFaceTB/SmolLM3-3B-Base,dtype=bfloat16,batch_size=8" \
  "mmlu_pro_biology_cf|0" \
  --custom-tasks mmlu_pro_biology.py \
  --output-dir .
```

Basic DNA subset:

```sh
uv run accelerate launch --config_file ddp.yaml -m lighteval accelerate \
  "model_name=HuggingFaceTB/SmolLM3-3B-Base,dtype=bfloat16,batch_size=8" \
  "basic_dna_cf|0" \
  --custom-tasks basic_dna.py \
  --output-dir .
```

## Smol check (minimal end-to-end examples)
These run a tiny pass through each script (downloads data + does a small amount of work).
Replace the model with your own if needed.

```sh
uv run python sequence_recovery_eval.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --max_samples 8 \
  --batch_size 1
```

```sh
uv run python clinvar_vep_eval.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --batch_size 1 \
  --num_processes 1 \
  --context_length 12000
```

```sh
uv run python cds_half_shuffle_eval.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --split test \
  --batch_size 1 \
  --max_length 512
```

```sh
uv run python dart_eval_task1.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --chroms chr22 \
  --batch_size 8
```

```sh
uv run python kegg_dna_classifier_train.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --eval_only \
  --batch_size 1 \
  --max_steps 5
```
