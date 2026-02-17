# Evaluation Setup

This folder contains standalone evaluation scripts.

## Create the virtual env
```sh
uv venv --python 3.12
source .venv/bin/activate
```

## Install dependencies
Install evaluation dependencies with `uv` (from `pyproject.toml`):

```sh
uv pip install -e .
```

Some evals support `--use_evo2` and require Evo2 plus YAML support:

```sh
uv pip install -e ".[evo2]"
```

## Smol check (minimal end-to-end examples)
These run a tiny pass through each script (downloads data + does a small amount of work).
Replace the model with your own if needed.

```sh
python sequence_recovery_eval.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --max_samples 8 \
  --batch_size 1
```

```sh
python clinvar_vep_eval.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --batch_size 1 \
  --num_processes 1 \
  --context_length 12000
```

```sh
python cds_half_shuffle_eval.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --split test \
  --batch_size 1 \
  --max_length 512
```

```sh
python dart_eval_task1.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --chroms chr22 \
  --batch_size 8
```

```sh
python kegg_dna_classifier_train.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --eval_only \
  --batch_size 1 \
  --max_steps 5
```
