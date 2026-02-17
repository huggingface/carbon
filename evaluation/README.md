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

## Smoke check (ensures scripts start)
Run a quick import/argparse check (no heavy compute):

```sh
python sequence_recovery_eval.py --help
python clinvar_vep_eval.py --help
python cds_half_shuffle_eval.py --help
python dart_eval_task1.py --help
python kegg_dna_classifier_train.py --help
```

If these commands print help text without import errors, the environment is ready for the evals.
