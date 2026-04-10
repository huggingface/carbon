# Data generation and processing pipelines

## Setup

Install the local data-generation environment with `uv`:

```sh
uv sync --directory data --python 3.12
```

This project is pinned to Python 3.12.

If you plan to use `--push-to-hub` or the Hub-first decontamination flow, verify your Hugging Face login from the same environment:

```sh
uv run --directory data hf auth whoami
```

The SeqQA generator streams training sources from the Hugging Face Hub, so network access is required.

## SeqQA

### Usage

Create the repo-root `scratch/` directory before running the generator:

```sh
mkdir -p scratch
```

Generate the full training-oriented SeqQA set with the default 40 examples per subtask:

```sh
uv run --directory data python seqqa/generate_data.py \
  --output ../scratch/seqqa_training.jsonl
```

Generate specific subtasks:

```sh
uv run --directory data python seqqa/generate_data.py \
  --subtasks seq_gc_pct restriction_fragment_count \
  --n-per-subtask 10 \
  --output ../scratch/seqqa_example.jsonl
```

Override DNA sequence length bounds for all subtasks:

```sh
uv run --directory data python seqqa/generate_data.py \
  --min-len 100 --max-len 2000 \
  --output ../scratch/seqqa_short.jsonl
```

Push directly to the Hub without a local dataset file:

```sh
uv run --directory data python seqqa/generate_data.py \
  --push-to-hub \
  --hub-repo hf-carbon/seqqa-synth \
  --hub-config v1
```

Duplicate an existing Hub config to a 4x larger variant and push it under a new config name:

```sh
uv run --directory data python seqqa/duplicate_hub_config.py \
  --source-repo hf-carbon/seqqa-synth \
  --source-config v0 \
  --target-config v0_4x \
  --multiplier 4 \
  --push
```

Decontaminate a source Hub config against public LAB-Bench using question text only and push the filtered result back to the same repo under `{config}_decontaminated`:

```sh
uv run --directory data python seqqa/decontaminate.py \
  --source-repo hf-carbon/seqqa-synth \
  --source-config v1
```

To decontaminate against a different single column:

```sh
uv run --directory data python seqqa/decontaminate.py \
  --source-repo hf-carbon/seqqa-synth \
  --source-config v1 \
  --text-column text
```

This pushes:

- `hf-carbon/seqqa-synth` / `v1_decontaminated`
- `hf-carbon/seqqa-synth` / `v1_decontamination_report`

Optional local files should live under `scratch/`; the default workflow is Hub-first.
