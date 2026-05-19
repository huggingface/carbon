# DeepSTARR Regression Fine-Tuning

This recipe fine-tunes Carbon on `GenerTeam/DeepSTARR-enhancer-activity`.
It trains on `train`, selects checkpoints by validation `pcc_mean`, and writes
validation/test PCC metrics for the Dev and Hk targets.

## Files

- `deepstarr_train.py`: Trainer-based DeepSTARR regression script.
- `fsdp2_carbon.yaml`: Accelerate FSDP2 config for Carbon.
- `deepstarr_regression.slurm`: Slurm wrapper for the same script.

## Environment

Install the repository environment and make sure the Hugging Face CLI can see
your token:

```sh
uv sync --frozen
source .venv/bin/activate
hf auth whoami
```

The Slurm wrappers default to `.venv/bin/accelerate`, so run them from a synced
checkout or override `ACCELERATE`. If launching multiple Accelerate jobs on the
same node, set distinct `MAIN_PROCESS_PORT` values for each job.

## Smoke Run

This exercises tokenization, model loading, FSDP2, training, checkpointing, and
metric writing on a small subset:

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/deepstarr/deepstarr_train.py \
  --model HuggingFaceBio/Carbon-3B \
  --output_dir scratch/deepstarr/smoke \
  --max_train_samples 256 \
  --max_eval_samples 128 \
  --max_steps 10 \
  --eval_steps 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --skip_test
```

## Full 3B Run

The defaults use full fine-tuning, Pearson loss, `auto_dna_tags=True`, raw DNA
truncation to a multiple of 6, no weight decay, bf16 compute, FlashAttention 3,
and AdamW with beta2 `0.95`.

FlashAttention 3 is selected with
`--attn_implementation kernels-community/flash-attn3` and requires compatible
GPU, CUDA, and Transformers versions. For a more portable smoke run, use
`--attn_implementation sdpa`; for Slurm, set `ATTN_IMPLEMENTATION=sdpa`.

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/deepstarr/deepstarr_train.py \
  --model HuggingFaceBio/Carbon-3B \
  --output_dir scratch/deepstarr/carbon-3b-regression-train \
  --run_name carbon-3b-regression-train
```

Use `--save_model` to write a copy of the best loaded model to `best_model/`.

## Slurm

Submit the same recipe from the repository root:

```sh
RUN_NAME=carbon-3b-deepstarr-full-ft \
sbatch finetuning/deepstarr/deepstarr_regression.slurm
```

Common overrides include `MAX_STEPS`, `MAX_TRAIN_SAMPLES`,
`MAX_EVAL_SAMPLES`, `EVAL_STEPS`, `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`,
`PER_DEVICE_TRAIN_BATCH_SIZE`, `PER_DEVICE_EVAL_BATCH_SIZE`, `SKIP_TEST`, and
`SAVE_MODEL`.

## Outputs

The script writes:

- `run_config.json`
- `train_results.json`
- `validation_metrics.json`
- `test_metrics.json` unless `--skip_test` is set
- Trainer checkpoints, with the best checkpoint selected by validation PCC
- `best_model/` when `--save_model` is set

The primary model-selection metric is `pcc_mean`. Metric files also include
Dev/Hk PCC and log-label PCC values.
