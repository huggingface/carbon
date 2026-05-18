# DeepSTARR Regression Fine-Tuning

This recipe fine-tunes Carbon on `GenerTeam/DeepSTARR-enhancer-activity`.
It trains on `train`, selects checkpoints by validation `pcc_mean`, and writes
validation/test PCC metrics for the Dev and Hk targets.

## Files

- `deepstarr_train.py`: Trainer-based DeepSTARR regression script.
- `fsdp2_carbon.yaml`: Accelerate FSDP2 config for Carbon.
- `deepstarr_regression.slurm`: Slurm wrapper for the same script.

## Environment

Install the normal Carbon requirements, then make sure the finetuning runtime is
available:

```sh
pip install -r requirements.txt
pip install -U "accelerate>=1.7.0"
hf auth whoami
```

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
