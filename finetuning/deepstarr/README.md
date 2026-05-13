# DeepSTARR Regression Fine-Tuning

This recipe fine-tunes Carbon on `GenerTeam/DeepSTARR-enhancer-activity`.
It trains on `train`, selects checkpoints by validation `pcc_mean`, and writes
final validation/test PCC plus log-space PCC for the Dev and Hk targets.

## What Is Included

- `deepstarr_train.py`: minimal DeepSTARR regression training script.
- `fsdp2_carbon.yaml`: Accelerate FSDP2 config for Carbon.
- `deepstarr_regression.slurm`: Slurm template for the minimal recipe.

## Environment

Install the normal Carbon requirements, then make sure the finetuning runtime
pieces are new enough for FSDP2:

```sh
pip install -r requirements.txt
pip install -U "accelerate>=1.7.0"
hf auth whoami
```

## Smoke Run

This command exercises tokenization, FSDP2 setup, training, and metric writing
without running a full experiment:

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/deepstarr/deepstarr_train.py \
  --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
  --output_dir scratch/deepstarr/smoke \
  --max_train_samples 256 \
  --max_eval_samples 128 \
  --max_steps 10 \
  --eval_steps 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2
```

## 3B Regression Recipe

The best 3B family of runs used a single GPU, global batch 32, Pearson loss,
`auto_dna_tags=True`, no weight decay, AdamW with beta2 `0.95`, and validation
every about 0.1 epoch. The best observed checkpoint came from continuing a
1.5-epoch run to 2.5 total epochs; the command below is the clean single-stage
version of that recipe.

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/deepstarr/deepstarr_train.py \
  --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
  --output_dir scratch/deepstarr/carbon-3b-regression-train \
  --run_name carbon-3b-regression-train
```

To reproduce the continuation-style run, pass the previous checkpoint:

```sh
--resume_from_checkpoint scratch/deepstarr/previous-run/checkpoint-18858
```

## Slurm

The Slurm template uses the same defaults as the main run and lets you override
the common knobs with environment variables.

```sh
RUN_NAME=carbon-3b-deepstarr-full-ft \
LEARNING_RATE=2e-5 \
NUM_TRAIN_EPOCHS=2.5 \
sbatch finetuning/deepstarr/deepstarr_regression.slurm
```

## Outputs

The script writes:

- `run_config.json`
- `train_results.json`
- `validation_metrics.json`
- `test_metrics.json`
- Trainer checkpoints, with the best checkpoint selected by validation PCC
- `best_model/` only when `--save_final_model` is enabled

The primary model-selection metric is `pcc_mean`. The metric files also include
`*_pcc_dev_scaled`, `*_pcc_hk_scaled`, `*_log_pcc_dev`, `*_log_pcc_hk`, and
`*_log_pcc_mean`.
