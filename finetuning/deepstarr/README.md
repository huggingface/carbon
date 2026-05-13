# DeepSTARR Regression Fine-Tuning

This recipe fine-tunes Carbon on `GenerTeam/DeepSTARR-enhancer-activity`.
It trains on `train`, selects checkpoints by validation `pcc_mean`, and writes
final validation/test PCC plus log-space PCC for the Dev and Hk targets.

## What Is Included

- `deepstarr_best_recipe.py`: minimal script for the best 3B recipe.
- `deepstarr_regression_train.py`: full experimental trainer with extra losses,
  frozen-LM mode, Trackio logging, augmentations, and categorical heads.
- `fsdp2_carbon.yaml`: Accelerate FSDP2 config for Carbon.
- `deepstarr_regression.slurm`: Slurm template for the full trainer.

## Environment

Install the normal Carbon requirements, then make sure the finetuning runtime
pieces are new enough:

```sh
pip install -r requirements.txt
pip install -U "accelerate>=1.7.0" "trackio>=0.25.1"
hf auth whoami
```

Trackio logging is optional. To log to a Space, set `TRACKIO_SPACE_ID` or pass
`--trackio_space_id`.

## Smoke Run

This command exercises tokenization, FSDP2 setup, Trackio-off training, and
metric writing without running a full experiment:

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/deepstarr/deepstarr_best_recipe.py \
  --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
  --output_dir scratch/deepstarr/smoke \
  --max_train_samples 256 \
  --max_eval_samples 128 \
  --max_steps 10 \
  --eval_steps 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2
```

## Best 3B Recipe

The best 3B family of runs used a single GPU, global batch 32, Pearson loss,
`auto_dna_tags=True`, no weight decay, AdamW with beta2 `0.95`, and validation
every about 0.1 epoch. The best observed checkpoint came from continuing a
1.5-epoch run to 2.5 total epochs; the command below is the clean single-stage
version of that recipe.

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/deepstarr/deepstarr_best_recipe.py \
  --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
  --output_dir scratch/deepstarr/carbon-3b-best-recipe \
  --run_name carbon-3b-best-recipe
```

To reproduce the continuation-style run, pass the previous checkpoint:

```sh
--resume_from_checkpoint scratch/deepstarr/previous-run/checkpoint-18858
```

The minimal script intentionally leaves out Trackio and experimental knobs. Use
`deepstarr_regression_train.py` for the larger scan surface.

## Slurm

The Slurm template uses the same defaults as the main run and lets you override
the common knobs with environment variables. It targets
`deepstarr_regression_train.py`, not the minimal best-recipe script.

```sh
TRACKIO_SPACE_ID=hf-carbon/deepstarr-regression \
RUN_NAME=carbon-3b-deepstarr-full-ft \
LEARNING_RATE=2e-5 \
NUM_TRAIN_EPOCHS=5 \
sbatch finetuning/deepstarr/deepstarr_regression.slurm
```

## Useful Variants

The following options are available in `deepstarr_regression_train.py`.

Frozen LM:

```sh
--finetune_mode frozen_lm --learning_rate 3e-4
```

K-mer phase augmentation:

```sh
--kmer_phase_augment \
--kmer_phase_augment_copies 1 \
--kmer_phase_max_shift 5 \
--kmer_phase_output_length 246
```

Reverse complement augmentation:

```sh
--reverse_complement_augment duplicate
```

Train-only token masking:

```sh
--train_token_mask_rate 0.05 --train_token_mask_mode oov
```

Warm start from a previous checkpoint while resetting optimizer/scheduler state:

```sh
--init_from_checkpoint scratch/deepstarr/previous-run/checkpoint-12345
```

## Outputs

The minimal best-recipe script writes:

- `run_config.json`
- `train_results.json`
- `validation_metrics.json`
- `test_metrics.json`
- Trainer checkpoints, with the best checkpoint selected by validation PCC
- `best_model/` only when `--save_final_model` is enabled

The full experimental script can also write:

- `validation_predictions.jsonl` and `test_predictions.jsonl`

The primary model-selection metric is `pcc_mean`. The full metric files also
include `*_pcc_dev_scaled`, `*_pcc_hk_scaled`, `*_log_pcc_dev`,
`*_log_pcc_hk`, and `*_log_pcc_mean`.
