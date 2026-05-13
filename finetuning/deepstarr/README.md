# DeepSTARR Regression Fine-Tuning

This recipe fine-tunes Carbon on `GenerTeam/DeepSTARR-enhancer-activity`.
It trains on `train`, selects checkpoints by validation `pcc_mean`, and writes
final validation/test PCC plus log-space PCC for the Dev and Hk targets.

## What Is Included

- Full fine-tuning and frozen-LM modes.
- FSDP2 launch config for Carbon 3B/8B.
- Direct Trackio logging because `Trainer` does not support Trackio natively.
- Carbon tokenizer options: explicit `<dna>...</dna>` tags, `auto_dna_tags=True`,
  and truncation to multiples of 6 to avoid tail k-mer padding.
- Regression losses: `mse`, `huber`, `pearson`, `mse_pearson`, `ccc`,
  `mse_ccc`, and `pearson_calibrated`.
- Categorical target-bin mode.
- Train-only augmentations: Carbon k-mer phase jitter, reverse complement
  views, and DNA 6-mer token masking.

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
  finetuning/deepstarr/deepstarr_regression_train.py \
  --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
  --output_dir scratch/deepstarr/smoke \
  --finetune_mode full_finetune \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --loss_type pearson \
  --dna_tokenization_mode auto_dna_tags \
  --truncate_dna_to_multiple 6 \
  --max_train_samples 256 \
  --max_eval_samples 128 \
  --max_steps 10 \
  --eval_strategy steps \
  --save_strategy steps \
  --eval_steps 5 \
  --save_steps 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --torch_dtype float32 \
  --bf16 \
  --require_fp32_master_weights \
  --gradient_checkpointing \
  --attn_implementation kernels-community/flash-attn3 \
  --disable_trackio
```

## Main 3B Run

```sh
export TRACKIO_SPACE_ID=hf-carbon/deepstarr-regression

accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 8 \
  finetuning/deepstarr/deepstarr_regression_train.py \
  --model hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 \
  --output_dir scratch/deepstarr/carbon-3b-full-ft \
  --run_name carbon-3b-full-ft \
  --trackio_run_name carbon-3b-full-ft \
  --trackio_group carbon-3b \
  --finetune_mode full_finetune \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --loss_type pearson \
  --label_transform dataset_scaled \
  --head_type sequence \
  --dna_tokenization_mode auto_dna_tags \
  --truncate_dna_to_multiple 6 \
  --max_length 512 \
  --num_train_epochs 5 \
  --eval_strategy steps \
  --save_strategy steps \
  --eval_steps 8555 \
  --save_steps 8555 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --torch_dtype float32 \
  --bf16 \
  --require_fp32_master_weights \
  --gradient_checkpointing \
  --attn_implementation kernels-community/flash-attn3 \
  --trackio
```

For the 8B model, change `--model` to `hf-carbon/carbon-8B-hybrid-loss-1T-v1`
and reduce batch size if needed.

## Slurm

The Slurm template uses the same defaults as the main run and lets you override
the common knobs with environment variables:

```sh
TRACKIO_SPACE_ID=hf-carbon/deepstarr-regression \
RUN_NAME=carbon-3b-deepstarr-full-ft \
LEARNING_RATE=2e-5 \
NUM_TRAIN_EPOCHS=5 \
sbatch finetuning/deepstarr/deepstarr_regression.slurm
```

## Useful Variants

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

Each run writes:

- `run_config.json`
- `train_results.json`
- `validation_metrics.json`
- `test_metrics.json`
- `validation_predictions.jsonl` and `test_predictions.jsonl`
- `best_model/` when `--save_final_model` is enabled

The primary model-selection metric is `pcc_mean`. The full metric files also
include `*_pcc_dev_scaled`, `*_pcc_hk_scaled`, `*_log_pcc_dev`,
`*_log_pcc_hk`, and `*_log_pcc_mean`.
