# Random Promoter DREAM Activity Fine-Tuning

This recipe fine-tunes Carbon-500M on the Random Promoter DREAM Challenge 2022
sequence-to-expression task using the validated 200k-example setup:

- dataset: `HuggingFaceBio/random-promoter-dream-2022`, config `supervised`
- model: `HuggingFaceBio/Carbon-500M-remote`
- objective: `1 - Pearson r + 0.2 * Huber`
- tokenizer mode: `auto_dna_tags`
- training: one epoch over 200k examples

In our single-process run, this setup reached test PCC `0.807` and Spearman
`0.806` on the 71,103-sequence labeled test file. These are plain full-test
correlations, not official DREAM subset-weighted leaderboard scores.

The Pearson-Huber loss gathers predictions and labels across distributed ranks
before computing Pearson correlation, so multi-device runs optimize the global
microbatch correlation instead of a per-rank local correlation.

## Smoke Run

```sh
python finetuning/promoter_activity/promoter_activity_train.py \
  --output_dir scratch/promoter_activity/smoke-500m \
  --max_train_samples 128 \
  --max_eval_samples 64 \
  --max_steps 2 \
  --eval_steps 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4
```

## Validated Run

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/promoter_activity/promoter_activity_train.py \
  --output_dir scratch/promoter_activity/carbon-500m-pearson-huber-200k \
  --run_name carbon-500m-promoter-pearson-huber-200k
```

## Multi-Device Run

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 8 \
  finetuning/promoter_activity/promoter_activity_train.py \
  --output_dir scratch/promoter_activity/carbon-500m-pearson-huber-200k-8gpu \
  --run_name carbon-500m-promoter-pearson-huber-200k-8gpu \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1
```

Keep the global per-step batch size at least 2, and preferably much larger,
because Pearson correlation is poorly conditioned on very small batches.

## Slurm

```sh
sbatch finetuning/promoter_activity/promoter_activity_regression.slurm
```

For eight GPUs on one node:

```sh
GPUS_PER_NODE=8 \
NUM_PROCESSES=8 \
sbatch --gres=gpu:8 finetuning/promoter_activity/promoter_activity_regression.slurm
```

Common overrides:

```sh
MAX_TRAIN_SAMPLES=10000 \
MAX_EVAL_SAMPLES=2000 \
MAX_STEPS=100 \
sbatch finetuning/promoter_activity/promoter_activity_regression.slurm
```

## Outputs

The trainer writes `run_config.json`, train metrics, validation/test metrics,
and validation/test prediction TSVs with raw-scale labels and predictions.
