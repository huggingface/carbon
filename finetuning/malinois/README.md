# Malinois MPRA Regression Fine-Tuning

This recipe fine-tunes Carbon on the Gosai et al. MPRA table used by the
Malinois benchmark. It predicts three log2 fold-change activity targets:
K562, HepG2, and SK-N-SH. The split follows the benchmark chromosomes:
validation on chromosomes 19, 21, and X, and test on chromosomes 7 and 13.

## Files

- `malinois_train.py`: MSE-only Trainer recipe for the MPRA regression task.
- `malinois_regression.slurm`: 8-GPU Slurm wrapper for the same recipe.

## What This Keeps

The defaults keep the settings that worked in our runs: full fine-tuning, MSE
on train-z-scored log2FC labels, `auto_dna_tags=True`, reverse-complement train
duplication, high-activity row duplication, reverse-complement averaged final
metrics, no weight decay, global batch size 256 on 8 GPUs, and `lr=1e-5`.

The script intentionally omits the exploratory loss variants, plotting code,
local data downloads, and report-generation utilities from the experiment
workspace.

## Smoke Run

```sh
accelerate launch \
  --config_file finetuning/deepstarr/fsdp2_carbon.yaml \
  --num_processes 1 \
  finetuning/malinois/malinois_train.py \
  --model HuggingFaceBio/Carbon-3B \
  --output_dir scratch/malinois/smoke \
  --max_train_samples 512 \
  --max_eval_samples 128 \
  --max_steps 10 \
  --eval_steps 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --skip_test
```

## Full 8-GPU Run

```sh
RUN_NAME=carbon-3b-malinois-mse \
sbatch finetuning/malinois/malinois_regression.slurm
```

For the exact internal 3B checkpoint used in our experiments, set `MODEL` to
the accessible checkpoint repo before submitting. Common scale-sweep overrides
are:

```sh
MODEL=HuggingFaceBio/Carbon-500M-remote \
RUN_NAME=carbon-500m-malinois-mse \
sbatch finetuning/malinois/malinois_regression.slurm

MODEL=hf-carbon/carbon-8B-hybrid-loss-1T-v1 \
RUN_NAME=carbon-8b-malinois-mse \
TORCH_DTYPE=bfloat16 \
sbatch finetuning/malinois/malinois_regression.slurm
```

## Outputs

The script writes:

- `run_config.json`
- `train_results.json`
- `validation_metrics.json`
- `test_metrics.json` unless `--skip_test` is set
- Trainer checkpoints, with the best checkpoint selected by validation PCC
- `best_model/` when `--save_model` is set

The primary selection metric is validation mean Pearson correlation across the
three cell-type targets. Metric files also include target-wise Pearson and
Spearman correlations.
