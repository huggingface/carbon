# Nanotron training

## Data tokenization
You can tokenize the data using [datatrove](https://github.com/huggingface/datatrove), we provide a script for tokenizing IMG/PR dataset in `tokenize_imgpr.py` using [hf-carbon/tokenizer-gene](https://huggingface.co/hf-carbon/tokenizer-gene).

## Training setup

Please refer to [nanotron](https://github.com/huggingface/nanotron/) for detailed instructions on setting up your training environment and launching jobs. For training, we use SmolLM3 nanotron [branch](https://github.com/huggingface/nanotron/tree/smollm3), and we use this [branch](https://github.com/huggingface/datatrove/tree/nouamane/avoid-s3) of [datatrove](https://github.com/huggingface/datatrove).

### Legacy trainer note
`nanotron_training/example/FNSTrainer.py` is an example Hugging Face `Trainer`-based BP-loss prototype and is not used by Nanotron training jobs. The active hybrid BP integration now lives in the Nanotron branch (`src/nanotron/models/qwen.py` + `src/nanotron/data/clm_collator.py`).

### Tail-aware `token_mask` requirement (important)
For BP-level training on k-mer tokenized DNA, the final (tail) token may represent fewer than `k` valid base pairs. If this is not encoded, loss will incorrectly supervise padded bases.

Expected `token_mask` semantics per token:
- `-1`: non-DNA/base tokens
- `0`: DNA special tokens
- `1..k`: number of valid supervised bases for that token

Current Nanotron collator behavior (in the Nanotron `carbon` branch):
- If dataset examples contain `token_mask`, collator uses it (preferred, tail-aware).
- Supported per-sample shapes:
  - `seq_len + 1`: shifted right by one to align with labels
  - `seq_len`: already label-aligned
- Nanotron hybrid BP loss now expects dataset-provided `token_mask`; missing/invalid masks should fail early.

Recommendation:
- Add `token_mask` during tokenization/preprocessing and store it in the training dataset to preserve correct tail supervision.

### Hybrid BP preflight
Before launching hybrid BP training (`hybrid_bp_loss_enabled: true`), validate tokenizer semantics and export the k-mer id range:

```bash
python nanotron_training/verify_token_mask_sample.py \
  --tokenizer_path /path/to/hybrid_tokenizer_dir \
  --k 6
```

This prints suggested environment exports for Slurm:

```bash
export ENABLE_HYBRID_BP=1
export HYBRID_BP_DNA_KMER_START_ID=<printed_value>
export HYBRID_BP_DNA_KMER_END_ID=<printed_value>
export HYBRID_BP_K=6
export HYBRID_BP_TOKENIZER_PATH=/path/to/hybrid_tokenizer_dir
```

The provided Slurm scripts now:
- require `HYBRID_BP_DNA_KMER_START_ID` and `HYBRID_BP_DNA_KMER_END_ID` when `ENABLE_HYBRID_BP=1`
- optionally run `verify_token_mask_sample.py` if `HYBRID_BP_TOKENIZER_PATH` is set
- default to non-hybrid behavior when `ENABLE_HYBRID_BP=0`

Below is an example of launching a training on 1 node (you can change the DP value and batch size in the config to change the number of GPUs) and run:

```bash
git clone https://github.com/huggingface/nanotron
cd nanotron
# follow installation
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file smollm3/stage1_8T.yaml
```

You can modify the `nanotron_slurm_1010M_imgpr.slurm` and launch the training with:

```bash
sbatch nanotron_slurm_1010M_imgpr.slurm
```
Don't forget to adjust the paths and create the logs directory before launching the job.

You can find more training configs under [hf-carbon/training-configs](https://huggingface.co/datasets/hf-carbon/training-configs/tree/main)

- Loss:

![image](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/EHTTVCAgSiiqVIwLYCxsa.png)

## Model conversion

You can convert the models to `transformers` using this [PR](https://github.com/huggingface/nanotron/pull/382)
```bash 
# edit the file
bash convert_nanotron_hf.sh
```
