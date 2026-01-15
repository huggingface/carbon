# Nanotron training

## Data tokenization
You can tokenize the data using [datatrove](https://github.com/huggingface/datatrove), we provide a script for tokenizing IMG/PR dataset in `tokenize_imgpr.py` using [hf-carbon/tokenizer-gene](https://huggingface.co/hf-carbon/tokenizer-gene).

## Training setup

Please refer to [nanotron](https://github.com/huggingface/nanotron/) for detailed instructions on setting up your training environment and launching jobs. For training, we use SmolLM3 nanotron [branch](https://github.com/huggingface/nanotron/tree/smollm3), and we use this [branch](https://github.com/huggingface/datatrove/tree/nouamane/avoid-s3) of [datatrove](https://github.com/huggingface/datatrove).

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
