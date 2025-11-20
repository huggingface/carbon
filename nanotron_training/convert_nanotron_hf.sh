#!/bin/bash
# source ~/.bashrc
# conda activate train
#  srun --nodes=1 --gres=gpu:1   --time=06:30:00  --partition=hopper-dev  --job-name=convert  --qos=high  --pty bash

run_name=imgpr-110M-fix
ckpt_num=28000


s3_path="s3://smollm3/blogpost-ablations/$run_name"
cd /fsx/loubna/projects_v2/carbon/nanotron_conversion/nanotron/examples/smollm3/

ckpt_path=/scratch/loubna/carbon/$run_name/$ckpt_num
save_path=/fsx/loubna/projects_v2/carbon/checkpoints/$run_name/$ckpt_num
echo "Copying $s3_path/$ckpt_num/* to $ckpt_path"
s5cmd sync $s3_path/$ckpt_num/* $ckpt_path

mkdir -p $ckpt_path
mkdir -p $save_path
echo "Converting $ckpt_path to $save_path"
torchrun --nproc_per_node=1 convert_nanotron_to_hf.py --checkpoint_path=$ckpt_path --save_path=$save_path --tokenizer_name=hf-carbon/tokenizer-gene
echo "🥳 Done with $ckpt_num!"