import argparse
import os

parser = argparse.ArgumentParser("Quick tokenization.")

# python tokenize_datatrove.py hf://datasets/HuggingFaceTB/carbon-raw-data/imgpr_sequences --save_name imgpr  --n_tasks 50 --tokenizer  hf-carbon/tokenizer-gene --text_key sequence --chunk_size 2048 --seed 0
# edit paths in the script

parser.add_argument("data_path", type=str, help="Path to the data to tokenize.")
parser.add_argument("--n_tasks", type=int, help="nb of tokenization tasks", default=100)
parser.add_argument("--tokenizer", type=str, help="tokenizer to use", default="hf-carbon/tokenizer-gene")
parser.add_argument("--max_toks", type=int, help="max tokens per file", default=1e9)
parser.add_argument("--text_key", type=str, default="sequence")
parser.add_argument("--chunk_size", type=int, help="chunk size", default=2048)
parser.add_argument("--seed", type=int, help="seed", default=0)
parser.add_argument("--save_name", type=str, help="dataset save name", default="default")


if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
    from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
    from datatrove.pipeline.filters import LambdaFilter


    print(f"📝 CONTEXT_SIZE: {args.chunk_size}, TOKENIZER: {args.tokenizer}, SAVE_NAME: {args.save_name}, SEED: {args.seed}")
    print(f"📖 Data read path: {args.data_path}")
    print(f"💾 Data save path: s3://smollm3/datasets/hf-carbon/dataset_{args.chunk_size}/{args.save_name}/tokenized/")
    print(f"💾 Logs save path: /fsx/loubna/projects_v2/logs_carbon/tokenization_{args.chunk_size}_v2/tokenize_{args.save_name}_s{args.seed}")

    dist_executor = SlurmPipelineExecutor(
        job_name=f"tokenize-{args.save_name}-{args.chunk_size}-s{args.seed}",
        pipeline=[
            ParquetReader(
                args.data_path, # read directly from huggingface
                glob_pattern="*.parquet",
                text_key=args.text_key,
            ),
        DocumentTokenizer(
            # output_folder=f"s3://smollm3/datasets/hf-carbon/dataset_{args.chunk_size}/{args.save_name}/tokenized/",
            output_folder="/fsx/loubna/projects_v2/carbon/data_fix/{args.save_name}_tokenized/tokenized/",
            tokenizer_name_or_path=args.tokenizer,
            eos_token="<|endoftext|>",
            local_working_dir=f"/scratch/loubna/{args.save_name}/tokenized/",
            save_filename=f"epoch_0{args.seed}_{args.save_name}",
            batch_size=100,
            max_tokens_per_file=args.max_toks,
            # Max 1 GT per file (i.e. btw 5 et 300 tokenized files per dump et about 100 dump extracts per merged file)
            shuffle_documents=True,
            shuffle_chunk_size=args.chunk_size,
            seed=args.seed,
        ),
    ],
    tasks=args.n_tasks,
    time="12:00:00",
    partition="hopper-cpu",
    cpus_per_task=24, # OOM with less
    qos="normal",
    mem_per_cpu_gb=2,
    logging_dir=f"/fsx/loubna/projects_v2/logs_carbon/tokenization_{args.chunk_size}_v2/tokenize_{args.save_name}_s{args.seed}",
    )
    dist_executor.run()

    merge_executor = SlurmPipelineExecutor(
        job_name=f"merge-{args.save_name}-{args.chunk_size}-s{args.seed}",
        pipeline=[
            DocumentTokenizerMerger(
                # input_folder=f"s3://smollm3/datasets/hf-carbon/dataset_{args.chunk_size}/{args.save_name}/tokenized/",
                # output_folder=f"s3://smollm3/datasets/hf-carbon/dataset_{args.chunk_size}/{args.save_name}/standard/",
                input_folder="/fsx/loubna/projects_v2/carbon/data_fix/{args.save_name}_tokenized/tokenized/",
                output_folder="/fsx/loubna/projects_v2/carbon/data_fix/{args.save_name}_tokenized/standard/",
                # output_folder=f"/fsx/{os.getlogin()}/datasets/llama_tokenized/{args.save_name}/standard/",
                save_filename=f"epoch_0{args.seed}_{args.save_name}",
                shuffle_chunk_size=args.chunk_size,
                shuffle=True,
                seed=args.seed,
            ),
        ],
        tasks=1,
        time="50:00:00",
        partition="hopper-cpu",
        logging_dir=f"/fsx/loubna/projects_v2/logs_carbon/tokenization_{args.chunk_size}_v2/tokenize_{args.save_name}_merged_s{args.seed}",
        cpus_per_task=24,
        qos="normal",
        depends=dist_executor,
    )

    merge_executor.run()