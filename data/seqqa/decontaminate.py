"""
This script is used to decontaminate a dataset by checking for n-gram overlap with other datasets.
It uses the same approach presented in https://huggingface.co/papers/2501.19393,
as found in: https://github.com/simplescaling/s1/blob/main/data/decontaminate_util.py

Usage:

uv run --directory data python seqqa/decontaminate.py \
    --dataset hf-carbon/seqqa-sft-v1_gemma-4-31B-it \
    --ngram_size 8 \
    --text_column question
"""

import argparse
import collections
import logging

from datasets import Dataset, load_dataset
from tqdm import tqdm


logger = logging.getLogger(__name__)


def normalize_string(text: str) -> str:
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text


def word_ngrams(text: str, n: int) -> list:
    """Generate word-level n-grams from text."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def build_ngram_lookup(documents: list[str], ngram_size: int = 8) -> dict[str, set[int]]:
    """Build ngram lookup for documents."""
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(tqdm(documents)):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(doc_id)

    return lookup


def build_ngram_single(document: str, ngram_size: int = 8) -> set[str]:
    normalized_text = normalize_string(document)
    ngrams = word_ngrams(normalized_text, ngram_size)

    return set(ngrams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to check for contamination.")
    parser.add_argument("--config", type=str, default="default", help="Name of the dataset config to load.")
    parser.add_argument("--ngram_size", type=int, default=8, help="Size of n-grams to build, defaults to 8.")
    parser.add_argument("--split", type=str, default="train", help="the split")
    parser.add_argument(
        "--text_column", type=str, default="text", help="Name of the column containing the text to check."
    )
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes to use.")
    parser.add_argument(
        "--num_shards", type=int, default=None, help="Number of shards to use when pushing to the Hub."
    )
    args = parser.parse_args()

    # Load the dataset to check for contamination
    ds = load_dataset(args.dataset, name=args.config, split=args.split, num_proc=args.num_proc)

    eval_datasets = {
        "lab_bench": (load_dataset("futurehouse/lab-bench", "SeqQA", split="train", num_proc=args.num_proc), "question"),
    }
    ngram_lookups = {}
    for ds_name, (eval_dataset, problem_col) in eval_datasets.items():
        ngram_lookups[ds_name] = build_ngram_lookup(eval_dataset[problem_col], ngram_size=args.ngram_size)

    for eval_name, ngram_lookup in ngram_lookups.items():
        # Update the ngram_lookup variable for each dataset
        def find_contaminated(row, eval_name=eval_name, ngram_lookup=ngram_lookup):
            if args.text_column == "messages":
                for message in row["messages"]:
                    if message["role"] == "user":
                        text = message["content"]
                        break
            elif args.text_column == "chosen":
                for message in row["chosen"]:
                    if message["role"] == "user":
                        text = message["content"]
                        break
            else:
                text = row[args.text_column]

            # For each example we have to build the ngrams and check for all of them on each row
            ngrams = build_ngram_single(text, ngram_size=args.ngram_size)
            row[f"contaminated_{eval_name}"] = any(ngram in ngram_lookup for ngram in ngrams)
            return row

        ds = ds.map(find_contaminated, num_proc=args.num_proc, desc=f"Checking contamination with {eval_name}")

    # Allow cleaning up via CLI args (removing the contaminated examples and dropping the columns)
    def cleanup(dataset: Dataset) -> Dataset:
        initial_size = len(dataset)
        contamination_cols = [col for col in dataset.column_names if col.startswith("contaminated_")]
        for col in contamination_cols:
            if col.startswith("contaminated_"):
                size_prior = len(dataset)
                dataset = dataset.filter(
                    lambda x, col=col: not x[col],
                    num_proc=args.num_proc,
                    desc=f"Removing contaminated samples from {col}",
                )
                if len(dataset) < size_prior:
                    logger.info(
                        f"Removed {size_prior - len(dataset)} samples from '{col.replace('contaminated_', '')}'"
                    )
        dataset = dataset.remove_columns(contamination_cols)
        logger.info(f"Initial size: {initial_size}, Final size: {len(dataset)}")
        return dataset

    ds = cleanup(ds)

    if args.config == "default":
        config_name = "dec"
    else:
        config_name = f"{args.config}_dec"
    url = ds.push_to_hub(args.dataset, config_name=config_name, split=args.split, num_shards=args.num_shards)
    logger.info(f"Decontaminated dataset: {url}")
