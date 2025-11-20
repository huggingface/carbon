from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset configuration
DATASET_NAME = "hf-carbon/raw-data"
DATA_FILES = ["parquet/imgpr_sequences.parquet"]

# Tokenizer configuration
TOKENIZER_NAME = "hf-carbon/tokenizer-gene"

# ============================================================================
# END CONFIGURATION
# ============================================================================


def print_statistics(name, lengths):
    """Print statistics for a list of lengths."""
    lengths_array = np.array(lengths)

    print(f"\n{name} Statistics:")
    print(f"  Count: {len(lengths):,}")
    print(f"  Total: {lengths_array.sum():,}")
    print(f"  Mean: {lengths_array.mean():.2f}")
    print(f"  Median: {np.median(lengths_array):.2f}")
    print(f"  Std Dev: {lengths_array.std():.2f}")
    print(f"  Min: {lengths_array.min():,}")
    print(f"  Max: {lengths_array.max():,}")
    print(f"  Percentiles:")
    print(f"    25th: {np.percentile(lengths_array, 25):.2f}")
    print(f"    50th: {np.percentile(lengths_array, 50):.2f}")
    print(f"    75th: {np.percentile(lengths_array, 75):.2f}")
    print(f"    90th: {np.percentile(lengths_array, 90):.2f}")
    print(f"    95th: {np.percentile(lengths_array, 95):.2f}")
    print(f"    99th: {np.percentile(lengths_array, 99):.2f}")


def main():
    # Load dataset
    print(f"Loading dataset from {DATASET_NAME}: {DATA_FILES}")
    dataset = load_dataset(DATASET_NAME, data_files=DATA_FILES, split="train")
    dataset = dataset.rename_column("sequence", "text")

    print(f"Analyzing {len(dataset):,} samples")

    # Load tokenizer
    print(f"Loading tokenizer from {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Calculate text lengths
    print("\nCalculating text lengths...")

    def get_text_length(examples):
        """Get text length."""
        return {"text_length": [len(text) for text in examples["text"]]}

    dataset_with_text_lengths = dataset.map(
        get_text_length,
        batched=True,
        batch_size=10_000,
        desc="Calculating text lengths",
    )

    text_lengths = dataset_with_text_lengths["text_length"]

    # Calculate token lengths - use EXACT same function as pretrain.py
    print("Calculating token lengths (using pretrain.py tokenization function)...")

    def tokenize_function(examples):
        """Tokenize text with EOS token prepended."""
        # Prepend EOS token to each text
        texts_with_eos = [tokenizer.eos_token + text for text in examples["text"]]
        # Tokenize without truncation - packing will handle long sequences
        tokenized = tokenizer(
            texts_with_eos,
            padding=False,
            return_attention_mask=False,
        )
        return {"input_ids": tokenized["input_ids"]}

    dataset_with_tokens = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10_000,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Compute token lengths using map
    dataset_with_token_lengths = dataset_with_tokens.map(
        lambda x: {"token_length": len(x["input_ids"])},
        num_proc=32,
        desc="Computing token lengths",
    )

    token_lengths = dataset_with_token_lengths["token_length"]

    # Print statistics
    print("\n" + "=" * 60)
    print_statistics("Text Length (characters)", text_lengths)
    print("\n" + "=" * 60)
    print_statistics("Token Length (with EOS prepended)", token_lengths)
    print("\n" + "=" * 60)

    # Calculate compression ratio
    total_chars = sum(text_lengths)
    total_tokens = sum(token_lengths)
    avg_text_len = np.mean(text_lengths)
    avg_token_len = np.mean(token_lengths)
    compression_ratio = avg_text_len / avg_token_len

    print(f"\nCompression Ratio:")
    print(f"  Characters per token: {compression_ratio:.2f}")
    print(f"  Tokens per character: {1/compression_ratio:.4f}")

    print(f"\nTotal Dataset Size:")
    print(f"  Total characters: {total_chars:,} ({total_chars/1e9:.2f}B)")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")


if __name__ == "__main__":
    main()
