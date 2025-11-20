from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tokenizers import (
    pre_tokenizers,
    decoders,
    Tokenizer,
)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# ============================================================================
# CONFIGURATION: Hyperparameters and Design Decisions
# ============================================================================

# Dataset configuration
DATA_FILE = "/fsx/leandro/data/samples/gene_tokenizer_sample_4096/00000.parquet"
NUM_SAMPLES = 1_000_000 # Number of samples to use for training (None for all)

# Tokenizer configuration
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = [
    "<|endoftext|>",
]
MAX_TOKEN_LENGTH=128

# Pre-tokenizer configuration
DIGITS_INDIVIDUAL = True  # Split digits into individual tokens
BYTELEVEL_ADD_PREFIX_SPACE = False
BYTELEVEL_USE_REGEX = True

# Training configuration
BATCH_SIZE = 1_000  # Note: Must divide the number of rows evenly
SHOW_PROGRESS = True

# Output configuration
HF_REPO_NAME = "hf-carbon/tokenizer-gene"

# ============================================================================
# END CONFIGURATION
# ============================================================================


def batch_iterator(dataset, batch_size=BATCH_SIZE):
    """Iterate over dataset in batches."""
    for i in range(0, len(dataset), batch_size):
        yield dataset.select(range(i, i + batch_size))["text"]


# Load and prepare dataset
ds = load_dataset("parquet", data_files=DATA_FILE, split="train")
if NUM_SAMPLES is not None:
    ds = ds.select(range(NUM_SAMPLES))

print(f"Training tokenizer with vocab size: {VOCAB_SIZE}")
print(f"Using {len(ds)} samples from dataset")

# Configure pre-tokenizers
digits_pretokenizer = pre_tokenizers.Digits(individual_digits=DIGITS_INDIVIDUAL)
bytelevel_pretokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=BYTELEVEL_ADD_PREFIX_SPACE,
    use_regex=BYTELEVEL_USE_REGEX
)

# Configure decoder
bytelevel_decoder = decoders.ByteLevel(
    add_prefix_space=BYTELEVEL_ADD_PREFIX_SPACE,
    use_regex=BYTELEVEL_USE_REGEX
)

# Initialize tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([digits_pretokenizer, bytelevel_pretokenizer])
tokenizer.decoder = bytelevel_decoder

# Train tokenizer
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    show_progress=SHOW_PROGRESS,
    special_tokens=SPECIAL_TOKENS,
    max_token_length=MAX_TOKEN_LENGTH
)
tokenizer.train_from_iterator(batch_iterator(ds), trainer=trainer)

# Wrap tokenizer for HuggingFace compatibility
tokenizer_wrapper = GPT2TokenizerFast(
    tokenizer_object=tokenizer,
    vocab_size=VOCAB_SIZE,
    additional_special_tokens=SPECIAL_TOKENS,
    bos_token=SPECIAL_TOKENS[0],
    eos_token=SPECIAL_TOKENS[0],
    unk_token=SPECIAL_TOKENS[0]
)

# Save to HuggingFace Hub
tokenizer_wrapper.push_to_hub(HF_REPO_NAME)