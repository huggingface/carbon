from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
from trl import SFTConfig, SFTTrainer
from accelerate import PartialState

# ============================================================================
# CONFIGURATION: Hyperparameters and Design Decisions
# ============================================================================

# Dataset configuration
NUM_SAMPLES = None  # Number of samples to use for training (None for all)
MAX_SEQ_LENGTH = 2048  # Maximum sequence length for training
PRETOKENIZE = True  # Pretokenize dataset and save input_ids
NUM_PROC = 16  # Number of processes for pretokenization

# Tokenizer configuration
TOKENIZER_NAME = "hf-carbon/tokenizer-gene"

# Model configuration
VOCAB_SIZE = 32_000
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 12
NUM_ATTENTION_HEADS = 12
INTERMEDIATE_SIZE = 3072
MAX_POSITION_EMBEDDINGS = MAX_SEQ_LENGTH

# Training configuration
OUTPUT_DIR = "/fsx/leandro/models/gene-pretrain-10"
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 10
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 3
BF16 = True  # Use bfloat16 mixed precision training
GRADIENT_CHECKPOINTING = False  # Enable gradient checkpointing to save memory
ATTN_IMPLEMENTATION = "kernels-community/flash-attn3"  # Use Flash Attention 3 via community kernels

# Packing configuration
PACKING = True
PACKING_STRATEGY = "wrapped"  # Use wrapped packing strategy

# DataLoader configuration
DATALOADER_NUM_WORKERS = 4
DATALOADER_PIN_MEMORY = True

# HuggingFace Hub configuration
PUSH_TO_HUB = False
HF_REPO_NAME = "hf-carbon/gene-model"

# ============================================================================
# END CONFIGURATION
# ============================================================================


def main():
    # Load tokenizer
    print(f"Loading tokenizer from {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    print(f"Loading dataset from hf-carbon/raw-data: parquet/imgpr_sequences.parquet")
    dataset = load_dataset("hf-carbon/raw-data", data_files=["parquet/imgpr_sequences.parquet"], split="train")
    dataset = dataset.rename_column("sequence", "text")

    if NUM_SAMPLES is not None:
        dataset = dataset.select(range(NUM_SAMPLES))

    print(f"Dataset size: {len(dataset)} samples")

    # Pretokenize dataset if enabled
    if PRETOKENIZE:
        state = PartialState()

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

        # Tokenize only on main process
        if state.is_main_process:
            print("Pretokenizing dataset with EOS token prepended (main process)...")
            dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=10_000,
                remove_columns=dataset.column_names,
                desc="Tokenizing dataset",
            )
            print(f"Dataset pretokenized.")

        # Wait for main process to finish tokenization
        state.wait_for_everyone()

        # Load cached dataset on all workers
        if not state.is_main_process:
            print(f"Loading cached tokenized dataset (worker {state.process_index})...")
            dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=10_000,
                remove_columns=dataset.column_names,
                desc="Loading cached dataset",
            )

    # Create model configuration
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=VOCAB_SIZE,
        n_positions=MAX_POSITION_EMBEDDINGS,
        n_embd=HIDDEN_SIZE,
        n_layer=NUM_HIDDEN_LAYERS,
        n_head=NUM_ATTENTION_HEADS,
        n_inner=INTERMEDIATE_SIZE,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    # Initialize model
    print("Initializing model from scratch with Flash Attention 3")
    model = GPT2LMHeadModel(config)

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text" if not PRETOKENIZE else None,
        packing=PACKING,
        packing_strategy=PACKING_STRATEGY,
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HF_REPO_NAME if PUSH_TO_HUB else None,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if PUSH_TO_HUB:
        print(f"Pushing model to HuggingFace Hub: {HF_REPO_NAME}")
        trainer.push_to_hub()

    print("Training completed!")


if __name__ == "__main__":
    main()
