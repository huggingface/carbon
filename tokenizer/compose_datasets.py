from datasets import load_dataset, concatenate_datasets

# ============================================================================
# CONFIGURATION
# ============================================================================
chunking = 512  # Chunk size for splitting long sequences (None to disable)
# ============================================================================

ds_gene = load_dataset("hf-carbon/raw-data", data_files=["parquet/imgpr_sequences.parquet"], split="train")
ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

ds_gene = ds_gene.rename_column("sequence", "text")

# Apply chunking to gene sequences if specified
if chunking is not None:
    print(f"Chunking gene sequences with chunk size: {chunking}")
    print(f"Gene samples before chunking: {len(ds_gene)}")

    def chunk_text(examples):
        """Split texts into chunks of specified size."""
        chunked_texts = []
        for text in examples["text"]:
            # Split each text into chunks
            for i in range(0, len(text), chunking):
                chunked_texts.append(text[i:i+chunking])
        return {"text": chunked_texts}

    # Apply chunking with batched=True to properly handle the flattening
    ds_gene = ds_gene.map(
        chunk_text,
        batched=True,
        remove_columns=ds_gene.column_names,
        num_proc=16
    )

    print(f"Gene samples after chunking: {len(ds_gene)}")

ds_fw = ds_fw.map(lambda x: {"len":len(x["text"])}, num_proc=16)
ds_gene = ds_gene.map(lambda x: {"len":len(x["text"])}, num_proc=16)

# Verify chunking worked correctly
if chunking is not None:
    gene_lengths = ds_gene['len'][:1_000_000]
    print(f"\nChunking verification:")
    print(f"  Min chunk size: {min(gene_lengths)}")
    print(f"  Max chunk size: {max(gene_lengths)}")
    print(f"  Chunks at target size: {sum(1 for l in gene_lengths if l == chunking)}")
    print(f"  Chunks smaller than target: {sum(1 for l in gene_lengths if l < chunking)}")
    print(f"  Sample of first 5 chunk sizes: {gene_lengths[:5]}")

#ds_fw = ds_fw.remove_columns(["metadata"])
#ds_gene = ds_gene.remove_columns(["metadata"])

chars_fw = sum(ds_fw['len'])
chars_gene = sum(ds_gene['len'])

print(f"Total FW characters: {chars_fw/1e9:.2f} B, Total documents: {len(ds_fw)}")
print(f"Total Gene characters: {chars_gene/1e9:.2f} B, Total documents: {len(ds_gene)}")

target_ratio = 1  # Target ratio of chars_fw / chars_gene
print(f"Current ratio (FW/Gene): {chars_fw/chars_gene:.2f}")

# Determine which dataset to downsample based on current vs target ratio
current_ratio = chars_fw / chars_gene

if current_ratio > target_ratio:
    # FW is too large, downsample it
    sample_ratio = (chars_gene * target_ratio) / chars_fw
    print(f"Downsampling FW by ratio: {sample_ratio:.2f}")
    print(f"FW samples: {int(sample_ratio * len(ds_fw))} (from {len(ds_fw)})")
    ds_fw = ds_fw.shuffle().select(range(int(sample_ratio * len(ds_fw))))
    chars_fw = sum(ds_fw['len'])
    print(f"Resulting FW/Gene ratio: {chars_fw/chars_gene:.2f}")
else:
    # Gene is too large, downsample it
    sample_ratio = chars_fw / (chars_gene * target_ratio)
    print(f"Downsampling Gene by ratio: {sample_ratio:.2f}")
    print(f"Gene samples: {int(sample_ratio * len(ds_gene))} (from {len(ds_gene)})")
    ds_gene = ds_gene.shuffle().select(range(int(sample_ratio * len(ds_gene))))
    chars_gene = sum(ds_gene['len'])
    print(f"Resulting FW/Gene ratio: {chars_fw/chars_gene:.2f}")

ds_tokenizer = concatenate_datasets([ds_gene, ds_fw]).shuffle().select_columns(["text"])
print(ds_tokenizer)
ds_tokenizer.to_parquet(f"/fsx/leandro/data/samples/gene_tokenizer_sample_{chunking}/00000.parquet")