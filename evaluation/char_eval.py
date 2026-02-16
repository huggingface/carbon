import gzip
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from tqdm import tqdm

model_path = "/fsx/leandro/models/gene-pretrain-10"
model_path = "lvwerra/PlasmidGPT"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
device = model.device
print(f"Loaded model {model_path} on device {model.device}.")

# Download file
file_path = hf_hub_download(
    repo_id="arcinstitute/opengenome2",
    filename="json/pretraining_or_both_phases/gtdb_v220_imgpr/data_gtdb_test_chunk1.jsonl.gz",
    repo_type="dataset",
)

# Split sequences into 1024-character chunks
chunk_size = 1024
char_offset = 32
batch_size = 64

chunks = []
with gzip.open(file_path, "rt") as f:
    for line in f:
        seq = json.loads(line)["text"]
        for start in range(0, len(seq), chunk_size):
            chunk = seq[start : start + chunk_size]
            if len(chunk) == chunk_size:  # Only keep full chunks
                chunks.append(chunk)

print(f"Evaluating on {len(chunks)} chunks of {chunk_size} characters.")

token_correct = 0
token_total = 0
char_correct = 0
char_total = 0
total_batches = 0

for i in tqdm(range(0, len(chunks), batch_size)):
    batch_texts = chunks[i : i + batch_size]
    total_batches += 1

    # 1. Tokenize Inputs with Offsets
    tokens = tokenizer(
        batch_texts, return_tensors="pt", padding=True, return_offsets_mapping=True
    ).to(device)

    offset_mapping = tokens.offset_mapping.cpu().numpy()

    with torch.no_grad():
        logits = model(
            input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
        ).logits

    pred_tokens = logits.argmax(dim=-1)

    for j, chunk_text in enumerate(batch_texts):
        # Get actual length (excluding padding)
        valid_seq_len = tokens.attention_mask[j].sum().item()

        # --- YOUR OPTIMIZED DECODING HERE ---
        # We take the predictions for this specific sequence: pred_tokens[j]
        # We reshape it to (Seq_Len, 1) so batch_decode treats every token as a separate sequence
        # This is equivalent to your list comprehension [[tid] for tid in ...] but faster
        current_seq_preds = pred_tokens[
            j, : valid_seq_len - 1
        ]  # Only decode what we need
        decoded_token_list = tokenizer.batch_decode(
            current_seq_preds.reshape(-1, 1), clean_up_tokenization_spaces=False
        )
        # ------------------------------------

        # Iterate tokens
        for t_idx in range(valid_seq_len - 1):
            true_token_id = tokens.input_ids[j, t_idx + 1]
            pred_token_id = pred_tokens[j, t_idx]

            # --- Token Accuracy ---
            tgt_start, tgt_end = offset_mapping[j][t_idx + 1]

            # Skip special tokens or tokens strictly before offset
            if tgt_start == tgt_end or tgt_end <= char_offset:
                continue

            token_total += 1
            if true_token_id == pred_token_id:
                token_correct += 1

            # --- Character Accuracy ---
            valid_start = max(tgt_start, char_offset)
            valid_end = min(tgt_end, chunk_size)
            overlap_len = valid_end - valid_start

            if overlap_len > 0:
                char_total += overlap_len

                # 1. Get Ground Truth Text (Slice raw input string)
                true_segment = chunk_text[valid_start:valid_end]

                # 2. Get Predicted Text (From our local list)
                # t_idx aligns perfectly with decoded_token_list because we sliced
                # pred_tokens with [:valid_seq_len-1] before decoding
                pred_token_text = decoded_token_list[t_idx]

                # 3. Align Prediction to the Slice
                slice_offset = valid_start - tgt_start

                if len(pred_token_text) > slice_offset:
                    pred_segment = pred_token_text[slice_offset:]
                    pred_segment = pred_segment[:overlap_len]
                else:
                    pred_segment = ""

                # 4. Compare
                char_correct += sum(
                    1 for p, t in zip(pred_segment, true_segment) if p == t
                )

print(f"\n{'=' * 60}")
print("Statistics:")
print(f"  Total batches: {total_batches}")
print(f"  Total chunks: {len(chunks)}")
print(f"  Token correct: {token_correct:,} / {token_total:,}")
print(f"  Char correct: {char_correct:,} / {char_total:,}")

# Sanity check
expected_char_total = len(chunks) * (chunk_size - char_offset)
print("\n  Sanity Check:")
print(f"    Expected char_total: ~{expected_char_total:,}")
print(f"    Actual char_total: {char_total:,}")

print(f"{'=' * 60}")
if token_total > 0:
    print(f"Token Accuracy: {token_correct / token_total:.4f}")
if char_total > 0:
    print(f"Character Accuracy: {char_correct / char_total:.4f}")
