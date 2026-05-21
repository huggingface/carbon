"""
Carbon 3B — content embedding extraction on test_new.parquet — middle 16K window.

Tokenization: [<dna>] [6mer_1] ... [6mer_N] [</dna>]

  content : hidden state at last DNA 6-mer (token before </dna>)

Slicing: seq[96000 − 16384 : 96000]  (middle 16K window, centered on CDS)
"""
import os
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

HF_TOKEN    = os.environ.get("HF_TOKEN")
MODEL_ID    = "hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1"
DATA_PATH   = "/fsx/dana_aubakirova/carbon_ablations/data/eukaryote/test_new.parquet"
OUT_DIR     = "/fsx/dana_aubakirova/carbon_ablations/clustering/output/carbon_test_new_3emb_v2_16k"
MAX_LENGTH  = 16384
BATCH_SIZE  = 4        # reduced from 8 — longer sequences need more GPU memory
SEED        = 42
SLICE_START = 96000 - MAX_LENGTH  # 79616

os.makedirs(OUT_DIR, exist_ok=True)

SPECIES_MAP = {
    "<fng>": "fungi", "<pln>": "plant", "<inv>": "invertebrate",
    "<prt>": "protozoa", "<vrt>": "vertebrate_other", "<mam>": "vertebrate_mammalian",
}
SPECIES_LIST = list(SPECIES_MAP.values())


def truncate_to_multiple_of_six(seq):
    n = len(seq) // 6 * 6
    return seq[:n] if n > 0 else ""


# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading test_new.parquet...")
df = pd.read_parquet(DATA_PATH)
df['seq_sliced']   = df['sequence'].str[SLICE_START:96000].apply(truncate_to_multiple_of_six)
df['species_name'] = df['species_type'].map(SPECIES_MAP)
df['codon_phase']  = (df['start'] - SLICE_START) % 3
df['strand_label'] = df['strand'].map({'<+>': 'forward (+)', '<->': 'reverse (-)'})
df = df[df['seq_sliced'].str.len() >= 6].reset_index(drop=True)
print(f"Records: {len(df)}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_ID}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, token=HF_TOKEN,
    torch_dtype=torch.bfloat16, trust_remote_code=True,
).to(device)
model.eval()
tokenizer.padding_side    = "right"
tokenizer.truncation_side = "left"

# <dna> + N 6-mer tokens + </dna>  →  N+2 tokens max
# 16380 bp / 6 = 2730 DNA tokens + 2 wrapper = 2732
max_tokens = ((MAX_LENGTH + 2 + 5) // 6) * 6

sequences = df['seq_sliced'].tolist()

# ── Extract content embeddings ────────────────────────────────────────────────
content_embs = []

print(f"Extracting content embeddings (N={len(sequences)}, batch_size={BATCH_SIZE}, max_tokens={max_tokens})...")
for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
    batch_seqs = sequences[i:i + BATCH_SIZE]
    batch = [f"<dna>{s}</dna>" for s in batch_seqs]

    enc = tokenizer(batch, add_special_tokens=False,
                    return_tensors="pt", padding=True,
                    truncation=True, max_length=max_tokens)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)

    hs      = out.hidden_states[-1]       # (B, T, D)
    mask    = enc["attention_mask"]        # (B, T)
    idx_sep = mask.sum(dim=1) - 1         # position of </dna>

    for j in range(hs.size(0)):
        content_pos = idx_sep[j].item() - 1   # last DNA 6-mer (token before </dna>)
        content_embs.append(hs[j, content_pos].float().cpu().numpy())

content_emb = np.array(content_embs, dtype=np.float32)
np.save(os.path.join(OUT_DIR, "content_embeddings.npy"), content_emb)
print(f"Saved content_embeddings.npy: shape={content_emb.shape}  →  {OUT_DIR}/")

del model; torch.cuda.empty_cache()
print(f"\nDone → {OUT_DIR}/")
