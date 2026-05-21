"""
Carbon 3B context-length sweep — last token embedding extraction.

Extracts </dna> (last token) embedding for all sequences in test_new.parquet.
Slicing: seq[-max_length:]  (last window)

Usage:
  python run_carbon_genstyle_ctx_full.py --max_length 8192  --out_dir .../carbon_genstyle_8k_full
  python run_carbon_genstyle_ctx_full.py --max_length 16384 --out_dir .../carbon_genstyle_16k_full
  python run_carbon_genstyle_ctx_full.py --max_length 49152 --out_dir .../carbon_genstyle_48k_full
"""
import argparse, os
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import Counter

HF_TOKEN  = os.environ.get("HF_TOKEN")
MODEL_ID  = "hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1"
DATA_PATH = "/fsx/dana_aubakirova/carbon_ablations/data/eukaryote/test_new.parquet"

SPECIES_MAP = {
    "<fng>": "fungi", "<pln>": "plant", "<inv>": "invertebrate",
    "<prt>": "protozoa", "<vrt>": "vertebrate_other", "<mam>": "vertebrate_mammalian",
}
TAXONOMY_TYPES = list(SPECIES_MAP.values())


def truncate_to_multiple_of_six(seq):
    n = len(seq) // 6 * 6
    return seq[:n] if n > 0 else ""


def compute_metrics(embs, label_ids, k=10):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    normed = embs / np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
    sims = cosine_similarity(normed)
    np.fill_diagonal(sims, -1.0)
    top_k = np.argsort(sims, axis=1)[:, -k:]
    knn_preds = np.array([
        Counter(label_ids[top_k[i]]).most_common(1)[0][0]
        for i in range(len(label_ids))
    ])
    knn_acc = float(np.mean(knn_preds == label_ids))
    km = KMeans(n_clusters=len(TAXONOMY_TYPES), n_init=10, random_state=42)
    km_labels = km.fit_predict(normed)
    ari = float(adjusted_rand_score(label_ids, km_labels))
    nmi = float(normalized_mutual_info_score(label_ids, km_labels))
    return {"knn_acc": knn_acc, "ari": ari, "nmi": nmi}


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load full dataset (no subsampling) ────────────────────────────────────
    print("Loading test_new.parquet (full dataset)...")
    df = pd.read_parquet(DATA_PATH)
    df['type'] = df['species_type'].map(SPECIES_MAP)
    df = df[df['type'].notna()].reset_index(drop=True)
    sequences = df['sequence'].tolist()
    labels    = df['type'].tolist()
    print(f"Total sequences: {len(sequences)}")
    print(Counter(labels))
    df[['record_id', 'type', 'species_type', 'strand', 'start', 'end']].to_parquet(
        os.path.join(args.out_dir, "df.parquet"), index=False)

    label2id  = {t: i for i, t in enumerate(TAXONOMY_TYPES)}
    label_ids = np.array([label2id[l] for l in labels])

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=HF_TOKEN,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    tokenizer.padding_side = "right"

    dna_start_id = tokenizer.convert_tokens_to_ids("<dna>")
    dna_end_id   = tokenizer.convert_tokens_to_ids("</dna>")
    print(f"<dna> id={dna_start_id},  </dna> id={dna_end_id}")

    # ── Extract last token embeddings ─────────────────────────────────────────
    last_embs = []

    print(f"Extracting (N={len(sequences)}, batch_size={args.batch_size}, max_length={args.max_length})...")
    for i in tqdm(range(0, len(sequences), args.batch_size)):
        batch_seqs = sequences[i:i + args.batch_size]
        processed = [f"<dna>{truncate_to_multiple_of_six(s[-args.max_length:])}</dna>"
                     for s in batch_seqs]

        inputs = tokenizer(
            processed, add_special_tokens=False,
            return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)

        hs       = outputs.hidden_states[-1]
        last_idx = inputs["attention_mask"].sum(dim=1) - 1

        for j, idx in enumerate(last_idx):
            last_embs.append(hs[j, idx].float().cpu().numpy())

    last_emb = np.array(last_embs, dtype=np.float32)
    np.save(os.path.join(args.out_dir, "last_token_embeddings.npy"), last_emb)
    print(f"Saved last_token_embeddings.npy: shape={last_emb.shape}  →  {args.out_dir}/")

    del model; torch.cuda.empty_cache()

    # ── Metrics ───────────────────────────────────────────────────────────────
    import json
    m = compute_metrics(last_emb, label_ids)
    print(f"last_token  kNN={m['knn_acc']:.3f}  ARI={m['ari']:.3f}  NMI={m['nmi']:.3f}")
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"max_length": args.max_length, "last_token": m}, f, indent=2)

    print(f"\nDone → {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length",  type=int, required=True)
    parser.add_argument("--out_dir",     type=str, required=True)
    parser.add_argument("--batch_size",  type=int, default=8)
    main(parser.parse_args())
