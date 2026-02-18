#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight check for hybrid BP token_mask semantics.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to a saved HybridTokenizer directory (must contain dna_config.json).",
    )
    parser.add_argument("--k", type=int, default=6, help="k-mer size expected by training config.")
    parser.add_argument(
        "--text",
        type=str,
        default="prefix <dna>TTT</dna> suffix",
        help="Probe text used to validate tail token_mask behavior.",
    )
    parser.add_argument(
        "--full_tokenizer_check",
        action="store_true",
        help="Run full HybridTokenizer encode check (imports transformers; slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_dir = Path(args.tokenizer_path).expanduser().resolve()
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_dir}")
    dna_cfg_path = tokenizer_dir / "dna_config.json"
    if not dna_cfg_path.exists():
        raise FileNotFoundError(f"Expected dna_config.json in tokenizer path: {tokenizer_dir}")

    dna_cfg = json.loads(dna_cfg_path.read_text(encoding="utf-8"))
    k = int(dna_cfg["k"])
    dna_start_id = int(dna_cfg["dna_start_id"])
    dna_vocab_size = int(dna_cfg["dna_vocab_size"])
    num_special = len(dna_cfg["dna_special_tokens"])
    dna_kmer_start_id = dna_start_id + num_special
    dna_kmer_end_id = dna_start_id + dna_vocab_size

    if k != args.k:
        raise ValueError(f"k mismatch: dna_config.k={k}, expected --k={args.k}")

    if args.full_tokenizer_check:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from tokenizer.hybrid_tokenizer import HybridTokenizer  # pylint: disable=import-outside-toplevel

        tokenizer = HybridTokenizer.from_pretrained(str(tokenizer_dir))
        encoded = tokenizer(
            args.text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_mask=True,
            max_length=None,
        )
        input_ids = encoded["input_ids"]
        token_mask = encoded["token_mask"]

        if tuple(input_ids.shape) != tuple(token_mask.shape):
            raise ValueError(f"Shape mismatch: input_ids={tuple(input_ids.shape)} token_mask={tuple(token_mask.shape)}")

        label_ids = input_ids[:, 1:]
        label_token_mask = token_mask[:, 1:]
        if tuple(label_ids.shape) != tuple(label_token_mask.shape):
            raise ValueError(
                f"Shifted shape mismatch: label_ids={tuple(label_ids.shape)} token_mask={tuple(label_token_mask.shape)}"
            )

        valid_values = set(range(-2, args.k + 1))
        flat_values = set(int(v) for v in label_token_mask[0].tolist())
        bad_values = sorted(v for v in flat_values if v not in valid_values)
        if bad_values:
            raise ValueError(f"Invalid token_mask values: {bad_values}. Expected values in [-2..{args.k}].")

        tail_valid_len = len("TTT")
        if tail_valid_len not in flat_values:
            raise ValueError(
                f"Expected tail valid_len={tail_valid_len} from '<dna>TTT</dna>' not found in token_mask values: {sorted(flat_values)}"
            )

    print("Hybrid BP token_mask preflight: OK")
    print(f"tokenizer.k={k}")
    print(f"dna_kmer_start_id={dna_kmer_start_id}")
    print(f"dna_kmer_end_id={dna_kmer_end_id}")
    print("Suggested exports:")
    print(f"  export HYBRID_BP_DNA_KMER_START_ID={dna_kmer_start_id}")
    print(f"  export HYBRID_BP_DNA_KMER_END_ID={dna_kmer_end_id}")
    print(f"  export HYBRID_BP_K={k}")
    print("  export ENABLE_HYBRID_BP=1")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
