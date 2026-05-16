"""Compatibility helpers for HF DNA models used by the eval scripts."""


_DNA_BASES = "ATCG"
_DNA_BASE_TO_IDX = {base: idx for idx, base in enumerate(_DNA_BASES)}


def patch_legacy_tokenizer_base():
    """Restore tokenizer attributes expected by GENERator's tokenizer code."""
    from transformers import PreTrainedTokenizerBase

    PreTrainedTokenizerBase._special_tokens_map = {}
    PreTrainedTokenizerBase._added_tokens_decoder = {}
    PreTrainedTokenizerBase._added_tokens_encoder = {}
    PreTrainedTokenizerBase.verbose = False


def patch_generator_sample(model):
    """Adapt GENERator's custom `_sample` signature to current `generate()`."""
    cls = model.__class__
    if cls.__name__ != "GENERatorForCausalLM" or getattr(cls, "_carbon_sample_patched", False):
        return

    original_sample = cls._sample

    def _sample(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus=False,
        streamer=None,
        **model_kwargs,
    ):
        return original_sample(
            self,
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus,
            streamer,
            **model_kwargs,
        )

    cls._sample = _sample
    cls._carbon_sample_patched = True


def score_dna_sequence_fallback(model, tokenizer, sequences):
    """Score DNA sequence(s) when an HF model lacks `score_sequence`.

    Carbon's public HF model is a plain `LlamaForCausalLM`, but its tokenizer
    exposes a 6-mer DNA block. This mirrors the Carbon remote-code scorer:
    prepend `<dna>`, run a forward pass, marginalize 6-mer logits to per-base
    probabilities, and return the actual-base probability at each position.
    """
    import torch
    import torch.nn.functional as F

    is_single = isinstance(sequences, str)
    if is_single:
        sequences = [sequences]

    state = _get_dna_scoring_state(model, tokenizer)
    k = state["k"]
    device = state["device"]

    original_lens = [len(seq) for seq in sequences]
    padded = []
    for seq in sequences:
        r = len(seq) % k
        padded.append(seq + "A" * (k - r) if r else seq)

    tagged = ["<dna>" + seq for seq in padded]
    inputs = tokenizer(tagged, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    logits = model(input_ids, attention_mask=attention_mask, return_dict=True).logits
    kmer_ids = state["kmer_ids"]
    base_masks = state["base_masks"]
    prefix_len = state["prefix_len"]

    bp_results = []
    actual_results = []
    for row_idx, (seq, orig_len, padded_seq) in enumerate(zip(sequences, original_lens, padded)):
        num_tokens = len(padded_seq) // k
        token_logits = logits[row_idx, prefix_len - 1 : prefix_len - 1 + num_tokens, :]
        kmer_probs = F.softmax(token_logits[:, kmer_ids].float(), dim=-1)

        bp_probs = torch.zeros(num_tokens, k, len(_DNA_BASES), device=device, dtype=kmer_probs.dtype)
        for pos in range(k):
            for base_idx, mask in enumerate(base_masks[pos]):
                bp_probs[:, pos, base_idx] = kmer_probs[:, mask].sum(dim=-1)
        bp_probs = bp_probs.reshape(-1, len(_DNA_BASES))[:orig_len]

        actual = torch.zeros(orig_len, device=device, dtype=bp_probs.dtype)
        for pos, base in enumerate(seq.upper()):
            base_idx = _DNA_BASE_TO_IDX.get(base)
            actual[pos] = bp_probs[pos].max() if base_idx is None else bp_probs[pos, base_idx]

        bp_results.append(bp_probs)
        actual_results.append(actual)

    if is_single:
        return bp_results[0], actual_results[0]
    return bp_results, actual_results


def _get_dna_scoring_state(model, tokenizer):
    import torch

    state = getattr(model, "_carbon_dna_scoring_state", None)
    device = next(model.parameters()).device
    if state is not None and state["device"] == device:
        return state

    vocab = tokenizer.get_vocab()
    k = int(getattr(tokenizer, "k", 6))
    kmer_items = [
        (token, token_id)
        for token, token_id in vocab.items()
        if isinstance(token, str)
        and len(token) == k
        and all(base in _DNA_BASE_TO_IDX for base in token)
    ]
    expected = len(_DNA_BASES) ** k
    if len(kmer_items) != expected:
        raise RuntimeError(f"Expected {expected} DNA {k}-mer tokens, found {len(kmer_items)}")

    kmer_items.sort(key=lambda item: item[1])
    kmers = [token for token, _ in kmer_items]
    kmer_ids = torch.tensor([token_id for _, token_id in kmer_items], dtype=torch.long, device=device)

    base_masks = []
    for pos in range(k):
        pos_masks = []
        for base in _DNA_BASES:
            pos_masks.append(torch.tensor([kmer[pos] == base for kmer in kmers], dtype=torch.bool, device=device))
        base_masks.append(tuple(pos_masks))

    prefix_ids = tokenizer("<dna>", add_special_tokens=False).get("input_ids", [])
    if not prefix_ids:
        raise RuntimeError("Tokenizer cannot encode <dna>; DNA scoring fallback is unavailable")

    state = {
        "k": k,
        "device": device,
        "kmer_ids": kmer_ids,
        "base_masks": tuple(base_masks),
        "prefix_len": len(prefix_ids),
    }
    model._carbon_dna_scoring_state = state
    return state
