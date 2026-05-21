from transformers import Trainer
import torch
import torch.nn.functional as F


class FNSTrainer(Trainer):
    """Base-pair level trainer for HybridDNATokenizer (BPE + DNA k-mers)."""
    def __init__(self, tokenizer=None, dna_loss_only=False, **kwargs):
        super().__init__(processing_class=tokenizer, **kwargs)
        self.tokenizer = tokenizer
        self.dna_loss_only = dna_loss_only
        # Class-level cache: build once
        self._dna_special_ids = None           # DNA special tokens: <dna>, </dna>, <oov>
        self._nucleotide_indices = None        # [V, k]  long - only for DNA k-mer tokens
        self._nucleotide_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self._dna_kmer_mask = None             # [V] bool - True for DNA k-mer tokens

    # ------------------ Entry point ------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")                      # [B, S]
        logits = model(**inputs).logits                    # [B, S, V]
        shift_logits = logits[..., :-1, :].contiguous()    # [B, S, V]
        shift_labels = labels[..., 1:].contiguous()        # [B, S]
        device = shift_logits.device
        k = self.tokenizer.k

        # 1. Build cache once
        if self._dna_special_ids is None:
            self._build_static_cache(model, k)

        # 2. Ignore padding and special ignore tokens
        ignore_ids = torch.tensor([self.tokenizer.oov_token_id,
                                   self.tokenizer.pad_token_id,
                                   -100], device=device)
        ignore_mask = torch.isin(shift_labels, ignore_ids)
        shift_labels = shift_labels.masked_fill(ignore_mask, -100)

        # 3. Classify tokens into three categories:
        #    - DNA k-mer tokens: use BP-level loss
        #    - DNA special tokens (<dna>, </dna>, <oov>): use token-level loss
        #    - BPE tokens: use token-level loss
        valid_mask = shift_labels != -100

        # DNA k-mer tokens (use BP loss)
        dna_kmer_mask = self._dna_kmer_mask[shift_labels] & valid_mask

        # DNA special tokens + BPE tokens (use token loss)
        token_level_mask = valid_mask & (~dna_kmer_mask)

        # 4. Loss computation
        if dna_kmer_mask.any():
            bp_loss = self._marginal_bp_loss(shift_logits, shift_labels, dna_kmer_mask, k, device)
        else:
            bp_loss = torch.tensor(0.0, device=device)

        if token_level_mask.any():
            token_loss = F.cross_entropy(
                shift_logits[token_level_mask],
                shift_labels[token_level_mask],
                ignore_index=-100,
                reduction='mean'
            )
            # Normalize token loss to be comparable with BP loss
            token_loss = token_loss / k
        else:
            token_loss = torch.tensor(0.0, device=device)

        # 5. Weighted combine
        bp_count = dna_kmer_mask.sum()
        token_count = token_level_mask.sum()
        total = bp_count + token_count
        if total == 0:
            total_loss = torch.tensor(0.0, device=device)
        else:
            total_loss = (bp_loss * bp_count + token_loss * token_count) / total

        total_loss = total_loss / self.args.gradient_accumulation_steps

        if self.dna_loss_only:
            return (bp_loss, logits) if return_outputs else bp_loss

        return (total_loss, logits) if return_outputs else total_loss

    # ------------------ Static cache (built once) ------------------
    def _build_static_cache(self, model, k):
        # If model is wrapped by DDP, get the underlying model
        if hasattr(model, 'module'):
            model = model.module

        vocab_size = model.config.vocab_size
        device = model.device

        # 1. DNA special token IDs: <dna>, </dna>, <oov>
        self._dna_special_ids = torch.tensor(
            [self.tokenizer.dna_begin_token_id,
             self.tokenizer.dna_end_token_id,
             self.tokenizer.oov_token_id],
            dtype=torch.long, device=device
        )

        # 2. Build DNA k-mer mask [V] - True for DNA k-mer tokens
        dna_kmer_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for kmer in self.tokenizer.kmers:
            tid = self.tokenizer.dna_token_to_id[kmer]
            if tid < vocab_size:
                dna_kmer_mask[tid] = True
        self._dna_kmer_mask = dna_kmer_mask

        # 3. Nucleotide indices [V, k] - only meaningful for DNA k-mer tokens
        indices = torch.zeros(vocab_size, k, dtype=torch.long, device=device)
        for tid in range(vocab_size):
            # Check if this is a DNA k-mer token
            if tid in self.tokenizer.dna_id_to_token:
                tok = self.tokenizer.dna_id_to_token[tid]
                # Only process actual k-mers, not special tokens
                if tok in self.tokenizer.kmers:
                    seq = tok[:k]
                    idx = [self._nucleotide_map.get(c, 0) for c in seq]
                    indices[tid] = torch.tensor(idx, dtype=torch.long, device=device)
                else:
                    # DNA special tokens or padding tokens - set to 0
                    indices[tid] = 0
            else:
                # BPE tokens - set to 0 (won't be used in BP loss anyway)
                indices[tid] = 0
        self._nucleotide_indices = indices

    # ------------------ Marginal BP loss (no Python loops) ------------------
    def _marginal_bp_loss(self, shift_logits, shift_labels, regular_mask, k, device):
        token_probs = F.softmax(shift_logits, dim=-1)                       # [B, S, V]
        bp_loss = torch.tensor(0.0, device=device)

        for pos in range(k):
            # 1. True nucleotide indices at the current position [B, S]
            target_nt = self._nucleotide_indices[shift_labels, pos].masked_fill(~regular_mask, -100)
            # 2. Build 4-class probabilities [B, S, 4]
            marginal_probs = torch.zeros(*shift_logits.shape[:2], 4, device=device)
            # 3. Scatter-add once: sum token_probs by mask
            src_indices = self._nucleotide_indices[:, pos]          # [V]  0~3
            for nt_idx in range(4):
                mask = src_indices == nt_idx                        # [V]
                marginal_probs[..., nt_idx] = token_probs[..., mask].sum(dim=-1)

            marginal_probs = marginal_probs.clamp(min=1e-8)
            log_marginal_probs = marginal_probs.log()
            # 4. NLL loss in one call
            pos_loss = F.nll_loss(
                log_marginal_probs.view(-1, 4),
                target_nt.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            bp_loss += pos_loss

        return bp_loss / k