from transformers import Trainer
import torch
import torch.nn.functional as F
import random
import numpy as np

class BPTrainer(Trainer):
    """
    Base-pair (BP) level trainer for k-mer tokenized DNA sequences.

    This trainer implements a marginal base-pair loss for regular k-mer tokens,
    while falling back to standard token-level cross-entropy for special tokens
    (e.g., [CLS], [SEP], [PAD], etc.).

    For each k-mer token, the model predicts a distribution over the vocabulary.
    The loss is computed by marginalizing token probabilities into per-base
    (A/T/C/G) probabilities at each of the k positions, and then applying
    negative log-likelihood at the base-pair level.
    """

    def __init__(self, processing_class=None, bp_loss_only=False, **kwargs):
        # Remove deprecated tokenizer argument to avoid HF warnings
        kwargs.pop("tokenizer", None)
        super().__init__(**kwargs)

        self.dna_tokenizer = processing_class
        self.bp_loss_only = True

        # Cached tensors built once and reused across steps
        self._special_ids = None               # [N_special]
        self._nucleotide_indices = None        # [V, k], mapping token -> nucleotide indices
        self._nucleotide_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    # ---------------------------------------------------------------------
    # Main entry: override HuggingFace Trainer.compute_loss
    # ---------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss.

        The loss consists of:
        1) Marginal base-pair loss for regular k-mer tokens
        2) Token-level cross-entropy loss for special tokens

        Losses are weighted by their respective token counts and normalized.
        """
        labels = inputs.get("labels")                      # [B, S]
        logits = model(**inputs).logits                    # [B, S, V]

        # Standard causal LM shifting
        shift_logits = logits[..., :-1, :].contiguous()    # [B, S-1, V]
        shift_labels = labels[..., 1:].contiguous()        # [B, S-1]

        device = shift_logits.device
        k = self.dna_tokenizer.k

        # ------------------------------------------------------------------
        # 1) Build static cache once (special ids + nucleotide index mapping)
        # ------------------------------------------------------------------
        if self._special_ids is None:
            self._build_static_cache(model, k)

        # Ignore tokens that should not contribute to the loss:
        # - UNK: predicting unknown tokens is meaningless
        # - PAD: padding
        # - -100: HuggingFace ignore index
        ignore_ids = torch.tensor(
            [
                self.dna_tokenizer.unk_token_id,
                self.dna_tokenizer.pad_token_id,
                -100
            ],
            device=device
        )
        ignore_mask = torch.isin(shift_labels, ignore_ids)
        shift_labels = shift_labels.masked_fill(ignore_mask, -100)

        # ------------------------------------------------------------------
        # 2) Construct masks
        # ------------------------------------------------------------------
        valid_mask = shift_labels != -100
        special_mask = torch.isin(shift_labels, self._special_ids) & valid_mask
        regular_mask = valid_mask & (~special_mask)

        # ------------------------------------------------------------------
        # 3) Compute losses
        # ------------------------------------------------------------------
        # Base-pair marginal loss for regular k-mer tokens
        if regular_mask.any():
            bp_loss = self._marginal_bp_loss(
                shift_logits, shift_labels, regular_mask, k, device
            )
        else:
            bp_loss = torch.tensor(0.0, device=device)

        # Token-level cross-entropy for special tokens
        if special_mask.any():
            token_loss = F.cross_entropy(
                shift_logits[special_mask],
                shift_labels[special_mask],
                ignore_index=-100,
                reduction='mean'
            )
            # Normalize to be on the same scale as BP loss
            token_loss = token_loss / k
        else:
            token_loss = torch.tensor(0.0, device=device)

        # ------------------------------------------------------------------
        # 4) Combine losses weighted by token counts
        # ------------------------------------------------------------------
        bp_count = regular_mask.sum()
        special_count = special_mask.sum()
        total_count = bp_count + special_count

        if total_count == 0:
            total_loss = torch.tensor(0.0, device=device)
        else:
            total_loss = (
                bp_loss * bp_count + token_loss * special_count
            ) / total_count

        # Match HuggingFace Trainer's gradient accumulation behavior
        total_loss = total_loss / self.args.gradient_accumulation_steps

        # Optional mode: return BP loss only (useful for diagnostics)
        if self.bp_loss_only:
            return (bp_loss, logits) if return_outputs else bp_loss

        return (total_loss, logits) if return_outputs else total_loss

    # ---------------------------------------------------------------------
    # Static cache construction (executed once)
    # ---------------------------------------------------------------------
    def _build_static_cache(self, model, k):
        """
        Build static tensors used for BP loss computation.

        This includes:
        - special token ids
        - a lookup table mapping each vocabulary token to its
          per-position nucleotide indices
        """
        # If model is wrapped in DDP, unwrap it
        if hasattr(model, 'module'):
            model = model.module

        vocab_size = model.config.vocab_size
        device = model.device

        # Cache ids of all special tokens
        self._special_ids = torch.tensor(
            [self.dna_tokenizer.vocab[e] for e in self.dna_tokenizer.special_tokens],
            dtype=torch.long,
            device=device
        )

        # Build nucleotide index table: [V, k]
        # Each row corresponds to a token, each column to a base position
        indices = torch.zeros(vocab_size, k, dtype=torch.long, device=device)
        for tid in range(vocab_size):
            tok = self.dna_tokenizer.ids_to_tokens[tid]
            if tok in self.dna_tokenizer.special_tokens:
                indices[tid] = 0
            else:
                seq = tok[:k]
                idx = [self._nucleotide_map.get(c, 0) for c in seq]
                indices[tid] = torch.tensor(idx, dtype=torch.long, device=device)

        self._nucleotide_indices = indices

    # ---------------------------------------------------------------------
    # Marginal base-pair loss
    # ---------------------------------------------------------------------
    def _marginal_bp_loss(self, shift_logits, shift_labels, regular_mask, k, device):
        """
        Compute marginal base-pair loss for regular k-mer tokens.

        For each of the k positions in a k-mer:
        1) Convert token-level probabilities into marginal nucleotide probabilities
           over {A, T, C, G}
        2) Apply negative log-likelihood loss against the ground-truth nucleotide
        3) Average the loss over k positions
        """
        token_probs = F.softmax(shift_logits, dim=-1)   # [B, S, V]
        bp_loss = torch.tensor(0.0, device=device)

        for pos in range(k):
            # --------------------------------------------------------------
            # 1) Ground-truth nucleotide indices at this position
            # --------------------------------------------------------------
            target_nt = self._nucleotide_indices[
                shift_labels, pos
            ].masked_fill(~regular_mask, -100)          # [B, S]

            # --------------------------------------------------------------
            # 2) Compute marginal probabilities over 4 nucleotides
            # --------------------------------------------------------------
            marginal_probs = torch.zeros(
                *shift_logits.shape[:2], 4, device=device
            )                                           # [B, S, 4]

            src_indices = self._nucleotide_indices[:, pos]  # [V], values in {0,1,2,3}
            for nt_idx in range(4):
                mask = src_indices == nt_idx
                marginal_probs[..., nt_idx] = token_probs[..., mask].sum(dim=-1)

            # Numerical stability
            marginal_probs = marginal_probs.clamp(min=1e-8)
            log_marginal_probs = marginal_probs.log()

            # --------------------------------------------------------------
            # 3) Negative log-likelihood loss
            # --------------------------------------------------------------
            pos_loss = F.nll_loss(
                log_marginal_probs.view(-1, 4),
                target_nt.view(-1),
                ignore_index=-100,
                reduction='mean'
            )

            bp_loss += pos_loss

        # Average over k positions
        return bp_loss / k
