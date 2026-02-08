import torch
from torch import nn
from typing import Dict, Tuple

from nanotron import distributed as dist
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy

def _to_bool_mask(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.bool else (x > 0)


class HybridLoss(nn.Module):
    """
    Hybrid loss (no z-loss).

    token_mask semantics (MUST be shifted exactly like label_ids):
      - -2 : padding (produced by tokenizer padding)
      - -1 : natural language token (token-level CE)
      -  0 : DNA special/ignored (<dna>, </dna>, <oov>, etc.; no loss)
      -  1..k : DNA k-mer token (BP marginalization), value = valid_len
               Tail A-padding is excluded by valid_len.

    Objective:
      - NL positions: token-level cross entropy (nanotron sharded_cross_entropy)
      - DNA positions: base-level NLL via probability marginalization over A/T/C/G
                       and counted in "base units" (sum(valid_len))
    """

    def __init__(
        self,
        tp_pg: dist.ProcessGroup,
        k: int,
        dna_start_id: int,
        dna_vocab_size: int,
        dna_special_tokens: list,
        dna_id_to_token: Dict[int, str],
        nl_weight: float = 1.0,
        bp_weight: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.tp_pg = tp_pg
        self.k = int(k)
        self.eps = float(eps)

        self.nl_weight = float(nl_weight)
        self.bp_weight = float(bp_weight)

        # DNA vocab layout provided by HybridTokenizer
        self.dna_start_id = int(dna_start_id)
        self.dna_vocab_size = int(dna_vocab_size)
        self.num_dna_special_tokens = len(dna_special_tokens)

        # k-mer id range: [dna_kmer_start_id, dna_kmer_end_id)
        self.dna_kmer_start_id = self.dna_start_id + self.num_dna_special_tokens
        self.dna_kmer_end_id = self.dna_start_id + self.dna_vocab_size
        self.num_dna_kmers = self.dna_kmer_end_id - self.dna_kmer_start_id

        # Prebuild: kmer_idx -> [k] nucleotide indices (A=0,T=1,C=2,G=3)
        nuc_map = {"A": 0, "T": 1, "C": 2, "G": 3}
        nt_table = torch.full((self.num_dna_kmers, self.k), -1, dtype=torch.long)

        for i in range(self.num_dna_kmers):
            tid = self.dna_kmer_start_id + i
            tok = dna_id_to_token.get(tid, None)
            if tok is None:
                continue
            tok = tok.strip()[: self.k]
            if len(tok) < self.k:
                continue
            idx = [nuc_map.get(c, -1) for c in tok]
            if all(v >= 0 for v in idx):
                nt_table[i] = torch.tensor(idx, dtype=torch.long)

        # Buffer moves with module; not persisted in checkpoints
        self.register_buffer("_dna_nt_table", nt_table, persistent=False)

        # Local cache for TP shard: [V_local, k] in {-1,0,1,2,3}
        self._local_nt_indices = None
        self._cached_device = None
        self._cached_vlocal = None

    # ---------------------------
    # TP helpers
    # ---------------------------
    def _tp_size(self) -> int:
        return int(self.tp_pg.size()) if hasattr(self.tp_pg, "size") else dist.get_world_size(self.tp_pg)

    def _tp_rank(self) -> int:
        return int(self.tp_pg.rank()) if hasattr(self.tp_pg, "rank") else dist.get_rank(self.tp_pg)

    def _maybe_build_local_cache(self, device: torch.device, v_local: int) -> None:
        """
        Build local vocab -> nucleotide lookup:
          self._local_nt_indices[local_vid, pos] in {-1,0,1,2,3}
        -1 means this vocab entry is NOT a DNA k-mer token at this pos.
        """
        if (
            self._local_nt_indices is not None
            and self._cached_device == device
            and self._cached_vlocal == v_local
        ):
            return

        tp_size = self._tp_size()
        tp_rank = self._tp_rank()

        # Robust vocab_start even if shards are uneven:
        size_t = torch.tensor([v_local], device=device, dtype=torch.long)
        gathered = [torch.empty_like(size_t) for _ in range(tp_size)]
        dist.all_gather(gathered, size_t, group=self.tp_pg)
        sizes = [int(x.item()) for x in gathered]
        vocab_start = sum(sizes[:tp_rank])

        global_ids = torch.arange(vocab_start, vocab_start + v_local, device=device, dtype=torch.long)

        nt = torch.full((v_local, self.k), -1, device=device, dtype=torch.long)

        is_kmer = (global_ids >= self.dna_kmer_start_id) & (global_ids < self.dna_kmer_end_id)
        if is_kmer.any():
            kmer_offset = (global_ids[is_kmer] - self.dna_kmer_start_id).to(torch.long)
            nt[is_kmer] = self._dna_nt_table[kmer_offset].to(device=device)

        self._local_nt_indices = nt
        self._cached_device = device
        self._cached_vlocal = v_local

    # ---------------------------
    # Strict alignment checks
    # ---------------------------
    def _assert_alignment(self, label_ids: torch.Tensor, label_mask: torch.Tensor, token_mask: torch.Tensor) -> None:
        """
        Fail fast if token_mask / label_ids are misaligned (e.g., shift mismatch).
        Only checks relative consistency among (label_ids, label_mask, token_mask).
        """
        if label_ids.shape != token_mask.shape or label_ids.shape != label_mask.shape:
            raise RuntimeError(
                f"[HybridLoss] shape mismatch: label_ids{tuple(label_ids.shape)}, "
                f"label_mask{tuple(label_mask.shape)}, token_mask{tuple(token_mask.shape)}"
            )

        if label_ids.dtype not in (torch.int64, torch.int32):
            raise RuntimeError(f"[HybridLoss] label_ids must be int dtype, got {label_ids.dtype}")

        if token_mask.dtype.is_floating_point:
            raise RuntimeError(f"[HybridLoss] token_mask must be integer dtype, got {token_mask.dtype}")

        lm = _to_bool_mask(label_mask)
        tm = token_mask.to(torch.long)
        lid = label_ids

        dna_end_id = self.dna_start_id + self.dna_vocab_size
        is_label_in_dna_vocab = (lid >= self.dna_start_id) & (lid < dna_end_id)
        is_label_kmer = (lid >= self.dna_kmer_start_id) & (lid < self.dna_kmer_end_id)

        # token_mask allowed values: -2, -1, 0..k
        bad_tm = (tm < -2) | (tm > self.k)
        if (bad_tm & lm).any():
            idx = (bad_tm & lm).nonzero(as_tuple=False)[:10]
            raise RuntimeError(
                "[HybridLoss] token_mask has invalid values on supervised positions.\n"
                f"Examples (b,t)={idx.tolist()}, token_mask={tm[idx[:,0], idx[:,1]].tolist()}\n"
                "Expected token_mask in {-2, -1, 0..k}."
            )

        # Padding positions must NOT be supervised
        if ((tm == -2) & lm).any():
            idx = (((tm == -2) & lm).nonzero(as_tuple=False)[:10])
            raise RuntimeError(
                "[HybridLoss] Found token_mask == -2 (padding) on supervised positions.\n"
                f"Examples (b,t)={idx.tolist()}"
            )

        # DNA supervision: tm in [1..k] => label must be a k-mer id
        dna_should_be_kmer = lm & (tm >= 1) & (tm <= self.k)
        if (dna_should_be_kmer & ~is_label_kmer).any():
            idx = (dna_should_be_kmer & ~is_label_kmer).nonzero(as_tuple=False)[:10]
            raise RuntimeError(
                "[HybridLoss] token_mask/label_ids mismatch: token_mask marks a DNA k-mer, "
                "but label_id is NOT in the k-mer id range.\n"
                f"kmer_id_range=[{self.dna_kmer_start_id},{self.dna_kmer_end_id})\n"
                f"Examples (b,t)={idx.tolist()}, label_id={lid[idx[:,0], idx[:,1]].tolist()}, "
                f"token_mask={tm[idx[:,0], idx[:,1]].tolist()}\n"
                "This usually indicates token_mask is not shifted the same way as label_ids, "
                "or token_mask incorrectly marks a non-DNA token as DNA."
            )

        # Special/ignored DNA positions (tm==0) must NOT carry a k-mer label,
        # otherwise supervision would be silently dropped.
        special_ignore = lm & (tm == 0)
        if (special_ignore & is_label_kmer).any():
            idx = (special_ignore & is_label_kmer).nonzero(as_tuple=False)[:10]
            raise RuntimeError(
                "[HybridLoss] token_mask/shift mismatch: token_mask==0 (ignored) but label_id is a DNA k-mer.\n"
                f"Examples (b,t)={idx.tolist()}, label_id={lid[idx[:,0], idx[:,1]].tolist()}\n"
                "This strongly suggests token_mask shift misalignment."
            )

        # NL supervision (tm==-1) should not see DNA vocab labels
        nl_pos = lm & (tm == -1)
        if (nl_pos & is_label_in_dna_vocab).any():
            idx = (nl_pos & is_label_in_dna_vocab).nonzero(as_tuple=False)[:10]
            raise RuntimeError(
                "[HybridLoss] token_mask/shift mismatch: token_mask==-1 (NL) but label_id falls in DNA vocab range.\n"
                f"Examples (b,t)={idx.tolist()}, label_id={lid[idx[:,0], idx[:,1]].tolist()}"
            )

        # Tail rule: tm in [1..k-1] indicates a tail k-mer, so the next supervised position
        # should NOT be another DNA k-mer (otherwise valid_len<k is suspicious).
        if label_ids.shape[1] > 1:
            tail = lm & (tm >= 1) & (tm < self.k)
            next_is_dna_kmer = (tm[:, 1:] >= 1) & (tm[:, 1:] <= self.k)
            bad = tail[:, :-1] & next_is_dna_kmer & lm[:, :-1]
            if bad.any():
                idx = bad.nonzero(as_tuple=False)[:10]
                raise RuntimeError(
                    "[HybridLoss] Tail rule violated: token_mask in [1..k-1] is followed by another DNA k-mer.\n"
                    f"Examples (b,t)={idx.tolist()}, token_mask[t]={tm[idx[:,0], idx[:,1]].tolist()}, "
                    f"token_mask[t+1]={tm[idx[:,0], idx[:,1]+1].tolist()}\n"
                    "This strongly suggests token_mask shift misalignment or corrupted DNA boundaries."
                )

    # ---------------------------
    # DNA marginal BP loss
    # ---------------------------
    def _bp_nll_sum_and_count(
        self,
        logits_dna: torch.Tensor,      # [N, V_local]
        label_ids_dna: torch.Tensor,   # [N] (MUST be k-mer ids)
        valid_len: torch.Tensor,       # [N] in [1..k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          total_sum   : sum of NLL over all supervised bases (float32)
          total_count : number of supervised bases (long), equals sum(valid_len)

        Requirements:
          - label_ids_dna must be all valid DNA k-mer token ids.
          - valid_len controls tail truncation (A-padding bases are excluded).
        """
        device = logits_dna.device
        N, V_local = logits_dna.shape
        self._maybe_build_local_cache(device, V_local)
        assert self._local_nt_indices is not None

        # Strict: all labels here must be k-mers
        is_kmer = (label_ids_dna >= self.dna_kmer_start_id) & (label_ids_dna < self.dna_kmer_end_id)
        if not is_kmer.all():
            bad = (~is_kmer).nonzero(as_tuple=True)[0][:10]
            raise RuntimeError(
                "[HybridLoss] INTERNAL ERROR: non-kmer labels passed into _bp_nll_sum_and_count.\n"
                f"kmer_id_range=[{self.dna_kmer_start_id},{self.dna_kmer_end_id})\n"
                f"bad label_ids={label_ids_dna[bad].tolist()}, valid_len={valid_len[bad].tolist()}"
            )

        kmer_idx = (label_ids_dna - self.dna_kmer_start_id).to(torch.long)  # [N] in [0..num_dna_kmers-1]

        # TP-stable softmax denominator: log(sum(exp(logits)))
        local_max = logits_dna.max(dim=-1).values.to(torch.float32)  # [N]
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=self.tp_pg)

        exp_logits = torch.exp(logits_dna.to(torch.float32) - global_max.unsqueeze(-1))  # [N, V_local]

        local_sumexp = exp_logits.sum(dim=-1)  # [N]
        global_sumexp = local_sumexp.clone()
        dist.all_reduce(global_sumexp, op=dist.ReduceOp.SUM, group=self.tp_pg)
        global_sumexp = global_sumexp.clamp_min(self.eps)
        log_denom = torch.log(global_sumexp)  # [N]

        total_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_count = torch.tensor(0, device=device, dtype=torch.long)

        # Marginalize over A/T/C/G for each base position inside the k-mer
        for pos in range(self.k):
            # Only supervise first valid_len bases (tail A-padding excluded)
            pos_mask = (valid_len > pos)
            if not pos_mask.any():
                continue

            # src_idx_pos: [V_local] in {-1,0,1,2,3}
            src_idx_pos = self._local_nt_indices[:, pos]
            valid_vocab = (src_idx_pos >= 0)

            # Exclude non-kmer tokens from the marginal sums
            exp_masked = exp_logits if valid_vocab.all() else exp_logits * valid_vocab.to(exp_logits.dtype).view(1, -1)

            idx_clamped = src_idx_pos.clamp_min(0)  # -1 -> 0, but exp_masked already zeros them out

            # marg[n, nt] = sum_{token: token[pos]==nt} exp_logits[n, token]
            marg = torch.zeros((N, 4), device=device, dtype=torch.float32)
            marg.scatter_add_(
                dim=1,
                index=idx_clamped.view(1, -1).expand(N, -1),
                src=exp_masked,
            )
            dist.all_reduce(marg, op=dist.ReduceOp.SUM, group=self.tp_pg)
            marg = marg.clamp_min(self.eps)

            log_marg = torch.log(marg) - log_denom.unsqueeze(-1)  # [N,4]

            target_nt = self._dna_nt_table[kmer_idx, pos].to(device=device)  # [N] in {0,1,2,3}
            if (target_nt < 0).any():
                bad = (target_nt < 0).nonzero(as_tuple=True)[0][:10]
                raise RuntimeError(
                    "[HybridLoss] Invalid nucleotide mapping (target_nt < 0). "
                    "Check dna_id_to_token / dna_start_id consistency.\n"
                    f"bad idx={bad.tolist()}, kmer_idx={kmer_idx[bad].tolist()}"
                )

            nll = -log_marg.gather(dim=1, index=target_nt.unsqueeze(1)).squeeze(1)  # [N]
            m = pos_mask.to(nll.dtype)

            total_sum = total_sum + (nll * m).sum()
            total_count = total_count + pos_mask.sum()

        return total_sum, total_count

    # ---------------------------
    # Main forward
    # ---------------------------
    def forward(
        self,
        sharded_logits: torch.Tensor,  # [B*S, V_local]
        label_ids: torch.Tensor,       # [B,S] (already shifted by dataloader)
        label_mask: torch.Tensor,      # [B,S]
        token_mask: torch.Tensor,      # [B,S] (MUST be shifted same as label_ids)
    ) -> Dict[str, torch.Tensor]:
        sharded_logits_3d = sharded_logits.view(label_ids.shape[0], label_ids.shape[1], -1)  # [B,S,V_local]
        device = sharded_logits_3d.device

        # Fail fast on shift/type mismatches
        self._assert_alignment(label_ids=label_ids, label_mask=label_mask, token_mask=token_mask)

        lm = _to_bool_mask(label_mask).to(device=device)
        tm = token_mask.to(device=device).to(torch.long)

        nl_pos = lm & (tm == -1)
        dna_pos = lm & (tm >= 1) & (tm <= self.k)

        # 1) NL: token-level CE
        if nl_pos.any():
            ce = sharded_cross_entropy(
                sharded_logits_3d,
                label_ids.contiguous(),
                group=self.tp_pg,
                dtype=torch.float,
            )  # [B,S]
            nl_sum = (ce * nl_pos.to(ce.dtype)).sum(dtype=torch.float32)
            nl_count = nl_pos.sum()
        else:
            nl_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
            nl_count = torch.tensor(0, device=device, dtype=torch.long)

        # 2) DNA: base-level marginal NLL
        if dna_pos.any():
            logits_dna = sharded_logits_3d[dna_pos]  # [N_dna,V_local]
            labels_dna = label_ids[dna_pos]          # [N_dna]
            valid_len = tm[dna_pos]                  # [N_dna] in [1..k]
            bp_sum, bp_count = self._bp_nll_sum_and_count(logits_dna, labels_dna, valid_len)
        else:
            bp_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
            bp_count = torch.tensor(0, device=device, dtype=torch.long)

        # 3) Combine by supervised units:
        #    NL is counted per token, DNA is counted per base (sum(valid_len)).
        total_count = nl_count + bp_count
        loss = (self.nl_weight * nl_sum + self.bp_weight * bp_sum) / torch.clamp(total_count, min=1)

        return {"loss": loss}


class HybridLossWithZLoss(HybridLoss):
    """
    Hybrid loss + z-loss, z applies to BOTH NL and DNA.

    Assumption (consistent with nanotron LossWithZLoss usage):
      - sharded_cross_entropy(..., z_loss_coef=coef) returns:
          (token_loss_with_z, z_scaled)
        where:
          token_loss_with_z = CE + coef * z_term
          z_scaled          = coef * z_term   (for logging or custom aggregation)

    DNA z-loss uses per-base scaling:
      - each DNA token's z_scaled is expanded into base units by multiplying valid_len,
        so tail A-padding bases are NOT counted.
    """

    def __init__(self, tp_pg: dist.ProcessGroup, z_loss_coefficient: float, **kwargs):
        super().__init__(tp_pg=tp_pg, **kwargs)
        self.z_loss_coef = float(z_loss_coefficient)

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [B*S, V_local]
        label_ids: torch.Tensor,       # [B,S]
        label_mask: torch.Tensor,      # [B,S]
        token_mask: torch.Tensor,      # [B,S]
    ) -> Dict[str, torch.Tensor]:
        sharded_logits_3d = sharded_logits.view(label_ids.shape[0], label_ids.shape[1], -1)  # [B,S,V_local]
        device = sharded_logits_3d.device

        self._assert_alignment(label_ids=label_ids, label_mask=label_mask, token_mask=token_mask)

        lm = _to_bool_mask(label_mask).to(device=device)
        tm = token_mask.to(device=device).to(torch.long)

        nl_pos = lm & (tm == -1)
        dna_pos = lm & (tm >= 1) & (tm <= self.k)

        # IMPORTANT:
        # We call sharded_cross_entropy on the full [B,S,V_local] tensor to keep TP collectives consistent.
        tok_loss_with_z, tok_z_scaled = sharded_cross_entropy(
            sharded_logits_3d,
            label_ids.contiguous(),
            group=self.tp_pg,
            dtype=torch.float,
            z_loss_coef=self.z_loss_coef,
        )  # both [B,S]

        # 1) NL: sum(CE + z) over NL token positions
        if nl_pos.any():
            nl_sum = (tok_loss_with_z * nl_pos.to(tok_loss_with_z.dtype)).sum(dtype=torch.float32)
            nl_count = nl_pos.sum()
            z_nl_sum = (tok_z_scaled.detach() * nl_pos.to(tok_z_scaled.dtype)).sum(dtype=torch.float32)
        else:
            nl_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
            nl_count = torch.tensor(0, device=device, dtype=torch.long)
            z_nl_sum = torch.tensor(0.0, device=device, dtype=torch.float32)

        # 2) DNA: BP marginal + per-base z-loss
        if dna_pos.any():
            logits_dna = sharded_logits_3d[dna_pos]
            labels_dna = label_ids[dna_pos]
            valid_len = tm[dna_pos].to(torch.long)  # [N_dna] in [1..k]

            bp_sum, bp_count = self._bp_nll_sum_and_count(logits_dna, labels_dna, valid_len)

            # per-base z: expand token-level z into base units via valid_len
            dna_z_tok = tok_z_scaled[dna_pos].to(torch.float32)
            dna_z_sum = (dna_z_tok * valid_len.to(torch.float32)).sum(dtype=torch.float32)
        else:
            bp_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
            bp_count = torch.tensor(0, device=device, dtype=torch.long)
            dna_z_sum = torch.tensor(0.0, device=device, dtype=torch.float32)

        # 3) Main loss uses supervised-unit denominator: NL tokens + DNA bases
        total_count = nl_count + bp_count
        loss = (
            self.nl_weight * nl_sum
            + self.bp_weight * (bp_sum + dna_z_sum)
        ) / torch.clamp(total_count, min=1)

        # 4) Logging-only z_loss (also per supervised unit, and consistent with weights)
        z_loss = (self.nl_weight * z_nl_sum + self.bp_weight * dna_z_sum.detach()) / torch.clamp(total_count, min=1)

        return {"loss": loss, "z_loss": z_loss}
