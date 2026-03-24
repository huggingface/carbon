#!/usr/bin/env python3
"""
Regulatory Element Annotations
===============================
Annotates: TATA boxes, CpG islands, restriction enzyme sites,
           transcription factor binding sites (TFBS via JASPAR).

Data sources:
  - JASPAR REST API — TFBS motif scanning
  - Biopython Restriction — enzyme recognition sites
  - Local scanning — promoter motifs, CpG islands

Produces annotations with category="regulatory_elements".
"""

import math
import re
import requests
from common import (
    GenomicRegion, Annotation, AnnotatedSequence, DEFAULT_REGION,
    make_annotated_sequence,
)

JASPAR_REST = "https://jaspar.elixir.no/api/v1"


def annotate(region: GenomicRegion) -> AnnotatedSequence:
    """Fetch sequence + regulatory element annotations for a genomic region."""
    result = make_annotated_sequence(region)
    seq = result.sequence

    # --- 1. Core promoter elements (local regex) ---
    patterns = {
        ("TATA_box", r"TATA[AT]A[AT]?", "TATA box promoter element"),
        ("CAAT_box", r"GG[CT]CAATCT", "CAAT box promoter element"),
        ("GC_box", r"GGGCGG", "GC box (Sp1 binding)"),
        ("Initiator", r"[CT][CT]A[ACGT]T[CT][CT]", "Initiator element (Inr)"),
    }
    for etype, pattern, label in patterns:
        for m in re.finditer(pattern, seq):
            result.annotations.append(Annotation(
                start=m.start() + 1, end=m.end(),
                type=etype, category="regulatory_elements",
                label=f"{label}: {m.group()}",
                metadata={"motif": m.group()},
            ))

    # --- 2. Restriction enzyme sites (Biopython) ---
    try:
        from Bio.Seq import Seq
        from Bio.Restriction import RestrictionBatch

        enzymes = ["EcoRI", "BamHI", "HindIII", "XbaI", "SalI",
                    "KpnI", "SmaI", "NotI", "PstI", "EcoRV"]
        rb = RestrictionBatch(enzymes)
        bio_seq = Seq(seq)
        cuts = rb.search(bio_seq)

        for enzyme, positions in cuts.items():
            for pos in positions:
                site = str(enzyme.site)
                result.annotations.append(Annotation(
                    start=pos, end=pos + len(site) - 1,
                    type="restriction_site", category="regulatory_elements",
                    label=f"{enzyme} restriction site ({site})",
                    metadata={"enzyme": str(enzyme), "recognition": site},
                ))
    except ImportError:
        pass

    # --- 3. CpG islands (sliding window) ---
    window = min(100, len(seq))
    step = max(1, window // 5)
    in_island = False
    island_start = 0

    for i in range(0, len(seq) - window + 1, step):
        subseq = seq[i:i + window]
        gc = (subseq.count("C") + subseq.count("G")) / len(subseq)
        cpg = subseq.count("CG")
        c_count, g_count = subseq.count("C"), subseq.count("G")
        expected = (c_count * g_count) / len(subseq) if len(subseq) > 0 else 0
        obs_exp = cpg / expected if expected > 0 else 0

        is_cpg = gc >= 0.5 and obs_exp >= 0.6

        if is_cpg and not in_island:
            island_start = i
            in_island = True
        elif not is_cpg and in_island:
            if i + window - island_start >= 50:
                result.annotations.append(Annotation(
                    start=island_start + 1, end=i + window,
                    type="CpG_island", category="regulatory_elements",
                    label=f"CpG island (GC={gc:.2f}, obs/exp={obs_exp:.2f})",
                    score=round(obs_exp, 3),
                    metadata={"gc_content": round(gc, 3), "obs_exp_cpg": round(obs_exp, 3)},
                ))
            in_island = False

    if in_island and len(seq) - island_start >= 50:
        result.annotations.append(Annotation(
            start=island_start + 1, end=len(seq),
            type="CpG_island", category="regulatory_elements",
            label="CpG island",
            metadata={"gc_content": round(gc, 3)},
        ))

    # --- 4. JASPAR TFBS motif scan (top 3 vertebrate motifs) ---
    try:
        for tf_name in ["CTCF", "SP1", "CREB1"]:
            url = f"{JASPAR_REST}/matrix/"
            params = {"name": tf_name, "tax_group": "vertebrates",
                      "format": "json", "page_size": 1}
            resp = requests.get(url, params=params, timeout=15)
            if not resp.ok:
                continue
            motifs = resp.json().get("results", [])
            if not motifs:
                continue

            # Get PFM
            pfm_resp = requests.get(f"{JASPAR_REST}/matrix/{motifs[0]['matrix_id']}/",
                                     params={"format": "json"}, timeout=15)
            if not pfm_resp.ok:
                continue
            pfm = pfm_resp.json().get("pfm", {})
            if not pfm or "A" not in pfm:
                continue

            motif_len = len(pfm["A"])
            if motif_len > len(seq):
                continue

            # Score each position
            bases = ["A", "C", "G", "T"]
            score_mat = []
            for i in range(motif_len):
                total = sum(pfm[b][i] for b in bases)
                score_mat.append({
                    b: math.log2((pfm[b][i] + 0.01) / (total + 0.04) / 0.25)
                    for b in bases
                })
            max_score = sum(max(pos.values()) for pos in score_mat)

            for s in range(len(seq) - motif_len + 1):
                subseq = seq[s:s + motif_len]
                if any(b not in "ACGT" for b in subseq):
                    continue
                sc = sum(score_mat[i][subseq[i]] for i in range(motif_len))
                rel = sc / max_score if max_score > 0 else 0
                if rel >= 0.85:
                    result.annotations.append(Annotation(
                        start=s + 1, end=s + motif_len,
                        type="TFBS", category="regulatory_elements",
                        label=f"{tf_name} binding site (JASPAR {motifs[0]['matrix_id']})",
                        score=round(rel, 3),
                        metadata={"tf": tf_name, "matrix_id": motifs[0]["matrix_id"],
                                  "match_sequence": subseq},
                    ))
    except Exception:
        pass

    return result


def main():
    print("Fetching regulatory element annotations...")
    result = annotate(DEFAULT_REGION)
    print(result.summary())
    print(f"\nSequence (first 80bp): {result.sequence[:80]}...")
    with open("02_output.json", "w") as f:
        f.write(result.to_json())
    print("Saved to 02_output.json")


if __name__ == "__main__":
    main()
