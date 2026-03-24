#!/usr/bin/env python3
"""
miRNA & Regulatory RNA Annotations
====================================
Annotates: miRNA seed matches, ncRNA genes, stem-loop structures.

Data sources:
  - Ensembl REST API — ncRNA gene annotations in region
  - Local — miRNA seed matching, hairpin structure prediction

Produces annotations with category="ncrna".
"""

import re
import requests
from common import (
    GenomicRegion, Annotation, AnnotatedSequence, DEFAULT_REGION,
    ENSEMBL_REST, make_annotated_sequence, clamp_to_region, ensembl_get,
)

# Well-known miRNA seed sequences (positions 2-8, DNA form)
MIRNA_SEEDS = {
    "miR-21-5p": "AGCTTAT",
    "miR-155-5p": "TAATGCT",
    "miR-let-7a": "GAGGTAG",
    "miR-122-5p": "GGAGTGT",
    "miR-34a-5p": "GGCAGTG",
    "miR-200c-3p": "AATACTG",
    "miR-17-5p": "AAAGTGC",
    "miR-10b-5p": "ACCCGT",
}


def annotate(region: GenomicRegion) -> AnnotatedSequence:
    """Fetch sequence + ncRNA annotations for a genomic region."""
    result = make_annotated_sequence(region)
    seq = result.sequence

    # --- 1. Ensembl ncRNA genes overlapping region ---
    url = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
    params = {"feature": ["gene", "transcript"], "content-type": "application/json"}
    try:
        resp = ensembl_get(url, params=params)
        resp.raise_for_status()

        ncrna_biotypes = {
            "miRNA", "lncRNA", "snoRNA", "snRNA", "rRNA", "tRNA",
            "scRNA", "scaRNA", "ribozyme", "antisense", "sense_intronic",
            "lincRNA", "misc_RNA", "processed_transcript",
        }

        for feat in resp.json():
            biotype = feat.get("biotype", "")
            if biotype not in ncrna_biotypes:
                continue

            local_start, local_end = clamp_to_region(feat["start"], feat["end"], region)
            if local_start == 0:
                continue

            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type=f"ncrna_{biotype}", category="ncrna",
                label=f"{feat.get('external_name', '')} ({biotype})",
                strand="+" if feat.get("strand", 1) >= 0 else "-",
                metadata={
                    "gene_id": feat.get("id"),
                    "gene_name": feat.get("external_name", ""),
                    "biotype": biotype,
                },
            ))
    except Exception:
        pass

    # --- 2. miRNA seed matching ---
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}

    for mirna, seed in MIRNA_SEEDS.items():
        seed_rc = "".join(complement.get(b, "N") for b in reversed(seed))

        for m in re.finditer(seed_rc, seq):
            site_type = "7mer-m8"
            if m.end() < len(seq) and seq[m.end()] == "A":
                site_type = "8mer"

            result.annotations.append(Annotation(
                start=m.start() + 1, end=m.end(),
                type="mirna_target", category="ncrna",
                label=f"{mirna} target site ({site_type})",
                metadata={
                    "mirna": mirna,
                    "seed": seed,
                    "site_type": site_type,
                    "match_sequence": m.group(),
                },
            ))

    # --- 3. Stem-loop / hairpin prediction ---
    rna_seq = seq.replace("T", "U")
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}

    hairpins = []
    for i in range(len(rna_seq)):
        for loop_size in range(3, 9):
            for stem_len in range(5, min(15, (len(rna_seq) - i - loop_size) // 2 + 1)):
                right_end = i + stem_len * 2 + loop_size
                if right_end > len(rna_seq):
                    break

                left = rna_seq[i:i + stem_len]
                right = rna_seq[i + stem_len + loop_size:right_end][::-1]

                pairs = sum(
                    1 for a, b in zip(left, right)
                    if comp.get(a) == b or {a, b} == {"G", "U"}
                )

                if pairs >= stem_len * 0.8:
                    hairpins.append({
                        "start": i + 1,
                        "end": right_end,
                        "stem": stem_len,
                        "loop": loop_size,
                        "pairs": pairs,
                        "energy_proxy": -(pairs * 2 + stem_len),
                    })

    # Keep best non-overlapping
    hairpins.sort(key=lambda x: x["energy_proxy"])
    covered = set()
    for hp in hairpins:
        positions = set(range(hp["start"], hp["end"] + 1))
        if len(positions & covered) < len(positions) * 0.3:
            result.annotations.append(Annotation(
                start=hp["start"], end=hp["end"],
                type="stem_loop", category="ncrna",
                label=f"Predicted stem-loop (stem={hp['stem']}bp, loop={hp['loop']}nt)",
                score=float(hp["energy_proxy"]),
                metadata={
                    "stem_length": hp["stem"],
                    "loop_size": hp["loop"],
                    "base_pairs": hp["pairs"],
                },
            ))
            covered |= positions

    return result


def main():
    print("Fetching ncRNA annotations...")
    result = annotate(DEFAULT_REGION)
    print(result.summary())
    print(f"\nSequence (first 80bp): {result.sequence[:80]}...")
    with open("09_output.json", "w") as f:
        f.write(result.to_json())
    print("Saved to 09_output.json")


if __name__ == "__main__":
    main()
