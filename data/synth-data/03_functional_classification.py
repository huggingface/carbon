#!/usr/bin/env python3
"""
Functional Classification
=========================
Classifies each position as: CDS, exon, UTR, intron, intergenic, regulatory.

Data sources:
  - Ensembl REST API — biotype and feature overlap
  - UniProt REST API — protein function annotations

Produces annotations with category="functional_class".
"""

import requests
from common import (
    GenomicRegion, Annotation, AnnotatedSequence, DEFAULT_REGION,
    ENSEMBL_REST, make_annotated_sequence, clamp_to_region, ensembl_get,
)


# Priority for per-position classification
_PRIORITY = {
    "intergenic": 0, "intron": 1, "regulatory": 2,
    "UTR": 3, "exon": 4, "CDS": 5,
}


def annotate(region: GenomicRegion) -> AnnotatedSequence:
    """Fetch sequence + functional classification for a genomic region."""
    result = make_annotated_sequence(region)
    n = result.length

    # --- 1. Ensembl overlap features ---
    url = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
    params = {
        "feature": ["gene", "transcript", "exon", "cds"],
        "content-type": "application/json",
    }
    resp = ensembl_get(url, params=params)
    resp.raise_for_status()

    # Build per-position classification
    classification = ["intergenic"] * n

    for feat in resp.json():
        feat_type = feat.get("feature_type", feat.get("object_type", "")).lower()
        local_start, local_end = clamp_to_region(feat["start"], feat["end"], region)
        if local_start == 0:
            continue

        if feat_type == "cds":
            cls = "CDS"
        elif feat_type == "exon":
            cls = "exon"
        elif feat_type in ("five_prime_utr", "three_prime_utr"):
            cls = "UTR"
        elif feat_type in ("gene", "transcript"):
            cls = "intron"
        else:
            continue

        for i in range(local_start - 1, local_end):
            if _PRIORITY.get(cls, 0) > _PRIORITY.get(classification[i], 0):
                classification[i] = cls

    # --- 2. Ensembl regulatory features ---
    url2 = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
    params2 = {"feature": "regulatory", "content-type": "application/json"}
    try:
        resp2 = ensembl_get(url2, params=params2)
        resp2.raise_for_status()
        for feat in resp2.json():
            local_start, local_end = clamp_to_region(feat["start"], feat["end"], region)
            if local_start == 0:
                continue
            for i in range(local_start - 1, local_end):
                if _PRIORITY.get("regulatory", 0) > _PRIORITY.get(classification[i], 0):
                    classification[i] = "regulatory"
    except Exception:
        pass

    # --- 3. Merge consecutive positions into range annotations ---
    current = classification[0]
    start = 0
    for i in range(1, n):
        if classification[i] != current:
            result.annotations.append(Annotation(
                start=start + 1, end=i,
                type=current, category="functional_class",
                label=f"{current} region",
                metadata={"length": i - start},
            ))
            current = classification[i]
            start = i
    result.annotations.append(Annotation(
        start=start + 1, end=n,
        type=current, category="functional_class",
        label=f"{current} region",
        metadata={"length": n - start},
    ))

    # --- 4. Gene-level biotype annotations ---
    for feat in resp.json():
        if feat.get("feature_type", feat.get("object_type", "")).lower() == "gene":
            local_start, local_end = clamp_to_region(feat["start"], feat["end"], region)
            if local_start == 0:
                continue
            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type=f"biotype_{feat.get('biotype', 'unknown')}",
                category="functional_class",
                label=f"{feat.get('external_name', '')} ({feat.get('biotype', '')})",
                metadata={"gene_id": feat.get("id"), "biotype": feat.get("biotype")},
            ))

    return result


def main():
    print("Fetching functional classification...")
    result = annotate(DEFAULT_REGION)
    print(result.summary())
    print(f"\nSequence (first 80bp): {result.sequence[:80]}...")
    with open("03_output.json", "w") as f:
        f.write(result.to_json())
    print("Saved to 03_output.json")


if __name__ == "__main__":
    main()
