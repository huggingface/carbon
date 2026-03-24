#!/usr/bin/env python3
"""
Conservation Annotations
========================
Annotates: constrained elements, conservation scores, cross-species alignment.

Data sources:
  - Ensembl Compara — GERP constrained elements
  - UCSC REST API — PhyloP/PhastCons conserved elements
  - Local — GC content as rough conservation proxy

Produces annotations with category="conservation".
"""

import requests
from common import (
    GenomicRegion, Annotation, AnnotatedSequence, DEFAULT_REGION,
    ENSEMBL_REST, make_annotated_sequence, clamp_to_region, ensembl_get,
)

UCSC_API = "https://api.genome.ucsc.edu"


def annotate(region: GenomicRegion) -> AnnotatedSequence:
    """Fetch sequence + conservation annotations for a genomic region."""
    result = make_annotated_sequence(region)

    # --- 1. Ensembl constrained elements (GERP) ---
    url = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
    params = {"feature": "constrained", "content-type": "application/json"}
    try:
        resp = ensembl_get(url, params=params)
        resp.raise_for_status()
        for feat in resp.json():
            local_start, local_end = clamp_to_region(feat["start"], feat["end"], region)
            if local_start == 0:
                continue
            score = feat.get("score")
            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type="constrained_element", category="conservation",
                label=f"GERP constrained element (score={score})",
                score=float(score) if score is not None else None,
                metadata={"method": feat.get("method_link_type", "GERP")},
            ))
    except Exception:
        pass

    # --- 2. UCSC phastCons conserved elements ---
    try:
        url2 = f"{UCSC_API}/getData/track"
        params2 = {
            "genome": "hg38",
            "track": "phastConsElements100way",
            "chrom": region.ucsc_chrom,
            "start": region.start,
            "end": region.end,
        }
        resp2 = requests.get(url2, params=params2, timeout=30)
        resp2.raise_for_status()
        data = resp2.json()

        for track_name, items in data.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                elem_start = item.get("chromStart", 0)
                elem_end = item.get("chromEnd", 0)
                local_start, local_end = clamp_to_region(elem_start, elem_end, region)
                if local_start == 0:
                    continue
                score = item.get("score", 0)
                result.annotations.append(Annotation(
                    start=local_start, end=local_end,
                    type="phastCons_element", category="conservation",
                    label=f"phastCons conserved element (100-way vertebrate)",
                    score=float(score),
                    metadata={"track": "phastConsElements100way", "name": item.get("name", "")},
                ))
    except Exception:
        pass

    # --- 3. Local GC content in windows (proxy for coding conservation) ---
    seq = result.sequence
    window = 50
    step = 25
    for i in range(0, len(seq) - window + 1, step):
        subseq = seq[i:i + window]
        gc = (subseq.count("G") + subseq.count("C")) / len(subseq)
        cpg_count = subseq.count("CG")
        result.annotations.append(Annotation(
            start=i + 1, end=i + window,
            type="gc_content", category="conservation",
            label=f"GC content window ({gc:.1%})",
            score=round(gc, 3),
            metadata={"window_size": window, "cpg_count": cpg_count},
        ))

    return result


def main():
    print("Fetching conservation annotations...")
    result = annotate(DEFAULT_REGION)
    print(result.summary())
    print(f"\nSequence (first 80bp): {result.sequence[:80]}...")
    with open("05_output.json", "w") as f:
        f.write(result.to_json())
    print("Saved to 05_output.json")


if __name__ == "__main__":
    main()
