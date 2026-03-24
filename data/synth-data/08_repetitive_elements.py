#!/usr/bin/env python3
"""
Repetitive Element Annotations
===============================
Annotates: Alu/SINE, LINE, simple repeats, microsatellites, transposons.

Data sources:
  - UCSC RepeatMasker track — pre-computed repeat annotations
  - Ensembl REST API — repeat features
  - Local — simple/tandem repeat detection

Produces annotations with category="repeats".
"""

import re
import requests
from common import (
    GenomicRegion, Annotation, AnnotatedSequence, DEFAULT_REGION,
    ENSEMBL_REST, make_annotated_sequence, clamp_to_region, ensembl_get,
)

UCSC_API = "https://api.genome.ucsc.edu"


def annotate(region: GenomicRegion) -> AnnotatedSequence:
    """Fetch sequence + repetitive element annotations for a genomic region."""
    result = make_annotated_sequence(region)

    # --- 1. UCSC RepeatMasker track ---
    try:
        url = f"{UCSC_API}/getData/track"
        params = {
            "genome": "hg38",
            "track": "rmsk",
            "chrom": region.ucsc_chrom,
            "start": region.start,
            "end": region.end,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for track_name, items in data.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                local_start, local_end = clamp_to_region(
                    item.get("genoStart", 0), item.get("genoEnd", 0), region)
                if local_start == 0:
                    continue

                rep_class = item.get("repClass", "")
                rep_family = item.get("repFamily", "")
                rep_name = item.get("repName", "")
                div = item.get("milliDiv", 0) / 10  # permille to percent

                result.annotations.append(Annotation(
                    start=local_start, end=local_end,
                    type=f"repeat_{rep_class.replace('/', '_').lower()}",
                    category="repeats",
                    label=f"{rep_name} ({rep_class}/{rep_family})",
                    strand=item.get("strand", "."),
                    score=round(div, 1),
                    metadata={
                        "repeat_name": rep_name,
                        "repeat_class": rep_class,
                        "repeat_family": rep_family,
                        "divergence_pct": round(div, 1),
                        "source": "UCSC_RepeatMasker",
                    },
                ))
    except Exception:
        pass

    # --- 2. Ensembl repeat features ---
    try:
        url2 = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
        params2 = {"feature": "repeat", "content-type": "application/json"}
        resp2 = ensembl_get(url2, params=params2)
        resp2.raise_for_status()
        for feat in resp2.json():
            local_start, local_end = clamp_to_region(feat["start"], feat["end"], region)
            if local_start == 0:
                continue
            desc = feat.get("description", "unknown repeat")
            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type="repeat_ensembl", category="repeats",
                label=f"Ensembl repeat: {desc}",
                strand="+" if feat.get("strand", 1) >= 0 else "-",
                metadata={"description": desc, "source": "Ensembl"},
            ))
    except Exception:
        pass

    # --- 3. Local: simple/tandem repeat detection ---
    seq = result.sequence
    for unit_len in range(1, 7):
        for i in range(len(seq) - unit_len * 3):
            unit = seq[i:i + unit_len]
            if unit_len > 1 and len(set(unit)) == 1:
                continue
            copies = 1
            pos = i + unit_len
            while pos + unit_len <= len(seq) and seq[pos:pos + unit_len] == unit:
                copies += 1
                pos += unit_len
            if copies >= 3 and unit_len * copies >= 6:
                result.annotations.append(Annotation(
                    start=i + 1, end=i + unit_len * copies,
                    type="simple_repeat", category="repeats",
                    label=f"({unit})n, n={copies}",
                    metadata={"unit": unit, "unit_length": unit_len, "copies": copies},
                ))

    # Deduplicate overlapping simple repeats (keep longest)
    simple = [a for a in result.annotations if a.type == "simple_repeat"]
    simple.sort(key=lambda x: -(x.end - x.start))
    covered = set()
    keep = set()
    for a in simple:
        positions = set(range(a.start, a.end + 1))
        if len(positions & covered) < len(positions) * 0.5:
            keep.add(id(a))
            covered |= positions

    result.annotations = [a for a in result.annotations
                          if a.type != "simple_repeat" or id(a) in keep]

    return result


def main():
    print("Fetching repetitive element annotations...")
    result = annotate(DEFAULT_REGION)
    print(result.summary())
    print(f"\nSequence (first 80bp): {result.sequence[:80]}...")
    with open("08_output.json", "w") as f:
        f.write(result.to_json())
    print("Saved to 08_output.json")


if __name__ == "__main__":
    main()
