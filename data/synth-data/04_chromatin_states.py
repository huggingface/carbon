#!/usr/bin/env python3
"""
ENCODE-style Chromatin State Annotations
=========================================
Annotates: histone modifications, chromatin states, regulatory features.

Data sources:
  - ENCODE REST API — experiments for histone ChIP-seq
  - Ensembl Regulatory Build — regulatory features + activity
  - Rule-based — histone mark to chromatin state mapping

Produces annotations with category="chromatin_state".
"""

import requests
from common import (
    GenomicRegion, Annotation, AnnotatedSequence, DEFAULT_REGION,
    ENSEMBL_REST, make_annotated_sequence, clamp_to_region, ensembl_get,
)

ENCODE_API = "https://www.encodeproject.org"


def annotate(region: GenomicRegion) -> AnnotatedSequence:
    """Fetch sequence + chromatin state annotations for a genomic region."""
    result = make_annotated_sequence(region)

    # --- 1. Ensembl Regulatory Build features ---
    url = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
    params = {"feature": "regulatory", "content-type": "application/json"}
    try:
        resp = ensembl_get(url, params=params)
        resp.raise_for_status()
        for feat in resp.json():
            local_start, local_end = clamp_to_region(feat["start"], feat["end"], region)
            if local_start == 0:
                continue
            feat_type = feat.get("description", feat.get("feature_type", "unknown"))
            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type=f"regulatory_{feat_type.replace(' ', '_').lower()}",
                category="chromatin_state",
                label=f"Ensembl regulatory: {feat_type}",
                metadata={
                    "regulatory_id": feat.get("id"),
                    "feature_type": feat_type,
                },
            ))
    except Exception:
        pass

    # --- 2. ENCODE: find histone ChIP-seq experiments for this region ---
    # We search for experiments, then note which marks are studied here
    histone_marks = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K27me3", "H3K9me3"]
    found_marks = []

    for mark in histone_marks:
        try:
            url = f"{ENCODE_API}/search/"
            params = {
                "type": "Experiment",
                "assay_title": "Histone ChIP-seq",
                "target.label": mark,
                "biosample_ontology.term_name": "K562",
                "status": "released",
                "limit": 1,
            }
            headers = {"Accept": "application/json"}
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            data = resp.json()
            if data.get("@graph"):
                exp = data["@graph"][0]
                found_marks.append(mark)
                result.annotations.append(Annotation(
                    start=1, end=result.length,
                    type=f"encode_experiment_{mark}",
                    category="chromatin_state",
                    label=f"ENCODE {mark} ChIP-seq available ({exp.get('accession')})",
                    metadata={
                        "encode_accession": exp.get("accession"),
                        "histone_mark": mark,
                        "biosample": "K562",
                        "note": "Download bigWig/peak files for position-specific signal",
                    },
                ))
        except Exception:
            continue

    # --- 3. Infer chromatin state from Ensembl regulatory features ---
    state_rules = {
        "promoter": "Active_Promoter",
        "enhancer": "Enhancer",
        "ctcf_binding_site": "CTCF_Insulator",
        "open_chromatin_region": "Open_Chromatin",
        "promoter_flanking_region": "Flanking_Promoter",
        "tf_binding_site": "TF_Binding",
    }
    for ann in list(result.annotations):
        if ann.category == "chromatin_state" and ann.type.startswith("regulatory_"):
            for key, state in state_rules.items():
                if key in ann.type:
                    result.annotations.append(Annotation(
                        start=ann.start, end=ann.end,
                        type=f"chromatin_state_{state}",
                        category="chromatin_state",
                        label=f"Inferred state: {state}",
                        metadata={"inferred_from": ann.metadata.get("regulatory_id")},
                    ))
                    break

    return result


def main():
    print("Fetching chromatin state annotations...")
    result = annotate(DEFAULT_REGION)
    print(result.summary())
    print(f"\nSequence (first 80bp): {result.sequence[:80]}...")
    with open("04_output.json", "w") as f:
        f.write(result.to_json())
    print("Saved to 04_output.json")


if __name__ == "__main__":
    main()
