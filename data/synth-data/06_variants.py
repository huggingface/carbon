#!/usr/bin/env python3
"""
Variant Annotations
===================
Annotates: known SNPs, clinical significance, consequence predictions.

Data sources:
  - Ensembl REST API — known variants in region + VEP predictions
  - Local — codon change analysis

Produces annotations with category="variants".
"""

import requests
from common import (
    GenomicRegion, Annotation, AnnotatedSequence, DEFAULT_REGION,
    ENSEMBL_REST, make_annotated_sequence, clamp_to_region, genomic_to_local, ensembl_get,
)


def annotate(region: GenomicRegion) -> AnnotatedSequence:
    """Fetch sequence + variant annotations for a genomic region."""
    result = make_annotated_sequence(region)

    # --- 1. Ensembl: known variants in region ---
    url = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
    params = {"feature": "variation", "content-type": "application/json"}
    resp = ensembl_get(url, params=params)
    resp.raise_for_status()

    variants = resp.json()

    for var in variants:
        local_start, local_end = clamp_to_region(var["start"], var["end"], region)
        if local_start == 0:
            continue

        clin_sig = var.get("clinical_significance", [])
        consequence = var.get("consequence_type", "unknown")

        result.annotations.append(Annotation(
            start=local_start, end=local_end,
            type="variant", category="variants",
            label=f"{var.get('id', 'unknown')} ({consequence})",
            strand=var.get("strand", "."),
            metadata={
                "rs_id": var.get("id"),
                "alleles": var.get("alleles"),
                "consequence": consequence,
                "clinical_significance": clin_sig,
            },
        ))

        # Separate annotation for clinical variants
        if clin_sig:
            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type="clinical_variant", category="variants",
                label=f"{var.get('id')} clinical: {', '.join(clin_sig)}",
                metadata={
                    "rs_id": var.get("id"),
                    "clinical_significance": clin_sig,
                },
            ))

    # --- 2. VEP prediction for a sample variant (first variant found) ---
    if variants:
        sample_var = variants[0]
        try:
            pos = sample_var["start"]
            alleles = sample_var.get("alleles", "").split("/")
            if len(alleles) >= 2:
                alt = alleles[1]
                vep_url = f"{ENSEMBL_REST}/vep/human/region/{region.ensembl_chrom}:{pos}-{pos}:1/{alt}"
                vep_params = {"content-type": "application/json"}
                vep_resp = ensembl_get(vep_url, params=vep_params)
                if vep_resp.ok:
                    local_pos = genomic_to_local(pos, region)
                    for entry in vep_resp.json():
                        for tc in entry.get("transcript_consequences", [])[:1]:
                            result.annotations.append(Annotation(
                                start=local_pos, end=local_pos,
                                type="vep_prediction", category="variants",
                                label=f"VEP: {', '.join(tc.get('consequence_terms', []))} "
                                      f"(impact={tc.get('impact', 'N/A')})",
                                metadata={
                                    "gene": tc.get("gene_symbol"),
                                    "consequence_terms": tc.get("consequence_terms"),
                                    "impact": tc.get("impact"),
                                    "sift": tc.get("sift_prediction"),
                                    "polyphen": tc.get("polyphen_prediction"),
                                },
                            ))
        except Exception:
            pass

    return result


def main():
    print("Fetching variant annotations...")
    result = annotate(DEFAULT_REGION)
    print(result.summary())
    print(f"\nSequence (first 80bp): {result.sequence[:80]}...")
    with open("06_output.json", "w") as f:
        f.write(result.to_json())
    print("Saved to 06_output.json")


if __name__ == "__main__":
    main()
