#!/usr/bin/env python3
"""
Gene Structure Annotations
==========================
Annotates: exons, introns, UTRs, start/stop codons, splice sites, CDS.

Data sources:
  - Ensembl REST API (overlap endpoint for gene features)
  - Local ORF/codon scanning on the fetched sequence

Produces annotations with category="gene_structure".
"""

import re
import requests
from common import (
    GenomicRegion, Annotation, AnnotatedSequence, DEFAULT_REGION,
    ENSEMBL_REST, make_annotated_sequence, clamp_to_region, ensembl_get,
)


def annotate(region: GenomicRegion) -> AnnotatedSequence:
    """Fetch sequence + gene structure annotations for a genomic region."""
    result = make_annotated_sequence(region)

    # --- Ensembl overlap: gene, transcript, exon, CDS features ---
    url = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
    params = {
        "feature": ["gene", "transcript", "exon", "cds"],
        "content-type": "application/json",
    }
    resp = ensembl_get(url, params=params)
    resp.raise_for_status()

    for feat in resp.json():
        feat_type = feat.get("feature_type", feat.get("object_type", "")).lower()
        local_start, local_end = clamp_to_region(feat["start"], feat["end"], region)
        if local_start == 0:
            continue

        strand = "+" if feat.get("strand", 1) >= 0 else "-"
        gene_name = feat.get("external_name") or feat.get("id", "")

        if feat_type == "gene":
            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type="gene", category="gene_structure",
                label=f"Gene {gene_name} ({feat.get('biotype', '')})",
                strand=strand,
                metadata={"gene_id": feat.get("id"), "biotype": feat.get("biotype")},
            ))
        elif feat_type == "transcript":
            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type="transcript", category="gene_structure",
                label=f"Transcript {feat.get('id', '')} ({feat.get('biotype', '')})",
                strand=strand,
                metadata={"transcript_id": feat.get("id"), "biotype": feat.get("biotype")},
            ))
        elif feat_type == "exon":
            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type="exon", category="gene_structure",
                label=f"Exon {feat.get('id', '')}",
                strand=strand,
                metadata={"exon_id": feat.get("id")},
            ))
        elif feat_type == "cds":
            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type="CDS", category="gene_structure",
                label=f"CDS ({gene_name})",
                strand=strand,
                metadata={"protein_id": feat.get("protein_id", "")},
            ))

    # --- Local: scan for start/stop codons and splice signals ---
    seq = result.sequence
    for m in re.finditer("ATG", seq):
        result.annotations.append(Annotation(
            start=m.start() + 1, end=m.end(),
            type="start_codon", category="gene_structure",
            label="ATG start codon",
        ))
    for m in re.finditer("T(?:AA|AG|GA)", seq):
        result.annotations.append(Annotation(
            start=m.start() + 1, end=m.end(),
            type="stop_codon", category="gene_structure",
            label=f"{m.group()} stop codon",
        ))
    # Splice donor (GT)
    for m in re.finditer("GT[AG]AGT", seq):
        result.annotations.append(Annotation(
            start=m.start() + 1, end=m.start() + 2,
            type="splice_donor", category="gene_structure",
            label="GT splice donor consensus",
        ))
    # Splice acceptor (polypyrimidine tract + AG)
    for m in re.finditer("[CT]{4,}AG", seq):
        result.annotations.append(Annotation(
            start=m.end() - 1, end=m.end(),
            type="splice_acceptor", category="gene_structure",
            label="AG splice acceptor with polypyrimidine tract",
        ))

    return result


def main():
    print("Fetching gene structure annotations...")
    result = annotate(DEFAULT_REGION)
    print(result.summary())
    print(f"\nSequence (first 80bp): {result.sequence[:80]}...")
    print(f"\nFull JSON output saved to 01_output.json")
    with open("01_output.json", "w") as f:
        f.write(result.to_json())


if __name__ == "__main__":
    main()
