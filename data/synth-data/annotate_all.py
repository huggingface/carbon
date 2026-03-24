#!/usr/bin/env python3
"""
Combined annotation pipeline.
Runs all 10 annotation scripts on a genomic region and produces
one AnnotatedSequence with all annotations merged.

Supports two modes:
  1. Region mode:  python annotate_all.py 17 43094000 43094500
  2. Gene mode:    python annotate_all.py --gene BRCA1
                   python annotate_all.py --genes-file genes_human.json --index 0

Gene mode adds randomized flanking (Beta-distributed) around the gene body.
"""

import argparse
import sys
import json
import importlib
from common import (
    GenomicRegion, AnnotatedSequence, Gene, DEFAULT_REGION,
    make_annotated_sequence, sample_flanking, MAX_REGION_SIZE,
)


SCRIPTS = [
    ("01_gene_structure", "gene_structure"),
    ("02_regulatory_elements", "regulatory_elements"),
    ("03_functional_classification", "functional_class"),
    ("04_chromatin_states", "chromatin_state"),
    ("05_conservation", "conservation"),
    ("06_variants", "variants"),
    ("07_expression_epigenetic", "expression_epigenetic"),
    ("08_repetitive_elements", "repeats"),
    ("09_mirna_regulatory_rna", "ncrna"),
    ("10_disease_clinical", "disease_clinical"),
]


def annotate_all(region: GenomicRegion, skip: list[str] = None) -> AnnotatedSequence:
    """Run all annotation scripts and merge results."""
    skip = skip or []

    # Fetch sequence once
    combined = make_annotated_sequence(region)

    for module_name, category in SCRIPTS:
        if category in skip:
            continue

        print(f"  Running {module_name}...", end=" ", flush=True)
        try:
            mod = importlib.import_module(module_name)
            result = mod.annotate(region)

            assert result.sequence == combined.sequence, \
                f"Sequence mismatch in {module_name}"

            combined.annotations.extend(result.annotations)
            print(f"OK ({len(result.annotations)} annotations)")
        except Exception as e:
            print(f"FAILED: {e}")

    return combined


def gene_to_region(gene: Gene, seed: int = None) -> GenomicRegion:
    """Convert a Gene to a GenomicRegion with randomized flanking."""
    upstream, downstream = sample_flanking(seed=seed)
    region = gene.to_region(upstream_flank=upstream, downstream_flank=downstream)

    # If region is too large, cap it (very large genes like DMD/TTN)
    region_size = region.end - region.start + 1
    if region_size > MAX_REGION_SIZE:
        print(f"  Warning: {gene.name} region is {region_size/1000:.0f}kb, "
              f"capping to {MAX_REGION_SIZE/1000:.0f}kb")
        region = GenomicRegion(
            chrom=region.chrom,
            start=region.start,
            end=region.start + MAX_REGION_SIZE - 1,
            assembly=region.assembly,
            strand=region.strand,
        )

    return region


def load_gene(name: str = None, genes_file: str = None, index: int = None) -> Gene:
    """Load a gene by name (from Ensembl REST) or from a gene list file by index."""
    if genes_file and index is not None:
        with open(genes_file) as f:
            genes = json.load(f)
        g = genes[index]
        return Gene(**g)

    if name:
        from common import ensembl_get, ENSEMBL_REST
        url = f"{ENSEMBL_REST}/lookup/symbol/human/{name}"
        resp = ensembl_get(url)
        resp.raise_for_status()
        data = resp.json()
        return Gene(
            gene_id=data["id"],
            name=name,
            chrom=data["seq_region_name"],
            start=data["start"],
            end=data["end"],
            strand="+" if data["strand"] >= 0 else "-",
            biotype=data.get("biotype", "unknown"),
            assembly="GRCh38",
        )

    raise ValueError("Provide --gene NAME or --genes-file FILE --index N")


def main():
    parser = argparse.ArgumentParser(description="Annotate a genomic region or gene")
    parser.add_argument("coords", nargs="*",
                        help="chrom start end (region mode)")
    parser.add_argument("--gene", type=str, default=None,
                        help="Gene symbol (e.g. BRCA1)")
    parser.add_argument("--genes-file", type=str, default=None,
                        help="JSON gene list file from fetch_genes.py")
    parser.add_argument("--index", type=int, default=None,
                        help="Gene index in --genes-file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for flanking (reproducible)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file. Default: combined_output.json or {gene_name}_output.json")
    args = parser.parse_args()

    gene = None

    if args.gene or args.genes_file:
        # Gene mode
        gene = load_gene(name=args.gene, genes_file=args.genes_file, index=args.index)
        region = gene_to_region(gene, seed=args.seed)
        upstream = region.start - (gene.start if gene.strand == "+" else gene.start)
        gene_size = gene.end - gene.start + 1
        region_size = region.end - region.start + 1
        print(f"Gene: {gene.name} ({gene.biotype}) on {gene.strand} strand")
        print(f"Gene body: chr{gene.chrom}:{gene.start}-{gene.end} ({gene_size:,} bp)")
        print(f"With flanking: chr{region.chrom}:{region.start}-{region.end} ({region_size:,} bp)")
        print()
    elif len(args.coords) >= 3:
        # Region mode
        chrom, start, end = args.coords[0], int(args.coords[1]), int(args.coords[2])
        region = GenomicRegion(chrom=chrom, start=start, end=end, assembly="GRCh38")
    else:
        region = DEFAULT_REGION

    print(f"Annotating {region.ucsc_chrom}:{region.start}-{region.end} "
          f"({region.end - region.start + 1:,} bp)\n")

    combined = annotate_all(region)

    # Add gene metadata if in gene mode
    if gene:
        combined_dict = combined.to_dict()
        combined_dict["gene"] = {
            "gene_id": gene.gene_id,
            "name": gene.name,
            "biotype": gene.biotype,
            "strand": gene.strand,
            "gene_start": gene.start,
            "gene_end": gene.end,
        }
    else:
        combined_dict = combined.to_dict()

    print(f"\n{'=' * 60}")
    print(combined.summary())

    output_file = args.output
    if not output_file:
        if gene:
            output_file = f"{gene.name}_output.json"
        else:
            output_file = "combined_output.json"

    with open(output_file, "w") as f:
        json.dump(combined_dict, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")
    print(f"Total annotations: {len(combined.annotations)}")


if __name__ == "__main__":
    main()
