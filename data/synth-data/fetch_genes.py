#!/usr/bin/env python3
"""
Fetch all genes for a species from Ensembl BioMart.

Retrieves protein-coding genes, lncRNAs, miRNAs, tRNAs, snoRNAs, and other
functional RNAs. Outputs a JSON gene list ready for the annotation pipeline.

Usage:
    python fetch_genes.py                        # human, all gene types
    python fetch_genes.py --species mouse        # mouse
    python fetch_genes.py --biotype protein_coding --biotype lncRNA  # specific types

Output: genes_{species}.json
"""

import argparse
import json
import sys
import requests
from common import Gene, GENE_BIOTYPES, ensembl_get, ENSEMBL_REST


# Ensembl BioMart XML template
BIOMART_URL = "https://www.ensembl.org/biomart/martservice"

BIOMART_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1"
       uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="{dataset}" interface="default">
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="external_gene_name"/>
    <Attribute name="chromosome_name"/>
    <Attribute name="start_position"/>
    <Attribute name="end_position"/>
    <Attribute name="strand"/>
    <Attribute name="gene_biotype"/>
  </Dataset>
</Query>"""

# Species to BioMart dataset mapping
SPECIES_DATASETS = {
    "human": "hsapiens_gene_ensembl",
    "mouse": "mmusculus_gene_ensembl",
    "rat": "rnorvegicus_gene_ensembl",
    "zebrafish": "drerio_gene_ensembl",
    "chicken": "ggallus_gene_ensembl",
    "fly": "dmelanogaster_gene_ensembl",
    "worm": "celegans_gene_ensembl",
    "yeast": "scerevisiae_gene_ensembl",
    "dog": "cfamiliaris_gene_ensembl",
    "pig": "sscrofa_gene_ensembl",
    "cow": "btaurus_gene_ensembl",
    "macaque": "mmulatta_gene_ensembl",
    "frog": "xtropicalis_gene_ensembl",
}

SPECIES_ASSEMBLIES = {
    "human": "GRCh38",
    "mouse": "GRCm39",
    "rat": "mRatBN7.2",
    "zebrafish": "GRCz11",
    "chicken": "GRCg7b",
    "fly": "BDGP6.46",
    "worm": "WBcel235",
    "yeast": "R64-1-1",
    "dog": "ROS_Cfam_1.0",
    "pig": "Sscrofa11.1",
    "cow": "ARS-UCD1.3",
    "macaque": "Mmul_10",
    "frog": "UCB_Xtro_10.0",
}

# Valid chromosome names (skip patches, scaffolds, etc.)
VALID_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}


def fetch_genes_biomart(species: str = "human",
                        biotypes: set[str] = None) -> list[Gene]:
    """Fetch genes from Ensembl BioMart."""
    dataset = SPECIES_DATASETS.get(species)
    if not dataset:
        raise ValueError(f"Unknown species: {species}. Available: {list(SPECIES_DATASETS)}")

    assembly = SPECIES_ASSEMBLIES.get(species, "unknown")
    biotypes = biotypes or GENE_BIOTYPES

    xml = BIOMART_XML.format(dataset=dataset)

    print(f"Fetching genes from Ensembl BioMart ({dataset})...", flush=True)
    resp = requests.get(BIOMART_URL, params={"query": xml}, timeout=120)
    resp.raise_for_status()

    genes = []
    lines = resp.text.strip().split("\n")
    header = lines[0].split("\t")

    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 7:
            continue

        gene_id = fields[0]
        name = fields[1] or gene_id
        chrom = fields[2]
        start = int(fields[3])
        end = int(fields[4])
        strand = "+" if fields[5] == "1" else "-"
        biotype = fields[6]

        # Filter by biotype
        if biotype not in biotypes:
            continue

        # Filter to main chromosomes (skip scaffolds/patches)
        if species == "human" and chrom not in VALID_CHROMS:
            continue

        genes.append(Gene(
            gene_id=gene_id,
            name=name,
            chrom=chrom,
            start=start,
            end=end,
            strand=strand,
            biotype=biotype,
            assembly=assembly,
        ))

    return genes


def fetch_genes_rest(species: str = "human",
                     biotypes: set[str] = None) -> list[Gene]:
    """Fallback: fetch genes from Ensembl REST API (slower, chromosome by chromosome)."""
    assembly = SPECIES_ASSEMBLIES.get(species, "unknown")
    biotypes = biotypes or GENE_BIOTYPES
    ens_species = {
        "human": "human", "mouse": "mouse", "rat": "rat",
        "zebrafish": "zebrafish", "chicken": "chicken",
        "fly": "drosophila_melanogaster", "worm": "caenorhabditis_elegans",
        "yeast": "saccharomyces_cerevisiae",
    }.get(species, species)

    genes = []
    chroms = [str(i) for i in range(1, 23)] + ["X", "Y"]

    for chrom in chroms:
        print(f"  Fetching chr{chrom}...", flush=True)
        url = f"{ENSEMBL_REST}/overlap/region/{ens_species}/{chrom}:1-300000000"
        params = {"feature": "gene", "content-type": "application/json"}
        resp = ensembl_get(url, params=params)
        if not resp.ok:
            continue

        for feat in resp.json():
            biotype = feat.get("biotype", "")
            if biotype not in biotypes:
                continue

            genes.append(Gene(
                gene_id=feat.get("id", ""),
                name=feat.get("external_name", "") or feat.get("id", ""),
                chrom=chrom,
                start=feat.get("start", 0),
                end=feat.get("end", 0),
                strand="+" if feat.get("strand", 1) >= 0 else "-",
                biotype=biotype,
                assembly=assembly,
            ))

    return genes


def fetch_genes_gtf(gtf_path: str, biotypes: set[str] = None,
                    assembly: str = "GRCh38") -> list[Gene]:
    """
    Parse genes from a local GENCODE/Ensembl GTF file (fastest, most reliable).

    Download:
      wget https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz
      # or for mouse:
      wget https://ftp.ensembl.org/pub/release-113/gtf/mus_musculus/Mus_musculus.GRCm39.113.gtf.gz
    """
    import gzip

    biotypes = biotypes or GENE_BIOTYPES
    opener = gzip.open if gtf_path.endswith(".gz") else open
    genes = []
    seen = set()

    print(f"Parsing GTF: {gtf_path}...", flush=True)
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue

            attrs = {}
            for attr in fields[8].split(";"):
                attr = attr.strip()
                if " " in attr:
                    key, val = attr.split(" ", 1)
                    attrs[key] = val.strip('"')

            gene_id = attrs.get("gene_id", "")
            if gene_id in seen:
                continue
            seen.add(gene_id)

            biotype = attrs.get("gene_biotype", "")
            if biotype not in biotypes:
                continue

            chrom = fields[0]
            # Skip scaffolds/patches
            if chrom not in VALID_CHROMS:
                continue

            genes.append(Gene(
                gene_id=gene_id,
                name=attrs.get("gene_name", gene_id),
                chrom=chrom,
                start=int(fields[3]),
                end=int(fields[4]),
                strand=fields[6],
                biotype=biotype,
                assembly=assembly,
            ))

    return genes


def main():
    parser = argparse.ArgumentParser(description="Fetch gene list from Ensembl")
    parser.add_argument("--species", default="human",
                        choices=list(SPECIES_DATASETS.keys()),
                        help="Species to fetch genes for")
    parser.add_argument("--biotype", action="append", default=None,
                        help="Filter to specific biotypes (can repeat). Default: all functional types.")
    parser.add_argument("--gtf", default=None,
                        help="Local GTF file (fastest). Download from Ensembl FTP.")
    parser.add_argument("--output", default=None,
                        help="Output JSON file. Default: genes_{species}.json")
    args = parser.parse_args()

    biotypes = set(args.biotype) if args.biotype else GENE_BIOTYPES

    # Try local GTF first, then BioMart, then REST
    if args.gtf:
        assembly = SPECIES_ASSEMBLIES.get(args.species, "unknown")
        genes = fetch_genes_gtf(args.gtf, biotypes, assembly)
    else:
        try:
            genes = fetch_genes_biomart(args.species, biotypes)
            if not genes:
                raise RuntimeError("BioMart returned 0 genes")
        except Exception as e:
            print(f"BioMart failed ({e}), falling back to REST API...")
            genes = fetch_genes_rest(args.species, biotypes)

    # Sort by chromosome then position
    chrom_order = {str(i): i for i in range(1, 23)}
    chrom_order.update({"X": 23, "Y": 24, "MT": 25})
    genes.sort(key=lambda g: (chrom_order.get(g.chrom, 99), g.start))

    # Summary
    from collections import Counter
    biotype_counts = Counter(g.biotype for g in genes)

    print(f"\nTotal genes: {len(genes)}")
    print("By biotype:")
    for bt, count in biotype_counts.most_common():
        print(f"  {bt}: {count}")

    # Gene size stats
    sizes = [g.end - g.start + 1 for g in genes]
    sizes.sort()
    print(f"\nGene sizes:")
    print(f"  Min: {min(sizes):,} bp")
    print(f"  Median: {sizes[len(sizes)//2]:,} bp")
    print(f"  Mean: {sum(sizes)//len(sizes):,} bp")
    print(f"  Max: {max(sizes):,} bp")
    print(f"  Total: {sum(sizes)/1e6:.0f} Mb")

    # Save
    output = args.output or f"genes_{args.species}.json"
    with open(output, "w") as f:
        json.dump([vars(g) for g in genes], f, indent=2)

    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
