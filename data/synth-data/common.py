"""
Shared data model and utilities for DNA sequence annotation scripts.

All scripts produce output in a uniform format:

    AnnotatedSequence:
        sequence: str           — raw DNA (5'→3', uppercase)
        region: GenomicRegion   — where this sequence lives in the genome
        annotations: list[Annotation]  — structured annotations

    Annotation:
        start: int              — 1-based position in the sequence
        end: int                — 1-based, inclusive
        type: str               — specific annotation type (e.g. "exon", "H3K4me3_peak")
        category: str           — one of the 10 annotation categories (see CATEGORIES)
        label: str              — human-readable label
        strand: str             — "+", "-", or "."
        score: float | None     — optional numeric score
        metadata: dict          — extra key-value pairs (database IDs, etc.)
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json
import os
import random
import requests

ENSEMBL_REST = "https://rest.ensembl.org"

# --- Local reference genome path ---
# Set REFERENCE_FASTA env var or place the file in the scripts directory.
# Download once:
#   wget https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
#   gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
#   samtools faidx Homo_sapiens.GRCh38.dna.primary_assembly.fa
#
# Or just chromosome 17 for testing (~85 MB):
#   wget https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.17.fa.gz
#   gunzip Homo_sapiens.GRCh38.dna.chromosome.17.fa.gz
#   samtools faidx Homo_sapiens.GRCh38.dna.chromosome.17.fa
REFERENCE_FASTA = os.environ.get("REFERENCE_FASTA", "")

# The 10 annotation categories (matching the 10 scripts)
CATEGORIES = [
    "gene_structure",         # 01: exons, introns, UTRs, codons, splice sites
    "regulatory_elements",    # 02: promoters, TATA box, TFBS, restriction sites, CpG islands
    "functional_class",       # 03: coding, non-coding, regulatory classification
    "chromatin_state",        # 04: histone marks, ChromHMM states
    "conservation",           # 05: PhyloP, PhastCons, constrained elements
    "variants",               # 06: SNPs, clinical significance, VEP consequences
    "expression_epigenetic",  # 07: DNase, histone peaks, methylation, expression
    "repeats",                # 08: Alu, LINE, simple repeats, transposons
    "ncrna",                  # 09: miRNA targets, lncRNA, snoRNA, hairpins
    "disease_clinical",       # 10: pathogenic variants, GWAS, disease associations
]


@dataclass
class GenomicRegion:
    chrom: str          # e.g. "chr17" or "17"
    start: int          # 1-based genomic start
    end: int            # 1-based genomic end (inclusive)
    assembly: str       # e.g. "GRCh38"
    strand: str = "+"

    @property
    def ensembl_chrom(self) -> str:
        """Ensembl uses bare chromosome names (no 'chr' prefix)."""
        return self.chrom.replace("chr", "")

    @property
    def ucsc_chrom(self) -> str:
        """UCSC uses 'chr' prefix."""
        c = self.chrom
        return c if c.startswith("chr") else f"chr{c}"


@dataclass
class Annotation:
    start: int                      # 1-based position in the LOCAL sequence
    end: int                        # 1-based, inclusive
    type: str                       # specific type (e.g. "exon", "TATA_box", "H3K4me3_peak")
    category: str                   # one of CATEGORIES
    label: str                      # human-readable description
    strand: str = "."               # "+", "-", or "."
    score: Optional[float] = None   # optional numeric score
    metadata: dict = field(default_factory=dict)


@dataclass
class AnnotatedSequence:
    sequence: str                   # raw DNA, uppercase
    region: GenomicRegion
    annotations: list[Annotation] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Print a human-readable summary."""
        lines = [
            f"Region: {self.region.ucsc_chrom}:{self.region.start}-{self.region.end} "
            f"({self.region.assembly})",
            f"Sequence length: {self.length} bp",
            f"Annotations: {len(self.annotations)}",
        ]
        # Group by category
        by_cat = {}
        for ann in self.annotations:
            by_cat.setdefault(ann.category, []).append(ann)
        for cat in CATEGORIES:
            if cat in by_cat:
                lines.append(f"  {cat}: {len(by_cat[cat])} annotations")
                for ann in by_cat[cat][:5]:
                    score_str = f" score={ann.score}" if ann.score is not None else ""
                    lines.append(f"    {ann.start}-{ann.end} {ann.type}: {ann.label}{score_str}")
                if len(by_cat[cat]) > 5:
                    lines.append(f"    ... and {len(by_cat[cat]) - 5} more")
        return "\n".join(lines)


def genomic_to_local(genomic_pos: int, region: GenomicRegion) -> int:
    """Convert a genomic coordinate to a 1-based local sequence position."""
    return genomic_pos - region.start + 1


def local_to_genomic(local_pos: int, region: GenomicRegion) -> int:
    """Convert a 1-based local position to a genomic coordinate."""
    return local_pos + region.start - 1


def clamp_to_region(start: int, end: int, region: GenomicRegion) -> tuple[int, int]:
    """
    Convert genomic start/end to local coordinates, clamped to the region.
    Returns (local_start, local_end) both 1-based inclusive, or (0, 0) if no overlap.
    """
    # Clamp to region boundaries
    clamped_start = max(start, region.start)
    clamped_end = min(end, region.end)

    if clamped_start > clamped_end:
        return (0, 0)

    local_start = genomic_to_local(clamped_start, region)
    local_end = genomic_to_local(clamped_end, region)
    return (local_start, local_end)


# ---------------------------------------------------------------------------
# HTTP with retry (Ensembl rate-limits at 15 req/s)
# ---------------------------------------------------------------------------

def ensembl_get(url: str, params: dict = None, timeout: int = 30) -> requests.Response:
    """GET with retry on 429/5xx, respecting Retry-After header."""
    import time
    params = params or {}
    if "content-type" not in params:
        params["content-type"] = "application/json"

    for attempt in range(3):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = float(resp.headers.get("Retry-After", 2 * (attempt + 1)))
            time.sleep(wait)
            continue
        return resp

    return resp  # return last response even if failed


# ---------------------------------------------------------------------------
# Sequence fetching
# ---------------------------------------------------------------------------

def _fetch_sequence_local(region: GenomicRegion) -> str:
    """Fetch sequence from a local indexed FASTA (samtools faidx / pysam)."""
    fasta_path = REFERENCE_FASTA

    # Try pysam first (pure Python, no subprocess)
    try:
        import pysam
        fa = pysam.FastaFile(fasta_path)
        # Try both "17" and "chr17" naming conventions
        for chrom in [region.ensembl_chrom, region.ucsc_chrom]:
            if chrom in fa.references:
                seq = fa.fetch(chrom, region.start - 1, region.end)  # pysam is 0-based
                fa.close()
                return seq.upper()
        fa.close()
    except ImportError:
        pass

    # Fallback: samtools faidx (requires samtools in PATH)
    import subprocess
    for chrom in [region.ensembl_chrom, region.ucsc_chrom]:
        try:
            result = subprocess.run(
                ["samtools", "faidx", fasta_path, f"{chrom}:{region.start}-{region.end}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                seq = "".join(lines[1:])  # skip header line
                if seq:
                    return seq.upper()
        except FileNotFoundError:
            break  # samtools not installed

    raise RuntimeError(f"Could not fetch sequence from {fasta_path} for {region.ensembl_chrom}:{region.start}-{region.end}")


def _fetch_sequence_ncbi(region: GenomicRegion) -> str:
    """Fetch sequence from NCBI Entrez efetch (different rate limit than Ensembl)."""
    from Bio import Entrez, SeqIO
    Entrez.email = "user@example.com"

    # GRCh38 RefSeq chromosome accessions
    chrom_accessions = {
        "1": "NC_000001.11", "2": "NC_000002.12", "3": "NC_000003.12",
        "4": "NC_000004.12", "5": "NC_000005.10", "6": "NC_000006.12",
        "7": "NC_000007.14", "8": "NC_000008.11", "9": "NC_000009.12",
        "10": "NC_000010.11", "11": "NC_000011.10", "12": "NC_000012.12",
        "13": "NC_000013.11", "14": "NC_000014.9", "15": "NC_000015.10",
        "16": "NC_000016.10", "17": "NC_000017.11", "18": "NC_000018.10",
        "19": "NC_000019.10", "20": "NC_000020.11", "21": "NC_000021.9",
        "22": "NC_000022.11", "X": "NC_000023.11", "Y": "NC_000024.10",
    }
    chrom = region.ensembl_chrom
    accession = chrom_accessions.get(chrom)
    if not accession:
        raise ValueError(f"Unknown chromosome: {chrom}")

    handle = Entrez.efetch(
        db="nucleotide", id=accession,
        rettype="fasta", retmode="text",
        seq_start=region.start, seq_stop=region.end,
    )
    record = SeqIO.read(handle, "fasta")
    handle.close()
    return str(record.seq).upper()


def _fetch_sequence_ensembl(region: GenomicRegion) -> str:
    """Fetch sequence from Ensembl REST API."""
    chrom = region.ensembl_chrom
    url = f"{ENSEMBL_REST}/sequence/region/human/{chrom}:{region.start}..{region.end}"
    resp = ensembl_get(url)
    resp.raise_for_status()
    return resp.json()["seq"].upper()


def fetch_sequence(region: GenomicRegion) -> str:
    """
    Fetch DNA sequence using the best available method:
      1. Local FASTA (if REFERENCE_FASTA is set) — instant, no network
      2. NCBI Entrez (if biopython installed) — separate rate limit from Ensembl
      3. Ensembl REST API — fallback

    Returns uppercase DNA string.
    """
    # 1. Local FASTA
    if REFERENCE_FASTA and Path(REFERENCE_FASTA).exists():
        try:
            return _fetch_sequence_local(region)
        except Exception:
            pass

    # 2. NCBI Entrez
    try:
        return _fetch_sequence_ncbi(region)
    except Exception:
        pass

    # 3. Ensembl REST
    return _fetch_sequence_ensembl(region)


_sequence_cache: dict[str, str] = {}


def make_annotated_sequence(region: GenomicRegion) -> AnnotatedSequence:
    """Create an AnnotatedSequence by fetching the DNA from Ensembl (cached)."""
    cache_key = f"{region.chrom}:{region.start}-{region.end}"
    if cache_key not in _sequence_cache:
        _sequence_cache[cache_key] = fetch_sequence(region)
    return AnnotatedSequence(sequence=_sequence_cache[cache_key], region=region)


# ---------------------------------------------------------------------------
# Gene dataclass and flanking logic
# ---------------------------------------------------------------------------

# Biotypes we consider "interesting" for the gene-centric pipeline
GENE_BIOTYPES = {
    # Protein-coding
    "protein_coding",
    # Long non-coding RNA
    "lncRNA", "lincRNA", "antisense", "sense_intronic", "sense_overlapping",
    "processed_transcript", "3prime_overlapping_ncRNA",
    # Small functional RNA
    "miRNA", "tRNA", "snoRNA", "snRNA", "rRNA", "scRNA", "scaRNA",
    "ribozyme", "vaultRNA",
    # Other
    "misc_RNA",
}

# Maximum region size before we split into chunks (to avoid huge API responses)
MAX_REGION_SIZE = 100_000  # 100kb


@dataclass
class Gene:
    gene_id: str        # e.g. "ENSG00000012048"
    name: str           # e.g. "BRCA1"
    chrom: str          # e.g. "17"
    start: int          # 1-based genomic start (always start < end)
    end: int            # 1-based genomic end
    strand: str         # "+" or "-"
    biotype: str        # e.g. "protein_coding", "lncRNA", "miRNA"
    assembly: str = "GRCh38"

    def to_region(self, upstream_flank: int = 0, downstream_flank: int = 0) -> GenomicRegion:
        """
        Convert gene to a GenomicRegion with flanking sequence.
        Upstream/downstream are relative to transcription direction:
          - upstream = before TSS (5' of gene on its strand)
          - downstream = after gene end (3' of gene on its strand)
        """
        if self.strand == "+":
            region_start = max(1, self.start - upstream_flank)
            region_end = self.end + downstream_flank
        else:  # minus strand: TSS is at self.end
            region_start = max(1, self.start - downstream_flank)
            region_end = self.end + upstream_flank

        return GenomicRegion(
            chrom=self.chrom,
            start=region_start,
            end=region_end,
            assembly=self.assembly,
            strand=self.strand,
        )


def sample_flanking(upstream_max: int = 4000, downstream_max: int = 3000,
                    seed: int = None) -> tuple[int, int]:
    """
    Sample random flanking distances using Beta distributions.

    Upstream:   Beta(6, 2) × upstream_max   → mean ≈ 3kb, mode ≈ 3.3kb
    Downstream: Beta(4, 2) × downstream_max → mean ≈ 2kb, mode ≈ 2kb

    Returns (upstream_bp, downstream_bp) as integers.
    """
    rng = random.Random(seed)
    upstream = int(rng.betavariate(6, 2) * upstream_max)
    downstream = int(rng.betavariate(4, 2) * downstream_max)
    return upstream, downstream


# ---------------------------------------------------------------------------
# Default demo region (BRCA1 exon 10 region, well-annotated)
# ---------------------------------------------------------------------------

DEFAULT_REGION = GenomicRegion(
    chrom="17",
    start=43094000,
    end=43094500,
    assembly="GRCh38",
    strand="+",
)
