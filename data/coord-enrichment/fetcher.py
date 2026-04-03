"""
CoordFetcher — parse genomic coordinates from text and fetch compact annotations.

For each coordinate (UCSC or Ensembl format) found in a document, returns a
compact annotation string to be inserted inline, e.g.:

  chr17:43,094,000-43,094,500
  [gene=BRCA1 (protein_coding) | seq=CTGATGTAGGTCTCCTTT... | strand=- | assembly=GRCh38]

Uses:
  - common.py (synth-data) for sequence fetching (local FASTA → NCBI → Ensembl)
  - Ensembl REST /overlap/region for gene lookup

One CoordFetcher instance is shared per Slurm task; results are cached in memory
so the same coordinate in multiple documents only triggers one API call.
"""

import os
import re
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Import common.py from sibling synth-data directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SYNTH_DATA = os.path.normpath(os.path.join(_HERE, '..', 'synth-data'))
if os.path.isdir(_SYNTH_DATA) and _SYNTH_DATA not in sys.path:
    sys.path.insert(0, _SYNTH_DATA)
from common import GenomicRegion, fetch_sequence, ensembl_get   # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coordinate patterns (same as GenomicCoordFilter in scan_genomic_coords.py)
# ---------------------------------------------------------------------------
UCSC_PAT = re.compile(
    r'\bchr(?:[1-9]|1[0-9]|2[0-2]|X|Y|M):\d[\d,]{4,}[-\u2013]\d[\d,]{4,}\b'
)
ENSEMBL_PAT = re.compile(
    r'\b(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT):[1-9]\d{5,}[-\u2013][1-9]\d{5,}\b'
)

# Maximum sequence snippet length inserted into text
SEQ_SNIPPET_LEN = 60

# Maximum region size to attempt sequence fetch (skip huge spans > 2 Mb)
MAX_REGION_BP = 2_000_000


def _parse_coord(coord_str: str) -> tuple[str, int, int] | None:
    """
    Parse a raw coordinate string into (chrom, start, end).
    Handles UCSC (chr17:43,044,295-43,170,245) and Ensembl (17:43044295-43170245).
    Returns None if parsing fails.
    """
    # Normalise: remove commas, replace en-dash with hyphen
    norm = coord_str.replace(',', '').replace('\u2013', '-')
    # Split on first colon
    parts = norm.split(':', 1)
    if len(parts) != 2:
        return None
    chrom_raw, range_part = parts
    # Split range
    dash_parts = range_part.split('-', 1)
    if len(dash_parts) != 2:
        return None
    try:
        start = int(dash_parts[0])
        end   = int(dash_parts[1])
    except ValueError:
        return None
    if start >= end:
        return None
    # Normalise chrom: Ensembl uses bare numbers, we store without 'chr' prefix
    chrom = chrom_raw.lstrip('chr') if chrom_raw.lower().startswith('chr') else chrom_raw
    return chrom, start, end


def _fetch_gene(chrom: str, start: int, end: int) -> dict | None:
    """
    Query Ensembl /overlap/region for genes overlapping the region.
    Returns the most relevant gene dict or None.
    Priority: protein_coding > lncRNA > any other biotype.
    """
    url = (
        f"https://rest.ensembl.org/overlap/region/human/"
        f"{chrom}:{start}-{end}"
    )
    params = {"feature": "gene", "content-type": "application/json"}
    try:
        resp = ensembl_get(url, params, timeout=15)
        if resp.status_code != 200:
            return None
        genes = resp.json()
        if not genes:
            return None
        # Sort: protein_coding first, then by overlap length descending
        def priority(g):
            biotype = g.get('biotype', '')
            bp = 2 if biotype == 'protein_coding' else (1 if biotype == 'lncRNA' else 0)
            overlap = min(g.get('end', end), end) - max(g.get('start', start), start)
            return (bp, overlap)
        genes.sort(key=priority, reverse=True)
        return genes[0]
    except Exception as e:
        logger.debug("Gene fetch failed for %s:%d-%d: %s", chrom, start, end, e)
        return None


def _build_compact(chrom: str, start: int, end: int,
                   gene: dict | None, sequence: str | None) -> str:
    """
    Build the compact annotation string inserted after a coordinate in text.

    Format:
      [gene=BRCA1 (protein_coding) | seq=CTGATGTAGGT...(60bp) | strand=- | assembly=GRCh38]

    If no gene found: gene=intergenic
    If sequence unavailable: seq field is omitted.
    """
    parts = []

    if gene:
        name    = gene.get('external_name') or gene.get('id', 'unknown')
        biotype = gene.get('biotype', 'unknown')
        strand  = '-' if gene.get('strand', 1) == -1 else '+'
        parts.append(f"gene={name} ({biotype})")
        parts.append(f"strand={strand}")
    else:
        parts.append("gene=intergenic")

    if sequence:
        snippet = sequence[:SEQ_SNIPPET_LEN]
        parts.append(f"seq={snippet}({'...' if len(sequence) > SEQ_SNIPPET_LEN else str(len(sequence)) + 'bp'})")

    parts.append("assembly=GRCh38")

    return " [" + " | ".join(parts) + "]"


class CoordFetcher:
    """
    Per-worker fetcher with in-memory cache and concurrent prefetching.

    For each call to enrich_text(), all unique uncached coordinates are
    fetched in parallel via a ThreadPoolExecutor (max_workers=4), so
    documents with multiple coordinates don't pay sequential API latency.

    Usage:
        fetcher = CoordFetcher()
        enriched_text = fetcher.enrich_text(original_text)
    """

    # Concurrent API calls per worker. Keep low enough that 4 parallel Slurm
    # tasks (4 × 4 = 16 threads) stay within Ensembl's 15 req/s per IP limit.
    MAX_WORKERS = 4

    def __init__(self):
        # coord_str → compact annotation string (or "" if fetch failed)
        self._cache: dict[str, str] = {}
        self._hits   = 0
        self._misses = 0
        self._errors = 0

    def _get_annotation(self, coord_str: str) -> str:
        """Return compact annotation for a coordinate string, using cache."""
        if coord_str in self._cache:
            self._hits += 1
            return self._cache[coord_str]

        self._misses += 1
        parsed = _parse_coord(coord_str)
        if parsed is None:
            self._cache[coord_str] = ""
            return ""

        chrom, start, end = parsed
        region_size = end - start

        # Skip unreasonably large regions (sequence fetch would be too big)
        if region_size > MAX_REGION_BP:
            self._cache[coord_str] = ""
            return ""

        # Fetch gene (single Ensembl API call)
        gene = _fetch_gene(chrom, start, end)

        # Fetch sequence — use a window of at most SEQ_SNIPPET_LEN * 4 bp
        # from the start of the region (no need to pull megabases for a snippet)
        sequence = None
        fetch_end = min(end, start + SEQ_SNIPPET_LEN * 4)
        try:
            region = GenomicRegion(
                chrom=chrom,
                start=start,
                end=fetch_end,
                assembly="GRCh38",
            )
            sequence = fetch_sequence(region)
        except Exception as e:
            logger.debug("Sequence fetch failed for %s:%d-%d: %s", chrom, start, end, e)
            self._errors += 1

        annotation = _build_compact(chrom, start, end, gene, sequence)
        self._cache[coord_str] = annotation
        return annotation

    def _prefetch(self, coord_strs: list) -> None:
        """Fetch all uncached coordinates concurrently."""
        uncached = [c for c in coord_strs if c not in self._cache]
        if not uncached:
            return
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            futures = {pool.submit(self._get_annotation, c): c for c in uncached}
            for _ in as_completed(futures):
                pass  # results already stored in self._cache by _get_annotation

    def enrich_text(self, text: str) -> str:
        """
        Find all genomic coordinates in text and insert a compact annotation
        immediately after each occurrence.

        All unique uncached coordinates are fetched concurrently before
        substitution so documents with multiple coords pay parallel latency.
        """
        # Collect all matches
        matches = []
        for pat in (UCSC_PAT, ENSEMBL_PAT):
            for m in pat.finditer(text):
                matches.append(m)

        if not matches:
            return text

        # De-duplicate spans
        seen_spans: set[tuple[int, int]] = set()
        unique_matches = []
        for m in matches:
            span = (m.start(), m.end())
            if span not in seen_spans:
                seen_spans.add(span)
                unique_matches.append(m)

        # Prefetch all unique coords in parallel before substituting
        self._prefetch([m.group() for m in unique_matches])

        # Insert annotations right-to-left so earlier offsets stay valid
        unique_matches.sort(key=lambda m: m.end(), reverse=True)
        result = text
        for m in unique_matches:
            coord_str = m.group()
            annotation = self._get_annotation(coord_str)
            if annotation:
                result = result[:m.end()] + annotation + result[m.end():]

        return result

    def stats(self) -> dict:
        return {
            "cache_size":  len(self._cache),
            "cache_hits":  self._hits,
            "api_calls":   self._misses,
            "fetch_errors": self._errors,
        }
