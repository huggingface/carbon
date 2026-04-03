"""
Genomic Coordinate Scanner for all processed_snowflake subsets

Scans every processed_snowflake subset for UCSC / Ensembl genomic coordinate
patterns.  For every document that contains at least one coordinate, the
document is kept and enriched with metadata:

  has_genomic_coords : True
  coord_type         : "ucsc" | "ensembl" | "both"
  n_ucsc             : int  – number of UCSC matches
  n_ensembl          : int  – number of Ensembl matches
  n_coords_total     : int  – n_ucsc + n_ensembl

Documents without any coordinate are dropped (not written to output).
Each Slurm task logs a stats block including total docs, match rate, and
per-discrete-score breakdown (scores 2–5) at completion.

Supported coordinate formats
-----------------------------
  UCSC    chr17:43044295-43170245
          chr17:43,044,295-43,170,245    (commas)
          chrX:43,094,000–43,094,500     (en-dash)
  Ensembl 17:43044295-43170245           (≥6 digit positions, i.e. ≥100,000 bp
          MT:110000000-120000000          to exclude journal citation patterns
                                          like "Oncotarget. 7:46042-46055")

Chromosomes: 1-22, X, Y, M / MT (standard human reference names).
One biological keyword is required alongside any coordinate match to
avoid stray numeric ranges (version strings, port numbers, etc.).

Targets
-------
  biorxiv_meca_ml_20260225       20 files →  20 tasks
  biostars_ml_20260225        63122 files → 200 tasks
  finepdfs_bio_filtered_20260303 17 files →  17 tasks
  snowflake_pipeline_t2_20260217160 files → 160 tasks
  snowflake_pubmed_20260218      25 files →  25 tasks

Usage
-----
  python scan_genomic_coords.py --run biorxiv
  python scan_genomic_coords.py --run biostars
  python scan_genomic_coords.py --run finepdfs
  python scan_genomic_coords.py --run pipeline_t2
  python scan_genomic_coords.py --run pubmed
  python scan_genomic_coords.py --run all
"""

import os
import re
import argparse
from collections import Counter

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_INPUT_SNOWFLAKE = "/fsx/dana_aubakirova/carbon_data/clean/processed_snowflake"
BASE_INPUT_REGEX     = "/fsx/dana_aubakirova/carbon_data/clean/processed_regex"
BASE_OUTPUT          = "/fsx/dana_aubakirova/carbon_data/clean/processed_snowflake_genomic_coords"
BASE_LOGS            = "/fsx/dana_aubakirova/carbon_project/logs"

SUBSETS = {
    # ── processed_snowflake ──────────────────────────────────────────────────
    "biorxiv": {
        "name":       "biorxiv_meca_ml_20260225",
        "base_input": BASE_INPUT_SNOWFLAKE,
        "tasks":      20,
    },
    "biostars": {
        "name":       "biostars_ml_20260225",
        "base_input": BASE_INPUT_SNOWFLAKE,
        "tasks":      200,   # ~315 files per task
    },
    "finepdfs": {
        "name":       "finepdfs_bio_filtered_20260303",
        "base_input": BASE_INPUT_SNOWFLAKE,
        "tasks":      17,
    },
    "pipeline_t2": {
        "name":       "snowflake_pipeline_t2_20260217",
        "base_input": BASE_INPUT_SNOWFLAKE,
        "tasks":      160,
    },
    "pubmed": {
        "name":       "snowflake_pubmed_20260218",
        "base_input": BASE_INPUT_SNOWFLAKE,
        "tasks":      25,
    },
    # ── processed_regex ──────────────────────────────────────────────────────
    "regex_biorxiv": {
        "name":       "biorxiv_meca_regex_20260225",
        "base_input": BASE_INPUT_REGEX,
        "tasks":      20,
    },
    "regex_biostars": {
        "name":       "biostars_regex_20260225",
        "base_input": BASE_INPUT_REGEX,
        "tasks":      20,
    },
    "regex_finepdfs": {
        "name":       "finepdfs_regex_outliers_removed_20260303",
        "base_input": BASE_INPUT_REGEX,
        "tasks":      20,
    },
    "regex_fineweb": {
        "name":       "fineweb_regex_snowflake_filtered_20260303",
        "base_input": BASE_INPUT_REGEX,
        "tasks":      20,
    },
    "regex_pubmed": {
        "name":       "pubmed_regex_snowflake_filtered_20260303",
        "base_input": BASE_INPUT_REGEX,
        "tasks":      20,
    },
}


# =============================================================================
# GENOMIC COORDINATE FILTER
# =============================================================================

class GenomicCoordFilter(BaseFilter):
    """
    Keeps documents that contain UCSC or Ensembl genomic coordinates AND at
    least one biological keyword.  Adds coord metadata to every kept document.

    Dropped if:
      - No coordinate pattern matched, OR
      - No biological keyword present (guards against stray numeric ranges)

    Stats logged at task completion:
      - total docs processed, matched, hit rate
      - coord type breakdown (ucsc / ensembl / both)
      - per discrete_score breakdown (scores 2–5)
    """
    name = "🧬 GenomicCoordFilter"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

        # Biological keywords — at least one must be present alongside coords
        self.bio_terms = {
            'dna', 'rna', 'nucleotide', 'genome', 'gene',
            'base pair', 'codon', 'primer', 'pcr', 'amplification',
            'sequencing', 'nucleic acid', 'oligonucleotide', 'plasmid',
            'transcription', 'translation', 'mrna', 'ribosome',
            'trna', 'rrna', 'uracil', 'chromosome', 'locus', 'allele',
            'variant', 'mutation', 'snp', 'crispr', 'exon', 'intron',
            'promoter', 'enhancer', 'genomic', 'region', 'coordinates',
        }

        # UCSC: chr1–chr22, chrX, chrY, chrM
        # Handles plain digits, comma-formatted, regular dash and en-dash (–).
        self.ucsc_pat = re.compile(
            r'\bchr(?:[1-9]|1[0-9]|2[0-2]|X|Y|M):\d[\d,]{4,}[-\u2013]\d[\d,]{4,}\b'
        )
        # Ensembl: 1–22, X, Y, MT
        # Requires ≥6 digits (≥100,000) to exclude journal citation patterns
        # like "Oncotarget. 7:46042-46055" or antibody dilutions "1:1,000-1,500".
        # Ensembl coordinates rarely use comma formatting so no comma variant needed.
        self.ensembl_pat = re.compile(
            r'\b(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT):[1-9]\d{5,}[-\u2013][1-9]\d{5,}\b'
        )

        self._total        = 0
        self._passed       = 0
        self._ucsc_only    = 0
        self._ensembl_only = 0
        self._both         = 0
        self._score_dist: Counter = Counter()   # discrete_score → count

    def filter(self, doc: Document) -> bool:
        self._total += 1
        text = doc.text
        text_lower = text.lower()

        # Require at least one biological keyword
        if not any(term in text_lower for term in self.bio_terms):
            return False

        ucsc_matches    = self.ucsc_pat.findall(text)
        ensembl_matches = self.ensembl_pat.findall(text)

        n_ucsc    = len(ucsc_matches)
        n_ensembl = len(ensembl_matches)

        if n_ucsc == 0 and n_ensembl == 0:
            return False

        if n_ucsc > 0 and n_ensembl > 0:
            coord_type = "both"
            self._both += 1
        elif n_ucsc > 0:
            coord_type = "ucsc"
            self._ucsc_only += 1
        else:
            coord_type = "ensembl"
            self._ensembl_only += 1

        self._passed += 1

        # Track per-score distribution
        score = doc.metadata.get("discrete_score")
        if score is not None:
            self._score_dist[int(score)] += 1

        # Enrich metadata
        doc.metadata["has_genomic_coords"] = True
        doc.metadata["coord_type"]         = coord_type
        doc.metadata["n_ucsc"]             = n_ucsc
        doc.metadata["n_ensembl"]          = n_ensembl
        doc.metadata["n_coords_total"]     = n_ucsc + n_ensembl

        if self._total % 50_000 == 0:
            self._print_stats()

        return True

    def _print_stats(self):
        pct = 100 * self._passed / self._total if self._total else 0.0
        print(
            f"\n[GenomicCoordFilter] {self._total:,} docs | "
            f"{self._passed:,} matched ({pct:.3f}%) | "
            f"ucsc={self._ucsc_only} ensembl={self._ensembl_only} both={self._both}"
        )
        if self._score_dist:
            dist_str = "  ".join(
                f"score{s}={self._score_dist[s]}"
                for s in sorted(self._score_dist)
            )
            print(f"  Score dist: {dist_str}")


# =============================================================================
# PIPELINE FACTORY
# =============================================================================

def make_executor(subset_key: str) -> SlurmPipelineExecutor:
    cfg         = SUBSETS[subset_key]
    subset_name = cfg["name"]
    n_tasks     = cfg["tasks"]

    input_dir  = f"{cfg['base_input']}/{subset_name}"
    output_dir = f"{BASE_OUTPUT}/{subset_name}"
    log_dir    = f"{BASE_LOGS}/genomic_coords_{subset_name}"

    return SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=input_dir,
                glob_pattern="*.jsonl.gz",
                text_key="text",
                id_key="id",
                recursive=False,
            ),
            GenomicCoordFilter(),
            JsonlWriter(
                output_folder=output_dir,
                max_file_size=int(500e6),
                compression="gzip",
            ),
        ],
        tasks=n_tasks,
        workers=n_tasks,
        time="01:00:00",
        partition="hopper-cpu",
        logging_dir=log_dir,
        cpus_per_task=2,
        mem_per_cpu_gb=8,
        qos="normal",
        job_name=f"genomic_coords_{subset_key}",
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan processed_snowflake subsets for genomic coordinates"
    )
    parser.add_argument(
        "--run",
        choices=list(SUBSETS.keys()) + ["all"],
        required=True,
    )
    args = parser.parse_args()

    targets = list(SUBSETS.keys()) if args.run == "all" else [args.run]

    for key in targets:
        cfg = SUBSETS[key]
        print(f"\nSubmitting: {cfg['name']}  ({cfg['tasks']} tasks)")
        print(f"  Input : {cfg['base_input']}/{cfg['name']}")
        print(f"  Output: {BASE_OUTPUT}/{cfg['name']}")
        make_executor(key).run()
        print(f"  ✓ Job submitted")
