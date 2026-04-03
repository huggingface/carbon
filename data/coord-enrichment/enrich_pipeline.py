"""
Coordinate Enrichment Pipeline

Reads the genomic-coord-matched documents from processed_snowflake_genomic_coords/,
fetches compact biological annotations for every coordinate reference in each
document, inserts the annotation inline immediately after the coordinate, and
writes the enriched documents to a new dataset:

  processed_snowflake_enriched/

Example transformation inside a document
-----------------------------------------
Before:
  "...the locus chr17:43,094,000-43,094,500 harbours a critical exon..."

After:
  "...the locus chr17:43,094,000-43,094,500 [gene=BRCA1 (protein_coding) |
   seq=CTGATGTAGGTCTCCTTT... | strand=- | assembly=GRCh38] harbours a critical exon..."

No LLMs are used.  All annotations come from:
  - Ensembl REST API (/overlap/region) for gene metadata
  - Sequence fetching via common.py (local FASTA → NCBI Entrez → Ensembl REST)

Rate limiting
-------------
Ensembl enforces 15 requests/second per client IP.  To stay within this limit
each subset uses a small number of tasks (see SUBSETS below).  The CoordFetcher
caches results per coordinate, so repeated references to the same locus within
a task only trigger one API call.

Usage
-----
  python enrich_pipeline.py --run biorxiv
  python enrich_pipeline.py --run biostars
  python enrich_pipeline.py --run finepdfs
  python enrich_pipeline.py --run pipeline_t2
  python enrich_pipeline.py --run pubmed
  python enrich_pipeline.py --run all
"""

import os
import sys
import argparse

# Make fetcher.py importable both when running locally and when the pickled
# pipeline is loaded by launch_pickled_pipeline on a worker node.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader

# CoordFetcher is imported lazily inside _init_fetcher() so the pickled
# pipeline resolves it correctly on the worker node at runtime.


# =============================================================================
# CONFIGURATION
# =============================================================================

# Scan output dir — both snowflake and regex subsets are written here by scan_genomic_coords.py
BASE_INPUT_SNOWFLAKE = "/fsx/dana_aubakirova/carbon_data/clean/processed_snowflake_genomic_coords"
BASE_INPUT_REGEX     = "/fsx/dana_aubakirova/carbon_data/clean/processed_snowflake_genomic_coords"  # same dir, different subset names
# Output: new enriched datasets
BASE_OUTPUT_SNOWFLAKE = "/fsx/dana_aubakirova/carbon_data/clean/processed_snowflake_enriched"
BASE_OUTPUT_REGEX     = "/fsx/dana_aubakirova/carbon_data/clean/processed_regex_enriched"
BASE_LOGS             = "/fsx/dana_aubakirova/carbon_project/logs"

# Tasks kept low to respect Ensembl's 15 req/s per IP.
# Each task fetches ~(docs_in_shard × avg_coords_per_doc) API calls.
SUBSETS = {
    # ── processed_snowflake ──────────────────────────────────────────────────
    "biorxiv": {
        "name":        "biorxiv_meca_ml_20260225",
        "base_input":  BASE_INPUT_SNOWFLAKE,
        "base_output": BASE_OUTPUT_SNOWFLAKE,
        "tasks":       4,
        "time":        "02:00:00",
    },
    "biostars": {
        "name":        "biostars_ml_20260225",
        "base_input":  BASE_INPUT_SNOWFLAKE,
        "base_output": BASE_OUTPUT_SNOWFLAKE,
        "tasks":       8,
        "time":        "04:00:00",
    },
    "finepdfs": {
        "name":        "finepdfs_bio_filtered_20260303",
        "base_input":  BASE_INPUT_SNOWFLAKE,
        "base_output": BASE_OUTPUT_SNOWFLAKE,
        "tasks":       2,
        "time":        "01:00:00",
    },
    "pipeline_t2": {
        "name":        "snowflake_pipeline_t2_20260217",
        "base_input":  BASE_INPUT_SNOWFLAKE,
        "base_output": BASE_OUTPUT_SNOWFLAKE,
        "tasks":       8,
        "time":        "03:00:00",
    },
    "pubmed": {
        "name":        "snowflake_pubmed_20260218",
        "base_input":  BASE_INPUT_SNOWFLAKE,
        "base_output": BASE_OUTPUT_SNOWFLAKE,
        "tasks":       5,
        "time":        "01:00:00",
    },
    # ── processed_regex ──────────────────────────────────────────────────────
    "regex_biorxiv": {
        "name":        "biorxiv_meca_regex_20260225",
        "base_input":  BASE_INPUT_REGEX,
        "base_output": BASE_OUTPUT_REGEX,
        "tasks":       4,
        "time":        "02:00:00",
    },
    "regex_biostars": {
        "name":        "biostars_regex_20260225",
        "base_input":  BASE_INPUT_REGEX,
        "base_output": BASE_OUTPUT_REGEX,
        "tasks":       8,
        "time":        "04:00:00",
    },
    "regex_finepdfs": {
        "name":        "finepdfs_regex_outliers_removed_20260303",
        "base_input":  BASE_INPUT_REGEX,
        "base_output": BASE_OUTPUT_REGEX,
        "tasks":       4,
        "time":        "06:00:00",
    },
    "regex_fineweb": {
        "name":        "fineweb_regex_snowflake_filtered_20260303",
        "base_input":  BASE_INPUT_REGEX,
        "base_output": BASE_OUTPUT_REGEX,
        "tasks":       4,
        "time":        "02:00:00",
    },
    "regex_pubmed": {
        "name":        "pubmed_regex_snowflake_filtered_20260303",
        "base_input":  BASE_INPUT_REGEX,
        "base_output": BASE_OUTPUT_REGEX,
        "tasks":       8,
        "time":        "03:00:00",
    },
}


# =============================================================================
# ENRICHMENT STEP (always-pass filter that modifies document text)
# =============================================================================

class CoordEnrichmentStep(BaseFilter):
    """
    Reads every document from the coord-matched input, inserts compact
    biological annotations after each genomic coordinate in the text,
    and passes all documents through (never drops any).

    One CoordFetcher (with its cache) is created per Slurm task so API
    results are reused across documents within the same task.
    """
    name = "🧬 CoordEnrichmentStep"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        self._fetcher: CoordFetcher | None = None
        self._total    = 0
        self._enriched = 0
        self._coords_inserted = 0

    def _init_fetcher(self):
        """Lazy init so the fetcher is created inside the worker process.
        Import is done here (not at module level) so that the pickled pipeline
        resolves fetcher.py correctly on the worker node at runtime regardless
        of the worker's working directory.

        NOTE: os/sys must be imported locally here.  When the pipeline is
        pickled and re-loaded by launch_pickled_pipeline the class __module__
        is '__main__' (the launcher), so module-level globals from
        enrich_pipeline.py (including 'os') are not in scope.
        """
        if self._fetcher is None:
            import os as _os
            import sys as _sys
            _coord_dir = '/fsx/dana_aubakirova/carbon_project/carbon/data/coord-enrichment'
            if _coord_dir not in _sys.path:
                _sys.path.insert(0, _coord_dir)
            from fetcher import CoordFetcher
            self._fetcher = CoordFetcher()

    def filter(self, doc: Document) -> bool:
        self._init_fetcher()
        self._total += 1

        original_text = doc.text
        enriched_text = self._fetcher.enrich_text(original_text)

        if enriched_text != original_text:
            self._enriched += 1
            # Count inserted annotations by comparing lengths isn't trivial,
            # so track via metadata flag instead
            doc.metadata["coord_enriched"] = True

        doc.text = enriched_text

        if self._total % 500 == 0:
            self._print_stats()

        # Always keep the document
        return True

    def _print_stats(self):
        fetcher_stats = self._fetcher.stats() if self._fetcher else {}
        print(
            f"\n[CoordEnrichmentStep] {self._total:,} docs processed | "
            f"{self._enriched:,} enriched "
            f"({100*self._enriched/self._total:.1f}%) | "
            f"api_calls={fetcher_stats.get('api_calls', 0)} | "
            f"cache_hits={fetcher_stats.get('cache_hits', 0)} | "
            f"errors={fetcher_stats.get('fetch_errors', 0)}"
        )


# =============================================================================
# PIPELINE FACTORY
# =============================================================================

def make_executor(subset_key: str) -> SlurmPipelineExecutor:
    cfg         = SUBSETS[subset_key]
    subset_name = cfg["name"]
    n_tasks     = cfg["tasks"]

    input_dir  = f"{cfg['base_input']}/{subset_name}"
    output_dir = f"{cfg['base_output']}/{subset_name}"
    log_dir    = f"{BASE_LOGS}/enriched_{subset_name}"

    return SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=input_dir,
                glob_pattern="*.jsonl.gz",
                text_key="text",
                id_key="id",
                recursive=False,
            ),
            CoordEnrichmentStep(),
            JsonlWriter(
                output_folder=output_dir,
                max_file_size=int(500e6),
                compression="gzip",
            ),
        ],
        tasks=n_tasks,
        workers=n_tasks,
        time=cfg["time"],
        partition="hopper-cpu",
        logging_dir=log_dir,
        cpus_per_task=2,
        mem_per_cpu_gb=8,
        qos="normal",
        job_name=f"enrich_{subset_key}",
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich processed_snowflake_genomic_coords with inline gene annotations"
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
        input_dir  = f"{cfg['base_input']}/{cfg['name']}"
        output_dir = f"{cfg['base_output']}/{cfg['name']}"

        # Check input exists and has data
        if not os.path.isdir(input_dir):
            print(f"  ⚠  Skipping {cfg['name']} — input not ready yet: {input_dir}")
            continue

        print(f"\nSubmitting: {cfg['name']}  ({cfg['tasks']} tasks)")
        print(f"  Input : {input_dir}")
        print(f"  Output: {output_dir}")
        make_executor(key).run()
        print(f"  ✓ Job submitted")
