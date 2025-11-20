from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, ParquetReader

import json
import re

N_TASKS = 10
N_WORKERS = 10


dataset_name = "HuggingFaceFW/fineweb-edu"
dataset_path = "/fsx/leandro/data/fineweb-edu-10bt/sample/"

output_keywords_folder = f"s3://hf-carbon/fp-edu-keywords"
output_nucleotide_folder = f"s3://hf-carbon/fp-edu-nucleotide"

class BiologyKeywordFilter(BaseFilter):
    name = "🔬 BiologyKeywordFilter"

    def __init__(self, min_keywords: int = 2,  exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        self.bio_terms = {
            'protein', 'gene', 'dna', 'rna', 'genome', 'nucleotide',
            'sequence', 'amino acid', 'mutation', 'chromosome', 'allele',
            'plasmid', 'vector', 'cloning', 'transformation', 'transfection',
            'expression', 'promoter', 'origin of replication', 'antibiotic resistance',
            'insert', 'backbone', 'recombinant', 'conjugation'
        }
        self.min_keywords = min_keywords

    def filter(self, doc: Document) -> bool:
        text = doc.text.lower()
        count = sum(1 for term in self.bio_terms if term in text)
        return count >= self.min_keywords


class NucleotideSequenceFilter(BaseFilter):
    name = "🧬 NucleotideSequenceFilter"

    def __init__(self, min_length: int = 20, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        self.nucleotide_pattern = re.compile(rf'[ATCGN]{{{min_length},}}', re.IGNORECASE)

    def filter(self, doc: Document) -> bool:
        return bool(self.nucleotide_pattern.search(doc.text))


executor_nl = SlurmPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=dataset_path),
            NucleotideSequenceFilter(min_length=10),
            JsonlWriter(output_folder=output_nucleotide_folder, max_file_size=int(10**9))
        ],
        tasks=N_TASKS,
        workers=N_WORKERS,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir="/fsx/leandro/logs/carbon-filter-pdf-nl",
        cpus_per_task=2,
        mem_per_cpu_gb=4,
        qos="normal",
    )

executor_kw = SlurmPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=dataset_path),
            BiologyKeywordFilter(min_keywords=2),
            JsonlWriter(output_folder=output_keywords_folder, max_file_size=int(10**9))
        ],
        tasks=N_TASKS,
        workers=N_WORKERS,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir="/fsx/leandro/logs/carbon-filter-pdf-kw",
        cpus_per_task=2,
        mem_per_cpu_gb=4,
        qos="normal",
    )


executor_nl.run()
executor_kw.run()