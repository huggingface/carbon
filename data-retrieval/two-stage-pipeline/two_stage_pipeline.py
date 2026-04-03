"""
DNA/RNA Sequence Filters for FineWeb-EDU Dataset

This module contains filters for identifying biological sequence content:
- DNAFilterV4Strict: High-precision regex filter (15+ bases, 3+ unique, keywords)
- DNAFilterV5Expanded: Extended filter with categorization and relaxed requirements
- GenomicCoordFilter: Captures documents referencing specific genomic regions via
  UCSC (chr17:43,044,295-43,170,245) or Ensembl (17:43044295-43170245) coordinates

Each filter can be run independently on the FineWeb-EDU dataset.
"""

import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader, HuggingFaceDatasetReader

import re
import sys
import json
from datetime import datetime

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_timestamped_log_dir(base_name: str) -> str:
    """
    Generate a unique timestamped logging directory to avoid datatrove caching issues.
    
    Args:
        base_name: Base name for the log directory
        
    Returns:
        Path with timestamp appended
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"/fsx/dana_aubakirova/carbon_project/logs/{base_name}_{timestamp}"

# =============================================================================
# CONFIGURATION
# =============================================================================
# NOTE: For parallel processing with datatrove, sharded parquet files must be 
# organized in subdirectories (one file per subdirectory) when using recursive=True.
# Example structure: data_dir/shard_00/shard_00.parquet, data_dir/shard_01/shard_01.parquet, etc.
# This allows datatrove to properly distribute tasks across the shards.

dataset_path_10bt = "/fsx/leandro/data/fineweb-edu-10bt/sample/10BT"
dataset_path_full = "/fsx/dana_aubakirova/.cache/datasets--HuggingFaceFW--fineweb-edu/snapshots/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/data"
dataset_path_pubmed = "/fsx/dana_aubakirova/carbon_project/data/pubmed"
dataset_path_finepdf_edu_v4 = "/fsx/dana_aubakirova/carbon_project/data/fineweb-edu-v4"
dataset_path_finepdfs_edu_eng = "/fsx/dana_aubakirova/carbon_project/data/finepdfs-edu/eng_Latn"
dataset_path_finepdfs_eng = "/fsx/dana_aubakirova/carbon_project/data/finepdfs/eng_Latn"

# =============================================================================
# V4 STRICT REGEX FILTER
# =============================================================================

class DNAFilterV4Strict(BaseFilter):
    """
    Strict regex filter for high-confidence DNA/RNA sequences.

    Requirements:
    - 15+ base sequences
    - 3+ unique bases
    - Biological keywords present
    - Not in LaTeX braces
    """
    name = "🧬 DNAFilterV4Strict"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        self.bio_terms = {
            'dna', 'rna', 'nucleotide', 'genome', 'gene',
            'base pair', 'codon', 'primer', 'pcr', 'amplification',
            'sequencing', 'nucleic acid', 'oligonucleotide', 'plasmid',
            'transcription', 'translation', 'mrna', 'ribosome',
            'trna', 'rrna', 'uracil'
        }
        self.min_keywords = 1
        self.nucleotide_pattern = re.compile(r'[ATCGUN]{15,}', re.IGNORECASE)

    def _has_mixed_bases(self, seq: str, min_unique: int = 3) -> bool:
        """Check if sequence contains at least min_unique different bases."""
        seq_upper = seq.upper()
        unique_bases = set(c for c in seq_upper if c in 'ATCGUN')
        return len(unique_bases) >= min_unique

    def _is_in_braces(self, text: str, match_start: int, match_end: int) -> bool:
        """Check if the match is inside curly braces (likely LaTeX)."""
        before = text[max(0, match_start - 100):match_start]
        after = text[match_end:min(len(text), match_end + 100)]
        open_braces = before.count('{') - before.count('}')
        if open_braces > 0 and '}' in after:
            return True
        return False

    def filter(self, doc: Document) -> bool:
        text = doc.text
        text_lower = text.lower()

        keyword_count = sum(1 for term in self.bio_terms if term in text_lower)
        if keyword_count < self.min_keywords:
            return False

        matches = list(self.nucleotide_pattern.finditer(text))
        if not matches:
            return False

        for match in matches:
            seq = match.group()
            if not self._has_mixed_bases(seq):
                continue
            if self._is_in_braces(text, match.start(), match.end()):
                continue

            doc.metadata["filter_type"] = "strict_regex"
            doc.metadata["stage"] = 1
            return True

        return False


# =============================================================================
# GENOMIC COORDINATE FILTER
# =============================================================================

class GenomicCoordFilter(BaseFilter):
    """
    Filter for documents that reference specific genomic regions via UCSC or
    Ensembl coordinate notation.

    These documents discuss a particular locus (e.g. chr17:43,044,295-43,170,245)
    without necessarily containing raw DNA sequences — a distinct signal from
    DNAFilterV4Strict.  Coordinates are much more specific than gene-name matching,
    which can match general-biology text that has no actionable sequence content.

    Supported formats
    -----------------
    UCSC:    chr17:43044295-43170245
             chr17:43,044,295-43,170,245   (comma-formatted)
             chr17:43,094,000–43,094,500   (en-dash)

    Ensembl: 17:43044295-43170245
             17:43,044,295-43,170,245

    Chromosomes: 1-22, X, Y, M / MT (human; common reference names).

    Requirements
    ------------
    - At least one UCSC or Ensembl coordinate match.
    - At least one biological keyword (same set as DNAFilterV4Strict) to avoid
      catching version numbers, IP-like strings, or other numeric ranges.

    Metadata added on pass
    ----------------------
    filter_type      : "genomic_coords"
    coord_type       : "ucsc" | "ensembl" | "both"
    genomic_coord_count : total number of coordinate matches found
    """
    name = "🧬 GenomicCoordFilter"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

        self.bio_terms = {
            'dna', 'rna', 'nucleotide', 'genome', 'gene',
            'base pair', 'codon', 'primer', 'pcr', 'amplification',
            'sequencing', 'nucleic acid', 'oligonucleotide', 'plasmid',
            'transcription', 'translation', 'mrna', 'ribosome',
            'trna', 'rrna', 'uracil', 'chromosome', 'locus', 'allele',
            'variant', 'mutation', 'snp', 'crispr', 'exon', 'intron',
            'promoter', 'enhancer', 'genomic', 'region', 'coordinates',
        }

        # UCSC: chr1-chr22, chrX, chrY, chrM  (with optional comma formatting
        # and both regular dash and en-dash as range separator)
        self.ucsc_pattern = re.compile(
            r'\bchr(?:[1-9]|1[0-9]|2[0-2]|X|Y|M):\d[\d,]{4,}[-\u2013]\d[\d,]{4,}\b'
        )
        # Ensembl: 1-22, X, Y, MT
        # Requires ≥6 digits (≥100,000) to exclude journal citation patterns
        # like "Oncotarget. 7:46042-46055" or antibody dilutions "1:1,000-1,500".
        # Ensembl coordinates rarely use comma formatting so no comma variant needed.
        self.ensembl_pattern = re.compile(
            r'\b(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT):[1-9]\d{5,}[-\u2013][1-9]\d{5,}\b'
        )

        self.filter_stats = {'total': 0, 'passed': 0, 'ucsc_only': 0,
                             'ensembl_only': 0, 'both': 0}

    def filter(self, doc: Document) -> bool:
        self.filter_stats['total'] += 1
        text = doc.text
        text_lower = text.lower()

        keyword_count = sum(1 for term in self.bio_terms if term in text_lower)
        if keyword_count < 1:
            return False

        ucsc_matches = self.ucsc_pattern.findall(text)
        ensembl_matches = self.ensembl_pattern.findall(text)

        has_ucsc = len(ucsc_matches) > 0
        has_ensembl = len(ensembl_matches) > 0

        if not has_ucsc and not has_ensembl:
            return False

        if has_ucsc and has_ensembl:
            coord_type = "both"
            self.filter_stats['both'] += 1
        elif has_ucsc:
            coord_type = "ucsc"
            self.filter_stats['ucsc_only'] += 1
        else:
            coord_type = "ensembl"
            self.filter_stats['ensembl_only'] += 1

        self.filter_stats['passed'] += 1
        doc.metadata["filter_type"] = "genomic_coords"
        doc.metadata["coord_type"] = coord_type
        doc.metadata["genomic_coord_count"] = len(ucsc_matches) + len(ensembl_matches)
        return True


# =============================================================================
# V5 EXPANDED REGEX FILTER WITH CATEGORIZATION
# =============================================================================

class DNAFilterV5Expanded(BaseFilter):
    """
    Expanded regex filter with relaxed requirements and document categorization.

    Categories:
    - A (bio_text): Molecular biology descriptions WITHOUT explicit sequences
    - B (bio_sequence): Explicit sequences with minimal explanation
    - C (interleaved): Sequences interleaved with natural language explanation

    Exclusions:
    - General medicine/clinical content
    - High-level neuroscience without molecular detail
    - Generic biology without molecular manipulation
    - Dictionary/vocabulary/SEO pages
    """
    name = "🧬 DNAFilterV5Expanded"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

        self.strong_mol_bio_terms = {
            'dna', 'rna', 'nucleotide', 'genome', 'plasmid', 'vector',
            'primer', 'pcr', 'qpcr', 'rt-pcr', 'amplification', 'amplicon',
            'oligonucleotide', 'nucleic acid', 'double helix',
            'codon', 'anticodon', 'base pair', 'base pairs',
            'exon', 'intron', 'promoter', 'enhancer', 'terminator',
            'open reading frame', 'orf', 'utr', "5' utr", "3' utr",
            'mrna', 'trna', 'rrna', 'sirna', 'mirna', 'snrna', 'lncrna', 'cdna',
            'crispr', 'cas9', 'cas12', 'guide rna', 'sgrna',
            'transgene', 'transgenic', 'knockout', 'knockin', 'knock-out', 'knock-in',
            'mutagenesis', 'site-directed mutagenesis',
            'recombinant', 'cloning', 'subcloning', 'ligation',
            'restriction enzyme', 'restriction site', 'endonuclease',
            'dna polymerase', 'rna polymerase', 'reverse transcriptase',
            'ligase', 'helicase', 'topoisomerase',
            'gel electrophoresis', 'agarose gel', 'southern blot', 'northern blot',
            'sequencing', 'sanger sequencing', 'next-generation sequencing', 'ngs',
            'illumina', 'nanopore', 'pacbio',
            'genbank', 'ncbi', 'fasta', 'fastq', 'blast', 'refseq',
            'uniprot', 'ensembl', 'embl',
        }

        self.supporting_terms = {
            'gene', 'chromosome', 'allele', 'locus', 'loci',
            'transcription', 'translation', 'expression',
            'mutation', 'polymorphism', 'snp', 'variant',
            'insertion', 'deletion', 'substitution', 'frameshift',
            'transformation', 'transfection', 'transduction',
            'bacterial', 'e. coli', 'yeast', 'mammalian cells',
            'antibiotic resistance', 'selection marker', 'reporter gene',
            'construct', 'cassette', 'backbone',
            'upstream', 'downstream', 'flanking',
            'hybridization', 'annealing', 'denaturation',
            'replication', 'origin of replication',
        }

        self.context_terms = {
            'protocol', 'procedure', 'method', 'reagent', 'buffer',
            'incubation', 'centrifugation', 'purification', 'extraction',
            'nanodrop', 'spectrophotometer', 'thermocycler',
            'petri dish', 'eppendorf', 'microcentrifuge',
            'stock solution', 'working concentration',
        }

        self.model_organisms = {
            'e. coli', 'escherichia coli', 'saccharomyces cerevisiae',
            'drosophila', 'c. elegans', 'caenorhabditis elegans',
            'arabidopsis', 'mus musculus', 'danio rerio', 'zebrafish',
            'xenopus', 'hela cells', 'hek293', 'cho cells',
        }

        self.clinical_exclusion_terms = {
            'patient', 'patients', 'clinical trial', 'diagnosis',
            'symptoms', 'treatment plan', 'prognosis', 'dosage',
            'medication', 'prescription', 'side effects',
            'hospital', 'physician', 'nursing', 'healthcare',
            'blood pressure', 'heart rate', 'body mass index',
        }

        self.generic_exclusion_terms = {
            'definition of', 'what is a', 'glossary', 'vocabulary',
            'quiz', 'flashcard', 'study guide', 'test prep',
            'click here', 'subscribe', 'newsletter', 'advertisement',
        }

        self.false_positive_indicators = {
            'motor sequencing', 'task sequencing', 'event sequencing',
            'story sequencing', 'number sequencing', 'sequencing activities',
            'sequencing worksheet', 'sequencing cards',
            'vector graphics', 'vector image', 'vector art',
            'vector space', 'vector field', 'vector calculus',
            'attack vector', 'vector addition', 'unit vector',
            'human cloning ban', 'cloning debate', 'reproductive cloning ethics',
            'cloning legislation', 'anti-cloning',
            "it's in our dna", "in the dna of", "part of our dna",
            "company dna", "brand dna", "cultural dna",
            'origin of species', 'natural selection', 'survival of the fittest',
            'darwin', 'evolution debate', 'creationism',
        }

        self.nucleotide_pattern = re.compile(r'[ATCGUN]{12,}', re.IGNORECASE)

        self.protein_pattern = re.compile(
            r'\b[ACDEFGHIKLMNPQRSTVWY]{8,}\b'
        )

        self.labeled_seq_patterns = [
            re.compile(r'(forward|reverse|fwd|rev)\s*(primer|oligo)\s*[:\-]?\s*[ATCGUN]{6,}', re.IGNORECASE),
            re.compile(r'(primer|oligo|probe)\s*\d*\s*[:\-]\s*[ATCGUN]{6,}', re.IGNORECASE),
            re.compile(r'sequence\s*[:\-#]?\s*[ATCGUN]{8,}', re.IGNORECASE),
            re.compile(r"5['\u2032]\s*-?\s*[ATCGUN]{6,}\s*-?\s*3['\u2032]", re.IGNORECASE),
        ]

        self.mutation_patterns = [
            re.compile(r'\b[ACGT]\d+[ACGT]\b'),
            re.compile(r'\bc\.\d+[ACGT]>[ACGT]\b', re.IGNORECASE),
            re.compile(r'\bp\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}\b'),
            re.compile(r'\b(del|ins|dup)\d+', re.IGNORECASE),
        ]

        self.bio_patterns = {
            'dir_seq': re.compile(r"5['\u2032]\s*[ACGTUN]{5,}\s*3['\u2032]", re.IGNORECASE),
            'dir_arrow': re.compile(r"5['\u2032]\s*(to|→|->|—)\s*3['\u2032]", re.IGNORECASE),
            'end_notation': re.compile(r"(5['\u2032]|3['\u2032])\s*(end|terminus|primer|overhang)", re.IGNORECASE),
            'bp_units': re.compile(r"\b\d+\.?\d*\s*(bp|kb|kbp|mb|gbp|nt|nucleotides)\b", re.IGNORECASE),
            'restriction_sites': re.compile(r"\b(EcoRI|BamHI|HindIII|XhoI|NotI|SalI|XbaI|PstI|SmaI|KpnI|SacI|NcoI|NdeI|BglII|ClaI|SpeI|ApaI|MluI|NheI|AgeI)\b", re.IGNORECASE),
            'primer_notation': re.compile(r"(forward|reverse|fwd|rev|sense|antisense)\s*(primer|oligo)", re.IGNORECASE),
            'accession': re.compile(r"\b([A-Z]{1,2}_?\d{5,9}(\.\d+)?)\b"),
            'codon_notation': re.compile(r"\b(start codon|stop codon|ATG|TAA|TAG|TGA|AUG|UAA|UAG|UGA)\b", re.IGNORECASE),
        }

        self.filter_stats = {
            'total': 0,
            'passed': 0,
            'category_A': 0,
            'category_B': 0,
            'category_C': 0,
            'excluded_clinical': 0,
            'excluded_generic': 0,
        }

    def _has_mixed_bases(self, seq: str, min_unique: int = 2) -> bool:
        """Check if sequence has at least min_unique different bases."""
        seq_upper = seq.upper()
        unique_bases = set(c for c in seq_upper if c in 'ATCGUN')
        return len(unique_bases) >= min_unique

    def _is_in_braces(self, text: str, match_start: int, match_end: int) -> bool:
        """Check if match is inside curly braces (LaTeX)."""
        before = text[max(0, match_start - 100):match_start]
        after = text[match_end:min(len(text), match_end + 100)]
        open_braces = before.count('{') - before.count('}')
        return open_braces > 0 and '}' in after

    def _count_terms(self, text_lower: str, term_set: set) -> int:
        """Count how many terms from a set appear in text."""
        return sum(1 for term in term_set if term in text_lower)

    def _has_valid_sequences(self, text: str) -> tuple:
        """
        Check for valid biological sequences.
        Returns (has_sequences, sequence_count, total_seq_length)
        """
        valid_seqs = []

        for match in self.nucleotide_pattern.finditer(text):
            seq = match.group()
            if not self._has_mixed_bases(seq, min_unique=3):
                continue
            if self._is_in_braces(text, match.start(), match.end()):
                continue
            valid_seqs.append(seq)

        if self.bio_patterns['dir_seq'].search(text):
            valid_seqs.append('directional')

        total_length = sum(len(s) for s in valid_seqs if s != 'directional')
        return len(valid_seqs) > 0, len(valid_seqs), total_length

    def _has_bio_patterns(self, text: str) -> dict:
        """Check for alternative biology patterns."""
        results = {}
        for name, pattern in self.bio_patterns.items():
            matches = pattern.findall(text)
            results[name] = len(matches) if isinstance(matches, list) else (1 if matches else 0)
        return results

    def _has_high_precision_terms(self, text_lower: str) -> bool:
        """Check for high-precision molecular biology terms."""
        high_precision_terms = {
            'pcr', 'qpcr', 'rt-pcr', 'gel electrophoresis', 'agarose gel',
            'southern blot', 'northern blot', 'western blot',
            'sanger sequencing', 'next-generation sequencing', 'ngs',
            'crispr', 'cas9', 'cas12', 'guide rna', 'sgrna',
            'plasmid', 'vector backbone', 'cloning vector',
            'restriction enzyme', 'restriction digest', 'ligation',
            'site-directed mutagenesis', 'knock-out', 'knock-in',
            'transgenic', 'transfection', 'transformation',
            'primer design', 'forward primer', 'reverse primer',
            'open reading frame', 'orf', 'start codon', 'stop codon',
            "5' utr", "3' utr", 'promoter region', 'enhancer region',
            'genbank', 'fasta format', 'fastq', 'refseq',
            'dna polymerase', 'rna polymerase', 'reverse transcriptase',
            'restriction endonuclease', 'ligase', 'helicase',
        }

        matches = sum(1 for term in high_precision_terms if term in text_lower)
        return matches >= 2

    def _categorize_document(self, _text: str, _text_lower: str,
                            _has_sequences: bool, _seq_count: int, _seq_length: int,
                            _strong_count: int, _supporting_count: int,
                            _bio_pattern_hits: int) -> str:
        """Categorize document - V5d only returns Category C (interleaved)."""
        return 'C'

    def filter(self, doc: Document) -> bool:
        """Filter and categorize documents."""
        self.filter_stats['total'] += 1
        text = doc.text
        text_lower = text.lower()
        word_count = len(text.split())

        clinical_count = self._count_terms(text_lower, self.clinical_exclusion_terms)
        if clinical_count >= 3:
            self.filter_stats['excluded_clinical'] += 1
            return False

        generic_count = self._count_terms(text_lower, self.generic_exclusion_terms)
        if generic_count >= 2:
            self.filter_stats['excluded_generic'] += 1
            return False

        fp_count = self._count_terms(text_lower, self.false_positive_indicators)
        if fp_count >= 1:
            self.filter_stats['excluded_false_positive_context'] = \
                self.filter_stats.get('excluded_false_positive_context', 0) + 1
            return False

        strong_count = self._count_terms(text_lower, self.strong_mol_bio_terms)
        supporting_count = self._count_terms(text_lower, self.supporting_terms)
        context_count = self._count_terms(text_lower, self.context_terms)
        organism_count = self._count_terms(text_lower, self.model_organisms)

        has_sequences, seq_count, seq_length = self._has_valid_sequences(text)

        bio_patterns = self._has_bio_patterns(text)
        bio_pattern_hits = sum(1 for v in bio_patterns.values() if v > 0)

        passed = False

        if has_sequences and strong_count >= 2:
            passed = True

        elif has_sequences and strong_count >= 1 and bio_pattern_hits >= 1:
            passed = True

        elif has_sequences and (strong_count >= 1 or supporting_count >= 3):
            passed = True

        if not passed:
            return False

        category = self._categorize_document(
            text, text_lower, has_sequences, seq_count, seq_length,
            strong_count, supporting_count, bio_pattern_hits
        )

        self.filter_stats['passed'] += 1
        self.filter_stats[f'category_{category}'] += 1

        doc.metadata['filter_type'] = 'v5_expanded'
        doc.metadata['category'] = category
        doc.metadata['strong_keywords'] = strong_count
        doc.metadata['supporting_keywords'] = supporting_count
        doc.metadata['has_sequences'] = has_sequences
        doc.metadata['sequence_count'] = seq_count
        doc.metadata['sequence_length'] = seq_length
        doc.metadata['bio_pattern_hits'] = bio_pattern_hits

        if self.filter_stats['total'] % 10000 == 0:
            self._print_stats()

        return True

    def _print_stats(self):
        """Print current statistics."""
        total = self.filter_stats['total']
        passed = self.filter_stats['passed']
        print(f"\n{'='*60}")
        print(f"DNAFilterV5Expanded STATISTICS - {total:,} docs processed")
        print(f"{'='*60}")
        print(f"Passed: {passed:,} ({100*passed/total:.3f}%)")
        print(f"  Category A (bio_text):     {self.filter_stats['category_A']:,}")
        print(f"  Category B (bio_sequence): {self.filter_stats['category_B']:,}")
        print(f"  Category C (interleaved):  {self.filter_stats['category_C']:,}")
        print(f"Excluded:")
        print(f"  Clinical content: {self.filter_stats['excluded_clinical']:,}")
        print(f"  Generic/SEO:      {self.filter_stats['excluded_generic']:,}")
        print(f"{'='*60}\n")


# =============================================================================
# LLM CLASSIFIER (DistilBERT)
# =============================================================================

class LLMClassifier(BaseFilter):
    """
    LLM-based classifier using fine-tuned DistilBERT for DNA/RNA content detection.
    
    This filter uses a transformer model (DistilBERT) that was fine-tuned on
    biological sequence data to classify documents.
    """
    name = "🤖 LLMClassifier"

    def __init__(
        self,
        model_path: str = "/fsx/dana_aubakirova/carbon_project/models/finetuned_final",
        threshold: float = 0.9,
        exclusion_writer: DiskWriter = None,
        log_stats: bool = True
    ):
        super().__init__(exclusion_writer)

        self.threshold = threshold
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self.log_stats = log_stats

        self.total_docs = 0
        self.passed = 0
        self.failed = 0

    def _ensure_model_loaded(self):
        """Load model once per worker process"""
        if self._model_loaded:
            return

        classifier_path = '/fsx/dana_aubakirova/carbon_project/carbon/classifier'
        if classifier_path not in sys.path:
            sys.path.insert(0, classifier_path)

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            print(f"Loading DistilBERT model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()

            import torch
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✓ Model loaded on GPU")
            else:
                print("✓ Model loaded on CPU")

            self._model_loaded = True
            print(f"✓ Worker loaded DistilBERT classifier with threshold={self.threshold}")

        except Exception as e:
            print(f"Error loading DistilBERT model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def filter(self, doc: Document) -> bool:
        """Filter documents using ML classifier."""
        self.total_docs += 1

        self._ensure_model_loaded()

        try:
            import torch

            inputs = self.tokenizer(
                doc.text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            prob_positive = probs[1]

            if self.log_stats and self.total_docs % 1000 == 0:
                self._print_stats()

            if prob_positive >= self.threshold:
                self.passed += 1
                doc.metadata["filter_type"] = "llm_classifier"
                doc.metadata["llm_model"] = "distilbert"
                doc.metadata["llm_score"] = float(prob_positive)
                return True

            self.failed += 1
            return False

        except Exception as e:
            print(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
            self.failed += 1
            return False

    def _print_stats(self):
        """Print statistics"""
        print(f"\n{'='*60}")
        print(f"LLM CLASSIFIER STATISTICS - {self.total_docs:,} docs processed")
        print(f"{'='*60}")
        print(f"Passed: {self.passed:,} ({100*self.passed/self.total_docs:.2f}%)")
        print(f"Failed: {self.failed:,} ({100*self.failed/self.total_docs:.2f}%)")
        if self.total_docs > 0:
            print(f"Pass rate: {100*self.passed/self.total_docs:.3f}%")
        print(f"{'='*60}\n")


# =============================================================================
# PIPELINE EXECUTORS
# =============================================================================

output_v5_test = "/fsx/dana_aubakirova/carbon_project/data/v5_final_100k"
logging_v5_test = "/fsx/dana_aubakirova/carbon_project/logs/v5_final_100k"

executor_v5_test_100k = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=dataset_path_10bt,
            limit=100000,
            text_key="text"
        ),
        DNAFilterV5Expanded(),
        JsonlWriter(
            output_folder=output_v5_test,
            max_file_size=int(10**9)
        )
    ],
    tasks=4,
    workers=4,
    time="01:00:00",
    partition="hopper-cpu",
    logging_dir=logging_v5_test,
    cpus_per_task=4,
    mem_per_cpu_gb=8,
    qos="normal",
)

output_v4_test = "/fsx/dana_aubakirova/carbon_project/data/v4_strict_400k"
logging_v4_test = "/fsx/dana_aubakirova/carbon_project/logs/v4_strict_400k"

executor_v4_test_400k = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=dataset_path_10bt,
            limit=100000,
            text_key="text"
        ),
        DNAFilterV4Strict(),
        JsonlWriter(
            output_folder=output_v4_test,
            max_file_size=int(10**9)
        )
    ],
    tasks=4,
    workers=4,
    time="01:00:00",
    partition="hopper-cpu",
    logging_dir=logging_v4_test,
    cpus_per_task=4,
    mem_per_cpu_gb=8,
    qos="normal",
)

output_v5_full_10bt = "s3://hf-carbon/v5_final_10BT_20260127"
logging_v5_full_10bt = "/fsx/dana_aubakirova/carbon_project/logs/v5_final_10BT"

executor_v5_full_10bt = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=dataset_path_10bt,
            limit=-1,
            text_key="text"
        ),
        DNAFilterV5Expanded(),
        JsonlWriter(
            output_folder=output_v5_full_10bt,
            max_file_size=int(10**9)
        )
    ],
    tasks=14,
    workers=8,
    time="04:00:00",
    partition="hopper-cpu",
    logging_dir=logging_v5_full_10bt,
    cpus_per_task=8,
    mem_per_cpu_gb=8,
    qos="normal",
)

output_v4_full = "s3://hf-carbon/fineweb-edu-v4-filtered-full-20260127"
logging_v4_full = "/fsx/dana_aubakirova/carbon_project/logs/fineweb-edu-v4-full"

executor_v4_full = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=dataset_path_full,
            limit=-1,
            text_key="text",
            recursive=True
        ),
        DNAFilterV4Strict(),
        JsonlWriter(
            output_folder=output_v4_full,
            max_file_size=int(10**9)
        )
    ],
    tasks=20,
    workers=8,
    time="48:00:00",
    partition="hopper-cpu",
    logging_dir=logging_v4_full,
    cpus_per_task=4,
    mem_per_cpu_gb=8,
    qos="normal",
)

output_llm_test = "/fsx/dana_aubakirova/carbon_project/data/llm_classifier_test"
logging_llm_test = "/fsx/dana_aubakirova/carbon_project/logs/llm_classifier_test"

executor_llm_test = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=dataset_path_10bt,
            limit=100000,
            text_key="text"
        ),
        LLMClassifier(
            model_path="/fsx/dana_aubakirova/carbon_project/models/finetuned_final",
            threshold=0.9,
            log_stats=True
        ),
        JsonlWriter(
            output_folder=output_llm_test,
            max_file_size=int(10**9)
        )
    ],
    tasks=4,
    workers=4,
    time="04:00:00",
    partition="hopper-prod",
    logging_dir=logging_llm_test,
    cpus_per_task=4,
    mem_per_cpu_gb=16,
    gpus_per_task=1,
    qos="normal",
)

# =============================================================================
# LLM CLASSIFIER - FULL FINEWEB-EDU (threshold 0.9)
# =============================================================================

output_llm_full = "s3://hf-carbon/llm_fineweb_edu_full_t09_20260202"
logging_llm_full = get_timestamped_log_dir("llm_fineweb_edu_full_t09")

executor_llm_full = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=dataset_path_full,
            limit=-1,
            text_key="text",
            recursive=True
        ),
        LLMClassifier(
            model_path="/fsx/dana_aubakirova/carbon_project/models/finetuned_final",
            threshold=0.9,
            log_stats=True
        ),
        JsonlWriter(
            output_folder=output_llm_full,
            max_file_size=int(10**9),
            compression="gzip"
        )
    ],
    tasks=40,
    workers=20,
    time="48:00:00",
    partition="hopper-prod",
    logging_dir=logging_llm_full,
    cpus_per_task=4,
    mem_per_cpu_gb=16,
    gpus_per_task=1,
    qos="normal",
)

output_v4_pubmed = "s3://hf-carbon/filtered_v4_pubmed_20260128"
logging_v4_pubmed = "/fsx/dana_aubakirova/carbon_project/logs/v4_pubmed"

executor_v4_pubmed = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=dataset_path_pubmed,
            limit=-1,
            text_key="article_text",
            glob_pattern="*.parquet"
        ),
        DNAFilterV4Strict(),
        JsonlWriter(
            output_folder=output_v4_pubmed,
            max_file_size=int(10**9)
        )
    ],
    tasks=20,
    workers=8,
    time="48:00:00",
    partition="hopper-cpu",
    logging_dir=logging_v4_pubmed,
    cpus_per_task=4,
    mem_per_cpu_gb=8,
    qos="normal",
)

output_v4_full_new = "s3://hf-carbon/v4_fineweb_edu_full_20260128"
logging_v4_full_new = "/fsx/dana_aubakirova/carbon_project/logs/v4_fineweb_edu_full_20260128"

executor_v4_full_new = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=dataset_path_full,
            limit=-1,
            text_key="text",
            recursive=True
        ),
        DNAFilterV4Strict(),
        JsonlWriter(
            output_folder=output_v4_full_new,
            max_file_size=int(10**9)
        )
    ],
    tasks=20,
    workers=8,
    time="48:00:00",
    partition="hopper-cpu",
    logging_dir=logging_v4_full_new,
    cpus_per_task=4,
    mem_per_cpu_gb=8,
    qos="normal",
)

# =============================================================================
# V4 FINEPDFS-EDU ENGLISH (23M documents, 183GB)
# =============================================================================

output_v4_finepdfs_edu = "s3://hf-carbon/v4_finepdfs_edu_english_20260130"
logging_v4_finepdfs_edu = get_timestamped_log_dir("v4_finepdfs_edu_english")

executor_v4_finepdfs_edu = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=dataset_path_finepdfs_edu_eng,
            limit=-1,
            text_key="text",
            recursive=True
        ),
        DNAFilterV4Strict(),
        JsonlWriter(
            output_folder=output_v4_finepdfs_edu,
            max_file_size=int(10**9),
            compression="gzip"
        )
    ],
    tasks=20,
    workers=8,
    time="48:00:00",
    partition="hopper-cpu",
    logging_dir=logging_v4_finepdfs_edu,
    cpus_per_task=4,
    mem_per_cpu_gb=8,
    qos="normal",
)

# =============================================================================
# V4 FINEPDFS ENGLISH (2.3TB - all languages but focusing on English)
# =============================================================================

output_v4_finepdfs_eng = "s3://hf-carbon/v4_finepdfs_english_20260130"
logging_v4_finepdfs_eng = get_timestamped_log_dir("v4_finepdfs_english")

executor_v4_finepdfs_eng = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=dataset_path_finepdfs_eng,
            limit=-1,
            text_key="text",
            recursive=False,
            glob_pattern="*.parquet"
        ),
        DNAFilterV4Strict(),
        JsonlWriter(
            output_folder=output_v4_finepdfs_eng,
            max_file_size=int(10**9),
            compression="gzip"
        )
    ],
    tasks=40,
    workers=20,
    time="24:00:00",
    partition="hopper-cpu",
    logging_dir=logging_v4_finepdfs_eng,
    cpus_per_task=4,
    mem_per_cpu_gb=8,
    qos="normal",
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_stats(logging_dir: str) -> dict:
    """Load and parse stats.json from logging directory."""
    import time
    stats_file = f"{logging_dir}/stats.json"

    for _ in range(10):
        if os.path.exists(stats_file):
            time.sleep(2)
            break
        time.sleep(1)

    with open(stats_file) as f:
        return json.load(f)




# =============================================================================
# MAIN EXECUTION
# =============================================================================

def analyze_v5_results():
    """Analyze V5 test results and show category breakdown."""
    import glob

    print("\n" + "="*80)
    print("V5 EXPANDED FILTER - 100K TEST RESULTS ANALYSIS")
    print("="*80)

    output_files = glob.glob(f"{output_v5_test}/*.jsonl")
    if not output_files:
        print(f"No output files found in {output_v5_test}")
        return

    all_docs = []
    for filepath in output_files:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    all_docs.append(doc)
                except:
                    continue

    print(f"\nTotal documents passed: {len(all_docs)}")

    categories = {'A': [], 'B': [], 'C': []}
    for doc in all_docs:
        cat = doc.get('metadata', {}).get('category', 'unknown')
        if cat in categories:
            categories[cat].append(doc)

    print(f"\nCategory Breakdown:")
    print(f"  A (bio_text - no sequences):     {len(categories['A']):,}")
    print(f"  B (bio_sequence - minimal text): {len(categories['B']):,}")
    print(f"  C (interleaved - seq + text):    {len(categories['C']):,}")

    import random
    print("\n" + "="*80)
    print("SAMPLE DOCUMENTS BY CATEGORY")
    print("="*80)

    for cat_name, cat_docs in categories.items():
        cat_labels = {
            'A': 'BIO-TEXT (molecular biology descriptions, no sequences)',
            'B': 'BIO-SEQUENCE (explicit sequences, minimal explanation)',
            'C': 'INTERLEAVED (sequences + natural language explanation)'
        }
        print(f"\n{'='*80}")
        print(f"CATEGORY {cat_name}: {cat_labels[cat_name]}")
        print(f"{'='*80}")

        if not cat_docs:
            print("  No documents in this category")
            continue

        samples = random.sample(cat_docs, min(5, len(cat_docs)))
        for i, doc in enumerate(samples, 1):
            text = doc.get('text', '')[:500]
            metadata = doc.get('metadata', {})
            print(f"\n--- Sample {i} ---")
            print(f"Strong keywords: {metadata.get('strong_keywords', 0)}")
            print(f"Supporting keywords: {metadata.get('supporting_keywords', 0)}")
            print(f"Has sequences: {metadata.get('has_sequences', False)}")
            print(f"Sequence count: {metadata.get('sequence_count', 0)}")
            print(f"Text preview:\n{text}...")
            print()

    try:
        stats = load_stats(logging_v5_test)
        for stage in stats:
            if 'V5' in stage.get('name', '').upper() or 'FILTER' in stage.get('name', '').upper():
                stage_stats = stage.get('stats', {})
                total = stage_stats.get('total', {}).get('total', 0)
                forwarded = stage_stats.get('forwarded', {}).get('total', 0)
                if total > 0:
                    print(f"\nPipeline Statistics:")
                    print(f"  Total processed: {total:,}")
                    print(f"  Passed filter: {forwarded:,} ({100*forwarded/total:.3f}%)")
    except Exception as e:
        print(f"\nCouldn't load pipeline stats: {e}")

    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DNA/RNA Filter Pipeline for FineWeb-EDU")
    parser.add_argument("--run", choices=["v5_test", "v5_full", "v4_test", "v4_full", "llm_test", "llm_full", "v4_pubmed", "v4_full_new", "v4_finepdfs_edu", "v4_finepdfs_eng"],
                       help="Which pipeline to run")
    parser.add_argument("--analyze-v5", action="store_true",
                       help="Analyze V5 test results")
    args = parser.parse_args()

    if args.run:
        if args.run == "v5_test":
            print("\n" + "="*60)
            print("SUBMITTING V5 EXPANDED FILTER TEST (100K docs)")
            print("Categories: A (bio_text), B (bio_sequence), C (interleaved)")
            print("="*60)
            executor_v5_test_100k.run()
            print("✓ V5 test job submitted!")
            print(f"Output: {output_v5_test}")
            print(f"Logs: {logging_v5_test}")

        elif args.run == "v5_full":
            print("\n" + "="*60)
            print("SUBMITTING V5 FINAL FILTER ON FULL 10BT (~9.7M docs)")
            print("Category C only: nucleotide sequences + keywords")
            print("="*60)
            executor_v5_full_10bt.run()
            print("✓ V5 full 10BT job submitted!")
            print(f"Output: {output_v5_full_10bt}")
            print(f"Logs: {logging_v5_full_10bt}")

        elif args.run == "v4_test":
            print("\n" + "="*60)
            print("SUBMITTING V4 STRICT FILTER TEST (400K docs)")
            print("15+ bases, 3+ unique, 1+ keyword")
            print("="*60)
            executor_v4_test_400k.run()
            print("✓ V4 test job submitted!")
            print(f"Output: {output_v4_test}")
            print(f"Logs: {logging_v4_test}")

        elif args.run == "v4_full":
            print("\n" + "="*60)
            print("SUBMITTING V4 FILTER ON FULL FINEWEB-EDU (1.5B docs)")
            print("15+ bases, 3+ unique, 1+ keyword")
            print("="*60)
            executor_v4_full.run()
            print("✓ V4 full FineWeb-EDU job submitted!")
            print(f"Output: {output_v4_full}")
            print(f"Logs: {logging_v4_full}")

        elif args.run == "llm_test":
            print("\n" + "="*60)
            print("SUBMITTING LLM CLASSIFIER TEST (100K docs)")
            print("Using fine-tuned DistilBERT (threshold=0.9)")
            print("="*60)
            executor_llm_test.run()
            print("✓ LLM test job submitted!")
            print(f"Output: {output_llm_test}")
            print(f"Logs: {logging_llm_test}")

        elif args.run == "llm_full":
            print("\n" + "="*60)
            print("SUBMITTING LLM CLASSIFIER ON FULL FINEWEB-EDU (~1.5B docs)")
            print("Using fine-tuned DistilBERT (threshold=0.9)")
            print("WARNING: Based on analysis, expect ~1.61% pass rate (~24M docs)")
            print("WARNING: Majority will be false positives (general biology)")
            print("S3 Output: s3://hf-carbon/llm_fineweb_edu_full_t09_20260202")
            print("="*60)
            print("\nEstimated results based on 10K sample analysis:")
            print("  - Expected captures: ~24 million documents (1.61%)")
            print("  - False positive rate: >90%")
            print("  - True Category A: <2.4 million")
            print("  - Content types: Science journalism, clinical research,")
            print("                   education, general biology")
            print("\n" + "="*60)

            response = input("Are you sure you want to proceed? (yes/no): ")
            if response.lower() == 'yes':
                executor_llm_full.run()
                print("✓ LLM full FineWeb-EDU job submitted!")
                print(f"Output: {output_llm_full}")
                print(f"Logs: {logging_llm_full}")
            else:
                print("Job submission cancelled.")

        elif args.run == "v4_pubmed":
            print("\n" + "="*60)
            print("SUBMITTING V4 FILTER ON PUBMED DATASET (~8.3M docs)")
            print("15+ bases, 3+ unique, 1+ keyword")
            print("S3 Output: s3://hf-carbon/v4_pubmed_20260128")
            print("="*60)
            executor_v4_pubmed.run()
            print("✓ V4 PubMed job submitted!")
            print(f"Output: {output_v4_pubmed}")
            print(f"Logs: {logging_v4_pubmed}")

        elif args.run == "v4_full_new":
            print("\n" + "="*60)
            print("SUBMITTING V4 FILTER ON FULL FINEWEB-EDU (1.5B docs)")
            print("15+ bases, 3+ unique, 1+ keyword")
            print("S3 Output: s3://hf-carbon/v4_fineweb_edu_full_20260128")
            print("="*60)
            executor_v4_full_new.run()
            print("✓ V4 full FineWeb-EDU job submitted!")
            print(f"Output: {output_v4_full_new}")
            print(f"Logs: {logging_v4_full_new}")

        elif args.run == "v4_finepdfs_edu":
            print("\n" + "="*60)
            print("SUBMITTING V4 FILTER ON FINEPDFS-EDU ENGLISH (23M docs)")
            print("15+ bases, 3+ unique, 1+ keyword")
            print("Dataset: FinePDFs-Edu English (183 GB)")
            print("S3 Output: s3://hf-carbon/v4_finepdfs_edu_english_20260129")
            print("="*60)
            executor_v4_finepdfs_edu.run()
            print("✓ V4 FinePDFs-Edu English job submitted!")
            print(f"Output: {output_v4_finepdfs_edu}")
            print(f"Logs: {logging_v4_finepdfs_edu}")

        elif args.run == "v4_finepdfs_eng":
            print("\n" + "="*60)
            print("SUBMITTING V4 FILTER ON FINEPDFS ENGLISH (2.3TB)")
            print("15+ bases, 3+ unique, 1+ keyword")
            print("Dataset: FinePDFs English (2.3 TB)")
            print("S3 Output: s3://hf-carbon/v4_finepdfs_english_20260130")
            print("="*60)
            executor_v4_finepdfs_eng.run()
            print("✓ V4 FinePDFs English job submitted!")
            print(f"Output: {output_v4_finepdfs_eng}")
            print(f"Logs: {logging_v4_finepdfs_eng}")

    if args.analyze_v5:
        analyze_v5_results()
