#!/usr/bin/env python3
"""Generate synthetic training data for sequence reasoning tasks.

The generator focuses on broad molecular-biology capabilities that are useful
for training across multiple sequence-analysis and cloning subtasks.

Usage::

    # Generate all subtasks and write to scratch
    uv run --directory data python seqqa/generate_data.py \\
        --output scratch/seqqa_training.jsonl

    # Generate and push directly to the Hub without a local dataset file
    uv run --directory data python seqqa/generate_data.py \\
        --push-to-hub --hub-repo hf-carbon/seqqa-synth --hub-config v1
"""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import logging
import random
import re
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from Bio.Restriction import RestrictionBatch
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

_active_progress: Any | None = None

PLASMID_SOURCE_DATASET_ID = "carbon-internal/AddGene"
VECTOR_SOURCE_DATASET_ID = "carbon-internal/AddGene"
PROCEDURAL_SOURCE_DATASET_ID = "procedural"
TRAINING_CLONING_ENZYME_NAMES = [
    "EcoRI",
    "BamHI",
    "HindIII",
    "SalI",
    "XbaI",
    "SphI",
    "PstI",
    "KpnI",
    "SacI",
    "SmaI",
    "NotI",
    "XhoI",
    "NcoI",
    "NdeI",
    "BglII",
    "NheI",
    "EcoRV",
    "ClaI",
    "AgeI",
    "SpeI",
]

AA_THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}
AA_ONE_TO_THREE = {value: key for key, value in AA_THREE_TO_ONE.items()}

# Kozak context hierarchy for translation-efficiency tasks.
# Each entry specifies the 3 nt before AUG (pre3) and 1 nt after AUG (post1).
# Strength ranking follows mammalian consensus (Kozak 1987):
#   -3 position: A or G (purine) is strong; C or U (pyrimidine) is weak
#   +4 position: G is strong; A/C/U is weak
# Tier 1 (strong): purine at -3 AND G at +4
# Tier 2 (moderate): purine at -3 OR G at +4, not both
# Tier 3 (weak): pyrimidine at -3 AND non-G at +4
KOZAK_CANDIDATES = [
    {"pre3": "ACC", "post1": "G", "tier": 1, "label": "strong"},
    {"pre3": "GCC", "post1": "G", "tier": 1, "label": "strong"},
    {"pre3": "ACA", "post1": "G", "tier": 1, "label": "strong"},
    {"pre3": "GCA", "post1": "G", "tier": 1, "label": "strong"},
    {"pre3": "ACC", "post1": "A", "tier": 2, "label": "moderate"},
    {"pre3": "GCC", "post1": "C", "tier": 2, "label": "moderate"},
    {"pre3": "ACA", "post1": "U", "tier": 2, "label": "moderate"},
    {"pre3": "CCC", "post1": "G", "tier": 2, "label": "moderate"},
    {"pre3": "UCC", "post1": "G", "tier": 2, "label": "moderate"},
    {"pre3": "UUC", "post1": "A", "tier": 3, "label": "weak"},
    {"pre3": "CUC", "post1": "U", "tier": 3, "label": "weak"},
    {"pre3": "UUA", "post1": "C", "tier": 3, "label": "weak"},
    {"pre3": "CCU", "post1": "U", "tier": 3, "label": "weak"},
]
TRANSLATION_EFFICIENCY_NUM_CANDIDATES = 4
TRANSCRIPT_LABELS = ["Transcript A", "Transcript B", "Transcript C", "Transcript D"]

# Legacy contexts used by generate_translation_upstream_aug_count.
TRANSLATION_CONTEXTS = [
    {"name": "strong_context", "kozak": "GCCACC", "leader_bias": 0.42},
    {"name": "moderate_context", "kozak": "GCCGCC", "leader_bias": 0.46},
    {"name": "weak_context", "kozak": "UUAUUC", "leader_bias": 0.53},
]
TRANSLATION_SAFE_FILLER_CODONS = [
    "CCC",
    "GCC",
    "CCG",
    "CGC",
    "GGC",
    "CUG",
    "GUC",
    "UCC",
]

RE_PRIMER_BINDING_LEN = 20
RE_PRIMER_PAD_LEN = 4
GIBSON_HOMOLOGY_LEN = 20
GIBSON_INSERT_BINDING_LEN = 20
PCR_PRIMER_BINDING_LEN = 20

# ---------------------------------------------------------------------------
# Prompt suffixes — modelled on the labbench2/seqqa2 <answer> XML format
# ---------------------------------------------------------------------------

PROMPT_SUFFIX_INTEGER = (
    "Provide your final answer as a single integer surrounded by "
    "XML tags like this: <answer>YOUR_INTEGER</answer>"
)
PROMPT_SUFFIX_GC_PERCENT = (
    "Provide your final answer as a single numeric value "
    "(percentage, without the % symbol) surrounded by "
    "XML tags like this: <answer>YOUR_ANSWER</answer>"
)
PROMPT_SUFFIX_PRIMER_PAIR = (
    "Provide your final answer as the forward and reverse primer "
    "sequences (5' to 3'), comma-separated, surrounded by "
    "XML tags like this: <answer>FORWARD_PRIMER,REVERSE_PRIMER</answer>"
)
PROMPT_SUFFIX_FRAGMENT_LENGTHS = (
    "Provide your final answer as comma-separated fragment lengths "
    "in base pairs (integers only, no units) surrounded by "
    "XML tags like this: <answer>LENGTH1,LENGTH2,LENGTH3</answer>"
)
PROMPT_SUFFIX_AMINO_ACID = (
    "Provide your final answer as a three-letter amino acid code "
    "surrounded by XML tags like this: <answer>Xxx</answer>"
)
PROMPT_SUFFIX_AA_SEQUENCE = (
    "Provide your final answer as the amino acid sequence "
    "(single-letter codes, no spaces) surrounded by "
    "XML tags like this: <answer>YOUR_SEQUENCE</answer>"
)
PROMPT_SUFFIX_DNA_SEQUENCE = (
    "Provide your final answer as the DNA sequence "
    "(5' to 3', using A/C/G/T only, no spaces) surrounded by "
    "XML tags like this: <answer>YOUR_SEQUENCE</answer>"
)
PROMPT_SUFFIX_ENZYME_PAIR = (
    "Provide your final answer as two enzyme names, comma-separated, "
    "surrounded by XML tags like this: <answer>ENZYME1,ENZYME2</answer>"
)
PROMPT_SUFFIX_TRANSCRIPT_LABEL = (
    "Provide your final answer as the transcript label "
    "(e.g. Transcript A) surrounded by "
    "XML tags like this: <answer>Transcript X</answer>"
)
# ---------------------------------------------------------------------------
# Answer regex patterns — machine-checkable extraction from model output
# ---------------------------------------------------------------------------

ANSWER_REGEX_INTEGER = r"-?\d+"
ANSWER_REGEX_GC_PERCENT = r"\d+"
ANSWER_REGEX_PRIMER_PAIR = r"(?P<forward>[ACGTacgt]+)\s*,\s*(?P<reverse>[ACGTacgt]+)"
ANSWER_REGEX_FRAGMENT_LENGTHS = r"\d+(?:\s*,\s*\d+)*"
ANSWER_REGEX_AMINO_ACID = r"[A-Z][a-z]{2}"
ANSWER_REGEX_AA_SEQUENCE = r"[ACDEFGHIKLMNPQRSTVWY]+"
ANSWER_REGEX_DNA_SEQUENCE = r"[ACGTacgt]+"
ANSWER_REGEX_ENZYME_PAIR = r"(?P<enzyme1>[A-Za-z0-9]+)\s*,\s*(?P<enzyme2>[A-Za-z0-9]+)"
ANSWER_REGEX_TRANSCRIPT_LABEL = r"Transcript\s+[A-D]"

TRANSLATION_LEADER_LEN = 36
TRANSLATION_UPSTREAM_AUG_COUNT_SUBTASK = "translation_upstream_aug_count"
VECTOR_COMPATIBILITY_LIST_SIZE = 6
ORF_AA_SEQUENCE_MIN_LEN = 6
DEFAULT_ORF_AA_SEQUENCE_MAX_LEN = 5000
LONG_AMPLICON_SEQUENCE_MIN_LEN = 10000
LONG_AMPLICON_SEQUENCE_MAX_LEN = 25000
LONG_AMPLICON_SEQUENCE_FLANK_LEN = 40
LONG_ORF_SEQUENCE_WINDOW_MIN_LEN = 1200
LONG_ORF_SEQUENCE_WINDOW_MAX_LEN = 50000

_rb = RestrictionBatch(TRAINING_CLONING_ENZYME_NAMES)
CLONING_ENZYMES: dict[str, Any] = {}
for _name in TRAINING_CLONING_ENZYME_NAMES:
    _enzyme = _rb.get(_name)
    CLONING_ENZYMES[_name] = {
        "enzyme": _enzyme,
        "site": _enzyme.site,
        "size": _enzyme.size,
        "ovhg": _enzyme.ovhg,
        "is_blunt": _enzyme.ovhg == 0,
    }

_plsdb_cache: list[str] | None = None
_long_plsdb_cache: list[str] | None = None
_long_orf_candidate_cache: list[tuple[int, str]] | None = None
_vector_catalog_cache: list[dict[str, str]] | None = None
_insert_catalog_cache: list[dict[str, str]] | None = None
_default_length_distribution: str = "uniform"
_default_min_len: int | None = None
_default_max_len: int | None = None


def _make_example(
    question: str,
    ideal: str,
    subtask_base: str,
    source: str,
    rng: random.Random,
    prompt_suffix: str = "",
    validator_type: str | None = None,
    validator_params: dict[str, object] | None = None,
    answer_regex: str | None = None,
) -> dict[str, object]:
    """Build one training example."""
    prompt = f"{question}\n\n{prompt_suffix}" if prompt_suffix else question
    example: dict[str, object] = {
        "id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
        "question": question,
        "ideal": ideal,
        "prompt_suffix": prompt_suffix,
        "prompt": prompt,
        "subtask": subtask_base,
        "source": source,
        "validator_type": validator_type or subtask_base,
        "validator_params": validator_params or {},
        "answer_regex": answer_regex or "",
        "question_len": len(question),
        "ideal_len": len(ideal),
    }
    if _active_progress is not None:
        _active_progress.update(1)
    return example


def load_plsdb_sequences(
    n: int,
    min_len: int = 500,
    max_len: int = 3000,
    rng: random.Random | None = None,
    length_distribution: str | None = None,
) -> list[str]:
    """Return *n* random DNA windows from the PLSDB plasmid collection.

    Parameters
    ----------
    length_distribution:
        ``"uniform"`` samples lengths uniformly between *min_len* and
        *max_len*.  ``"normal"`` uses a truncated normal distribution
        centred on the midpoint so that lengths cluster around the
        middle of the range while still covering the full span.
        When *None* (the default), uses ``_default_length_distribution``.
    """
    global _plsdb_cache  # noqa: PLW0603

    if rng is None:
        rng = random.Random()
    if _default_min_len is not None:
        min_len = _default_min_len
    if _default_max_len is not None:
        max_len = _default_max_len
    if length_distribution is None:
        length_distribution = _default_length_distribution
    if max_len < min_len:
        raise ValueError(f"max_len ({max_len}) must be >= min_len ({min_len})")
    if length_distribution not in ("uniform", "normal"):
        raise ValueError(
            f"length_distribution must be 'uniform' or 'normal', got {length_distribution!r}"
        )

    if _plsdb_cache is None:
        from datasets import load_dataset

        logger.info("Downloading natural plasmid sequences (streaming) ...")
        ds = load_dataset(
            PLASMID_SOURCE_DATASET_ID,
            data_files="sequences.jsonl",
            split="train",
            streaming=True,
        )
        seqs: list[str] = []
        for row in ds:
            seq = (row.get("sequence") or "").upper()
            if re.fullmatch(r"[ACGT]+", seq):
                seqs.append(seq)
            if len(seqs) >= 2500:
                break
        _plsdb_cache = seqs
        logger.info("Cached %d plasmid sequences.", len(seqs))

    eligible = [seq for seq in _plsdb_cache if len(seq) >= min_len]
    if not eligible:
        logger.warning("No cached plasmid sequences satisfy min_len=%d.", min_len)
        return []

    mu = (min_len + max_len) / 2
    sigma = (max_len - min_len) / 4  # ~95% of draws within [min_len, max_len]

    windows: list[str] = []
    attempts = 0
    while len(windows) < n and attempts < n * 30:
        attempts += 1
        seq = rng.choice(eligible)
        effective_max = min(max_len, len(seq))
        if length_distribution == "normal":
            raw = rng.gauss(mu, sigma)
            length = int(round(max(min_len, min(effective_max, raw))))
        else:
            length = rng.randint(min_len, effective_max)
        start = rng.randint(0, len(seq) - length)
        windows.append(seq[start : start + length])

    if len(windows) < n:
        logger.warning(
            "Requested %d plasmid windows but only generated %d after %d attempts.",
            n,
            len(windows),
            attempts,
        )
    return windows


def load_long_plsdb_sequences(
    n: int,
    min_len: int = LONG_AMPLICON_SEQUENCE_MIN_LEN + (2 * LONG_AMPLICON_SEQUENCE_FLANK_LEN),
    max_len: int = LONG_AMPLICON_SEQUENCE_MAX_LEN + (2 * LONG_AMPLICON_SEQUENCE_FLANK_LEN),
    rng: random.Random | None = None,
    length_distribution: str | None = None,
) -> list[str]:
    """Return long DNA windows from a larger cache of eligible AddGene plasmids."""
    global _long_plsdb_cache  # noqa: PLW0603

    if rng is None:
        rng = random.Random()
    if length_distribution is None:
        length_distribution = _default_length_distribution
    if max_len < min_len:
        raise ValueError(f"max_len ({max_len}) must be >= min_len ({min_len})")
    if length_distribution not in ("uniform", "normal"):
        raise ValueError(
            f"length_distribution must be 'uniform' or 'normal', got {length_distribution!r}"
        )

    if _long_plsdb_cache is None:
        from datasets import load_dataset

        logger.info("Downloading long natural plasmid sequences (streaming) ...")
        ds = load_dataset(
            PLASMID_SOURCE_DATASET_ID,
            data_files="sequences.jsonl",
            split="train",
            streaming=True,
        )
        seqs: list[str] = []
        for row in ds:
            seq = (row.get("sequence") or "").upper()
            if re.fullmatch(r"[ACGT]+", seq) and len(seq) >= LONG_ORF_SEQUENCE_WINDOW_MIN_LEN:
                seqs.append(seq)
            if len(seqs) >= 1500:
                break
        _long_plsdb_cache = seqs
        logger.info("Cached %d long plasmid sequences.", len(seqs))

    eligible = [seq for seq in _long_plsdb_cache if len(seq) >= min_len]
    if not eligible:
        logger.warning("No cached long plasmid sequences satisfy min_len=%d.", min_len)
        return []

    mu = (min_len + max_len) / 2
    sigma = (max_len - min_len) / 4

    windows: list[str] = []
    attempts = 0
    while len(windows) < n and attempts < n * 30:
        attempts += 1
        seq = rng.choice(eligible)
        effective_max = min(max_len, len(seq))
        if length_distribution == "normal":
            raw = rng.gauss(mu, sigma)
            length = int(round(max(min_len, min(effective_max, raw))))
        else:
            length = rng.randint(min_len, effective_max)
        start = rng.randint(0, len(seq) - length)
        windows.append(seq[start : start + length])

    if len(windows) < n:
        logger.warning(
            "Requested %d long plasmid windows but only generated %d after %d attempts.",
            n,
            len(windows),
            attempts,
        )
    return windows


def load_long_plsdb_full_sequences(
    n: int,
    min_len: int = LONG_ORF_SEQUENCE_WINDOW_MIN_LEN,
    max_len: int = LONG_ORF_SEQUENCE_WINDOW_MAX_LEN,
    rng: random.Random | None = None,
) -> list[str]:
    """Return full long plasmid sequences instead of subwindows."""
    # Reuse the shared long-sequence cache, then sample full templates from it.
    _ = load_long_plsdb_sequences(1, min_len=min_len, max_len=max_len, rng=rng)
    eligible = [
        seq
        for seq in (_long_plsdb_cache or [])
        if len(seq) >= min_len and len(seq) <= max_len
    ]
    if not eligible:
        logger.warning(
            "No cached long plasmid sequences satisfy %d <= len <= %d.",
            min_len,
            max_len,
        )
        return []
    if rng is None:
        rng = random.Random()
    longest_eligible = sorted(eligible, key=len, reverse=True)[: max(200, n * 4)]
    return [rng.choice(longest_eligible) for _ in range(n)]


def load_long_orf_candidate_sequences(
    n: int,
    min_aa_len: int = ORF_AA_SEQUENCE_MIN_LEN,
    max_aa_len: int = DEFAULT_ORF_AA_SEQUENCE_MAX_LEN,
    rng: random.Random | None = None,
) -> list[str]:
    """Return full plasmid sequences biased toward the longest natural ORFs."""
    global _long_orf_candidate_cache  # noqa: PLW0603

    if _long_orf_candidate_cache is None:
        from datasets import load_dataset

        logger.info("Scanning AddGene plasmids for long natural ORFs (streaming) ...")
        ds = load_dataset(
            PLASMID_SOURCE_DATASET_ID,
            data_files="sequences.jsonl",
            split="train",
            streaming=True,
        )
        candidates: list[tuple[int, str]] = []
        valid_rows = 0
        for row in ds:
            seq = (row.get("sequence") or "").upper()
            if not re.fullmatch(r"[ACGT]+", seq):
                continue
            if len(seq) < LONG_ORF_SEQUENCE_WINDOW_MIN_LEN or len(seq) > LONG_ORF_SEQUENCE_WINDOW_MAX_LEN:
                continue
            valid_rows += 1
            orf = longest_orf(seq)
            if orf is None:
                continue
            orf_len = int(orf["length_aa"])
            candidates.append((orf_len, seq))
            if valid_rows >= 20000:
                break
        candidates.sort(key=lambda item: item[0], reverse=True)
        _long_orf_candidate_cache = candidates[:1500]
        logger.info(
            "Cached %d long-ORF candidate plasmids from %d scanned rows.",
            len(_long_orf_candidate_cache),
            valid_rows,
        )

    eligible = [
        seq
        for orf_len, seq in (_long_orf_candidate_cache or [])
        if min_aa_len <= orf_len <= max_aa_len
    ]
    if not eligible:
        logger.warning(
            "No cached long-ORF candidate plasmids satisfy %d <= longest_orf <= %d aa.",
            min_aa_len,
            max_aa_len,
        )
        return []
    if rng is None:
        rng = random.Random()
    preferred = eligible[: max(200, n * 4)]
    return [rng.choice(preferred) for _ in range(n)]


def generate_random_dna(
    length: int,
    rng: random.Random,
    gc_target: float | None = None,
) -> str:
    """Generate a random DNA sequence."""
    if gc_target is None:
        weights = [1.0, 1.0, 1.0, 1.0]
    else:
        gc_weight = gc_target / 2
        at_weight = (1.0 - gc_target) / 2
        weights = [at_weight, gc_weight, gc_weight, at_weight]
    return "".join(rng.choices("ACGT", weights=weights, k=length))


def _pick_public_vector_sequence(row: dict[str, object]) -> str | None:
    sequences = row.get("sequences") or {}
    if not isinstance(sequences, dict):
        return None

    candidate_keys = (
        "public_addgene_full_sequences",
        "public_user_full_sequences",
    )
    for key in candidate_keys:
        entries = sequences.get(key) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            seq = (entry or {}).get("sequence", "").upper()
            if re.fullmatch(r"[ACGT]+", seq):
                return seq
    return None


def cut_positions(seq: str, enzyme_names: list[str]) -> list[int]:
    """Return unique cut positions for the named enzymes."""
    rb = RestrictionBatch(enzyme_names)
    result = rb.search(Seq(seq))
    cuts: set[int] = set()
    for positions in result.values():
        cuts.update(positions)
    return sorted(cuts)


def enzyme_cut_count(seq: str, enzyme_name: str) -> int:
    """Return the number of cuts for *enzyme_name* on *seq*."""
    return len(cut_positions(seq, [enzyme_name]))


def has_internal_sites(seq: str, enzyme_names: list[str]) -> bool:
    """Check whether *seq* contains sites for any of *enzyme_names*."""
    return any(enzyme_cut_count(seq, enzyme_name) > 0 for enzyme_name in enzyme_names)


def digest(seq: str, enzyme_names: list[str]) -> list[int]:
    """Digest a linear sequence and return fragment lengths."""
    cuts = cut_positions(seq, enzyme_names)
    if not cuts:
        return [len(seq)]
    fragments = []
    previous = 0
    for cut in cuts:
        fragments.append(cut - previous)
        previous = cut
    fragments.append(len(seq) - previous)
    return fragments


def find_orfs(seq: str, min_aa: int = 0) -> list[dict[str, object]]:
    """Find ORFs in all six frames."""
    orfs: list[dict[str, object]] = []
    for strand, nucleotide_sequence in (
        ("+", seq),
        ("-", str(Seq(seq).reverse_complement())),
    ):
        for frame in range(3):
            index = frame
            while index + 2 < len(nucleotide_sequence):
                codon = nucleotide_sequence[index : index + 3]
                if codon != "ATG":
                    index += 3
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    protein = str(Seq(nucleotide_sequence[index:]).translate(to_stop=True))
                if len(protein) >= min_aa:
                    orfs.append(
                        {
                            "start": index,
                            "end": index + len(protein) * 3 + 3,
                            "strand": strand,
                            "frame": frame,
                            "aa_seq": protein,
                            "length_aa": len(protein),
                        }
                    )
                index += len(protein) * 3 + 3
    return orfs


def longest_orf(seq: str) -> dict[str, object] | None:
    """Return the longest ORF, if any."""
    orfs = find_orfs(seq, min_aa=1)
    if not orfs:
        return None
    return max(orfs, key=lambda row: row["length_aa"])


def extract_orf_dna(seq: str, orf: dict[str, object]) -> str:
    """Return the ORF DNA sequence on the coding strand."""
    start = int(orf["start"])
    end = int(orf["end"])
    strand = str(orf["strand"])
    if strand == "+":
        return seq[start:end]
    if strand == "-":
        reverse_complement = str(Seq(seq).reverse_complement())
        return reverse_complement[start:end]
    raise ValueError(f"Unexpected ORF strand {strand!r}")


def load_insert_catalog(
    min_records: int = 250,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """Build a catalog of insert-like CDS sequences from plasmid ORFs."""
    global _insert_catalog_cache  # noqa: PLW0603

    if _insert_catalog_cache is not None:
        if rng is None:
            return list(_insert_catalog_cache)
        shuffled = list(_insert_catalog_cache)
        rng.shuffle(shuffled)
        return shuffled

    if rng is None:
        rng = random.Random(0)

    seed_rng = random.Random(17)
    windows = load_plsdb_sequences(1200, min_len=1200, max_len=3200, rng=seed_rng)
    records: list[dict[str, str]] = []
    seen_prefixes: set[str] = set()
    for seq in windows:
        orf = longest_orf(seq)
        if orf is None:
            continue
        coding_seq = extract_orf_dna(seq, orf)
        if len(coding_seq) < 240 or len(coding_seq) > 1800:
            continue
        prefix = coding_seq[:40]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        record_index = len(records) + 1
        records.append(
            {
                "name": f"insert_{record_index:03d}",
                "label": f"coding insert {record_index:03d}",
                "sequence": coding_seq,
                "source": PLASMID_SOURCE_DATASET_ID,
            }
        )
        if len(records) >= min_records:
            break

    if not records:
        raise RuntimeError("No insert catalog records could be generated.")

    _insert_catalog_cache = records
    shuffled = list(records)
    rng.shuffle(shuffled)
    logger.info("Built insert catalog with %d records.", len(records))
    return shuffled


def load_vector_catalog(
    min_records: int = 200,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """Build a catalog of recipient vectors from AddGene."""
    global _vector_catalog_cache  # noqa: PLW0603

    if _vector_catalog_cache is not None:
        if rng is None:
            return list(_vector_catalog_cache)
        shuffled = list(_vector_catalog_cache)
        rng.shuffle(shuffled)
        return shuffled

    from datasets import load_dataset

    logger.info("Downloading vector catalog from AddGene (streaming) ...")
    ds = load_dataset(VECTOR_SOURCE_DATASET_ID, split="train", streaming=True)
    vectors: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for row in ds:
        vector_seq = _pick_public_vector_sequence(row)
        if vector_seq is None or len(vector_seq) < 2000:
            continue

        vector_name = str(row.get("name") or "").strip()
        cloning = row.get("cloning") or {}
        backbone = str((cloning or {}).get("backbone") or "").strip() or vector_name
        if not vector_name:
            vector_name = backbone or f"vector_{len(vectors) + 1:03d}"
        vector_types = ", ".join((cloning or {}).get("vector_types") or [])
        if vector_name in seen_names:
            continue

        single_cut_enzymes = [
            enzyme_name
            for enzyme_name in TRAINING_CLONING_ENZYME_NAMES
            if enzyme_cut_count(vector_seq, enzyme_name) == 1
        ]
        if len(single_cut_enzymes) < 2:
            continue

        vectors.append(
            {
                "name": vector_name,
                "backbone": backbone or vector_name,
                "sequence": vector_seq,
                "vector_types": vector_types,
                "single_cut_enzymes": ",".join(single_cut_enzymes),
                "source": VECTOR_SOURCE_DATASET_ID,
            }
        )
        seen_names.add(vector_name)
        if len(vectors) >= min_records:
            break

    if not vectors:
        raise RuntimeError("No usable AddGene vector records were found.")

    _vector_catalog_cache = vectors
    if rng is None:
        return list(vectors)
    shuffled = list(vectors)
    rng.shuffle(shuffled)
    logger.info("Built vector catalog with %d records.", len(vectors))
    return shuffled


def vector_enzymes(vector_record: dict[str, str]) -> list[str]:
    """Decode the cached single-cut enzyme list for a vector."""
    return [part for part in vector_record["single_cut_enzymes"].split(",") if part]


def count_exact_occurrences(sequence: str, query: str) -> int:
    """Count overlapping exact matches of *query* within *sequence*."""
    if not query:
        return 0
    count = 0
    start = 0
    sequence = sequence.upper()
    query = query.upper()
    while True:
        match = sequence.find(query, start)
        if match < 0:
            return count
        count += 1
        start = match + 1


def exact_primer_binding_count(
    template: str,
    primer: str,
    *,
    reverse: bool = False,
) -> int:
    """Count exact full-length binding matches for a primer on a template."""
    query = primer.upper()
    if reverse:
        query = str(Seq(query).reverse_complement())
    return count_exact_occurrences(template, query)


def primer_pair_binds_uniquely(template: str, fwd: str, rev: str) -> bool:
    """Check whether both primers have unique exact binding sites."""
    return (
        exact_primer_binding_count(template, fwd) == 1
        and exact_primer_binding_count(template, rev, reverse=True) == 1
    )


def compatible_vector_enzymes(
    vector_sequence: str,
    insert_sequence: str,
    enzyme_names: list[str],
) -> list[str]:
    """Return single-cut vector enzymes that do not cut the insert."""
    return [
        enzyme_name
        for enzyme_name in enzyme_names
        if enzyme_cut_count(vector_sequence, enzyme_name) == 1
        and not has_internal_sites(insert_sequence, [enzyme_name])
    ]


def find_compatible_vector_enzyme_pairs(
    vector_sequence: str,
    insert_sequence: str,
    enzyme_names: list[str],
) -> list[tuple[str, str]]:
    """Return ordered enzyme pairs that cut the vector once and avoid the insert."""
    compatible = compatible_vector_enzymes(vector_sequence, insert_sequence, enzyme_names)
    return [
        (enzyme_5p, enzyme_3p)
        for enzyme_5p in compatible
        for enzyme_3p in compatible
        if enzyme_5p != enzyme_3p
    ]


def find_unique_compatible_vector_enzyme_pairs(
    vector_sequence: str,
    insert_sequence: str,
    enzyme_names: list[str],
) -> list[tuple[str, str]]:
    """Return unique unordered compatible enzyme pairs."""
    compatible = compatible_vector_enzymes(vector_sequence, insert_sequence, enzyme_names)
    return [pair for pair in combinations(sorted(compatible), 2)]


def circular_flanks(
    vector_sequence: str,
    enzyme_name: str,
    flank_len: int = 24,
) -> tuple[str, str] | None:
    """Return flanking sequences around a unique circular cut site."""
    positions = cut_positions(vector_sequence, [enzyme_name])
    if len(positions) != 1:
        return None
    cut_index = positions[0] - 1
    left = "".join(
        vector_sequence[(cut_index - flank_len + offset) % len(vector_sequence)]
        for offset in range(flank_len)
    )
    right = "".join(
        vector_sequence[(cut_index + offset) % len(vector_sequence)]
        for offset in range(flank_len)
    )
    return left, right


def gc_percent(seq: str) -> int:
    """Return GC percentage rounded to the nearest integer."""
    return round(gc_fraction(seq) * 100)


def enumerate_amplicon_primer_pair_candidates(
    template: str,
    amplicon_len: int,
    binding_len: int = PCR_PRIMER_BINDING_LEN,
) -> list[tuple[int, str, str]]:
    """Return exact-match primer-pair candidates for a target amplicon length."""
    candidates: list[tuple[int, str, str]] = []
    seen: set[tuple[str, str]] = set()
    max_start = len(template) - amplicon_len
    for start in range(max_start + 1):
        end = start + amplicon_len
        fwd = template[start : start + binding_len]
        rev_binding = template[end - binding_len : end]
        if len(fwd) != binding_len or len(rev_binding) != binding_len:
            continue
        rev = str(Seq(rev_binding).reverse_complement())
        if not primer_pair_binds_uniquely(template, fwd, rev):
            continue
        pair = (fwd, rev)
        if pair in seen:
            continue
        candidates.append((start, fwd, rev))
        seen.add(pair)
    return candidates


def enumerate_amplicon_primer_pairs(
    template: str,
    amplicon_len: int,
    binding_len: int = PCR_PRIMER_BINDING_LEN,
) -> list[tuple[str, str]]:
    """Return unique exact-match primer pairs that yield a target amplicon length."""
    return [
        (fwd, rev)
        for _, fwd, rev in enumerate_amplicon_primer_pair_candidates(
            template,
            amplicon_len,
            binding_len=binding_len,
        )
    ]


def find_leftmost_amplicon_length_candidate(
    template: str,
    min_amplicon_len: int,
    max_amplicon_len: int,
    rng: random.Random,
    binding_len: int = PCR_PRIMER_BINDING_LEN,
) -> tuple[int, tuple[str, str]] | None:
    """Find one amplicon length and choose the leftmost valid exact-match primer pair."""
    template = template.upper()
    last_start = len(template) - binding_len
    if last_start < 0 or max_amplicon_len < min_amplicon_len:
        return None

    kmers = [template[start : start + binding_len] for start in range(last_start + 1)]
    counts = Counter(kmers)
    unique_positions = [counts[kmer] == 1 for kmer in kmers]

    candidate_lengths = list(range(min_amplicon_len, max_amplicon_len + 1))
    rng.shuffle(candidate_lengths)
    for amplicon_len in candidate_lengths:
        max_start = len(template) - amplicon_len
        for start in range(max_start + 1):
            end_binding_start = start + amplicon_len - binding_len
            if not unique_positions[start] or not unique_positions[end_binding_start]:
                continue
            fwd = template[start : start + binding_len]
            rev_binding_start = start + amplicon_len - binding_len
            rev_binding = template[rev_binding_start : rev_binding_start + binding_len]
            rev = str(Seq(rev_binding).reverse_complement())
            return amplicon_len, (fwd, rev)
    return None


def design_re_primers(
    insert_sequence: str,
    enzyme_5p: str,
    enzyme_3p: str,
    binding_len: int = RE_PRIMER_BINDING_LEN,
    pad_len: int = RE_PRIMER_PAD_LEN,
) -> tuple[str, str]:
    """Design restriction-cloning primers with enzyme-site tails."""
    site_5p = CLONING_ENZYMES[enzyme_5p]["site"]
    site_3p = CLONING_ENZYMES[enzyme_3p]["site"]
    padding = "G" * pad_len
    binding_fwd = insert_sequence[:binding_len].upper()
    binding_rev = str(Seq(insert_sequence[-binding_len:]).reverse_complement()).upper()
    return f"{padding}{site_5p}{binding_fwd}", f"{padding}{site_3p}{binding_rev}"


def design_gibson_primers(
    insert_sequence: str,
    left_flank: str,
    right_flank: str,
    overlap_len: int = GIBSON_HOMOLOGY_LEN,
    binding_len: int = GIBSON_INSERT_BINDING_LEN,
) -> tuple[str, str]:
    """Design Gibson primers using vector homology flanks."""
    homology_fwd = left_flank[-overlap_len:]
    homology_rev = str(Seq(right_flank[:overlap_len]).reverse_complement())
    binding_fwd = insert_sequence[:binding_len].upper()
    binding_rev = str(Seq(insert_sequence[-binding_len:]).reverse_complement()).upper()
    return f"{homology_fwd}{binding_fwd}", f"{homology_rev}{binding_rev}"


def find_primer_binding(template: str, primer: str, min_match: int | None = None) -> int | None:
    """Find where a primer binds on a template using its 3' end.

    When *min_match* is ``None`` (the default), the full primer length is used
    so that only exact full-length matches are accepted.
    """
    if min_match is None:
        min_match = len(primer)
    query = primer[-min_match:].upper()
    match = template.upper().find(query)
    if match < 0:
        return None
    return match


def compute_amplicon_length(template: str, fwd: str, rev: str) -> int | None:
    """Compute the expected amplicon length from primer binding."""
    fwd_pos = find_primer_binding(template, fwd)
    reverse_rc = str(Seq(rev).reverse_complement())
    rev_pos = find_primer_binding(template, reverse_rc)
    if fwd_pos is None or rev_pos is None:
        return None
    end_pos = rev_pos + len(reverse_rc)
    if end_pos < fwd_pos:
        return None
    return end_pos - fwd_pos


def extract_amplicon_sequence(template: str, fwd: str, rev: str) -> str | None:
    """Return the exact amplicon sequence implied by a primer pair."""
    fwd_pos = find_primer_binding(template, fwd)
    reverse_rc = str(Seq(rev).reverse_complement())
    rev_pos = find_primer_binding(template, reverse_rc)
    if fwd_pos is None or rev_pos is None:
        return None
    end_pos = rev_pos + len(reverse_rc)
    if end_pos < fwd_pos:
        return None
    return template[fwd_pos:end_pos]


def find_random_amplicon_candidate(
    template: str,
    min_amplicon_len: int,
    max_amplicon_len: int,
    rng: random.Random,
    *,
    binding_len: int = PCR_PRIMER_BINDING_LEN,
    edge_buffer: int = LONG_AMPLICON_SEQUENCE_FLANK_LEN,
    max_attempts: int = 80,
) -> tuple[int, str, str, str] | None:
    """Sample one exact-match amplicon candidate from a template."""
    template = template.upper()
    max_amplicon_len = min(max_amplicon_len, len(template) - (2 * edge_buffer))
    if max_amplicon_len < min_amplicon_len:
        return None
    preferred_min_amplicon_len = max(min_amplicon_len, int(max_amplicon_len * 0.6))

    for _ in range(max_attempts):
        amplicon_len = rng.randint(preferred_min_amplicon_len, max_amplicon_len)
        latest_start = len(template) - amplicon_len - edge_buffer
        if latest_start < edge_buffer:
            continue
        start = rng.randint(edge_buffer, latest_start)
        amplicon = template[start : start + amplicon_len]
        fwd = template[start : start + binding_len]
        rev = str(Seq(amplicon[-binding_len:]).reverse_complement())
        if not primer_pair_binds_uniquely(template, fwd, rev):
            continue
        if extract_amplicon_sequence(template, fwd, rev) != amplicon:
            continue
        return amplicon_len, fwd, rev, amplicon

    return None


def count_orfs_strictly_over_threshold(seq: str, threshold: int) -> int:
    """Count ORFs whose translated length is strictly greater than *threshold*."""
    return sum(
        int(orf["length_aa"]) > threshold
        for orf in find_orfs(seq, min_aa=threshold)
    )


def count_upstream_aug_codons(leader_rna: str) -> int:
    """Count AUG codons in a 5' leader sequence."""
    return count_exact_occurrences(leader_rna, "AUG")


def generate_translation_leader_with_upstream_augs(
    rng: random.Random,
    upstream_aug_count: int,
    length: int = TRANSLATION_LEADER_LEN,
) -> str:
    """Generate a 5' RNA leader containing exactly *upstream_aug_count* AUG codons."""
    if length % 3 != 0:
        raise ValueError("Translation leader length must be a multiple of 3.")

    codon_count = length // 3
    if upstream_aug_count > codon_count:
        raise ValueError("Requested more upstream AUG codons than leader codons.")

    aug_positions = set(rng.sample(range(codon_count), upstream_aug_count))
    leader_codons = [
        "AUG" if index in aug_positions else rng.choice(TRANSLATION_SAFE_FILLER_CODONS)
        for index in range(codon_count)
    ]
    return "".join(leader_codons)


def generate_seq_gc_pct(n: int, seed: int) -> list[dict[str, object]]:
    """Generate GC-percentage questions."""
    rng = random.Random(seed)
    sequences = load_plsdb_sequences(n, min_len=50, max_len=5000, rng=rng)
    examples = []
    for seq in sequences[:n]:
        answer = gc_percent(seq)
        question = (
            f"What is the GC content of the following DNA sequence?\n"
            f"Sequence: {seq}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=str(answer),
                subtask_base="seq_gc_pct",
                source=PLASMID_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_GC_PERCENT,
                answer_regex=ANSWER_REGEX_GC_PERCENT,
                validator_params={"sequence": seq, "rounding": "nearest_integer"},
            )
        )
    return examples


def generate_restriction_fragment_count(n: int, seed: int) -> list[dict[str, object]]:
    """Generate fragment-count questions after restriction digest."""
    rng = random.Random(seed)
    sequences = load_plsdb_sequences(n * 3, min_len=50, max_len=5000, rng=rng)
    examples = []
    seq_index = 0
    while len(examples) < n and seq_index < len(sequences):
        seq = sequences[seq_index]
        seq_index += 1
        cutting = [
            enzyme_name
            for enzyme_name in rng.sample(
                TRAINING_CLONING_ENZYME_NAMES,
                len(TRAINING_CLONING_ENZYME_NAMES),
            )
            if enzyme_cut_count(seq, enzyme_name) > 0
        ]
        if not cutting:
            continue
        chosen = cutting[: rng.choice([1, 2])]
        fragments = digest(seq, chosen)
        question = (
            f"How many fragments result from digesting the following linear DNA sequence "
            f"with {' and '.join(chosen)}?\n"
            f"Sequence: {seq}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=str(len(fragments)),
                subtask_base="restriction_fragment_count",
                source=PLASMID_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_INTEGER,
                answer_regex=ANSWER_REGEX_INTEGER,
                validator_params={
                    "sequence": seq,
                    "enzymes": chosen,
                    "topology": "linear",
                },
            )
        )
    return examples


def generate_restriction_fragment_lengths(
    n: int,
    seed: int,
) -> list[dict[str, object]]:
    """Generate fragment-length questions after restriction digest."""
    rng = random.Random(seed)
    sequences = load_plsdb_sequences(n * 3, min_len=50, max_len=5000, rng=rng)
    examples = []
    seq_index = 0
    while len(examples) < n and seq_index < len(sequences):
        seq = sequences[seq_index]
        seq_index += 1
        cutting = [
            enzyme_name
            for enzyme_name in rng.sample(
                TRAINING_CLONING_ENZYME_NAMES,
                len(TRAINING_CLONING_ENZYME_NAMES),
            )
            if enzyme_cut_count(seq, enzyme_name) > 0
        ]
        if not cutting:
            continue
        chosen = cutting[: rng.choice([1, 2])]
        fragments = sorted(digest(seq, chosen))
        answer = ", ".join(str(length) for length in fragments)
        question = (
            f"What fragment lengths would result from digesting the following linear DNA "
            f"sequence with {' and '.join(chosen)}?\n"
            f"Sequence: {seq}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=answer,
                subtask_base="restriction_fragment_lengths",
                source=PLASMID_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_FRAGMENT_LENGTHS,
                answer_regex=ANSWER_REGEX_FRAGMENT_LENGTHS,
                validator_params={
                    "sequence": seq,
                    "enzymes": chosen,
                    "topology": "linear",
                    "ordering": "sorted_ascending",
                },
            )
        )
    return examples


def generate_orf_aa_position(n: int, seed: int) -> list[dict[str, object]]:
    """Generate amino-acid-at-position questions for the longest ORF."""
    rng = random.Random(seed)
    sequences = load_plsdb_sequences(n * 4, min_len=50, max_len=5000, rng=rng)
    examples = []
    seq_index = 0
    while len(examples) < n and seq_index < len(sequences):
        seq = sequences[seq_index]
        seq_index += 1
        orf = longest_orf(seq)
        if orf is None or int(orf["length_aa"]) < 10:
            continue
        position = rng.randint(1, int(orf["length_aa"]))
        aa_one = str(orf["aa_seq"])[position - 1]
        answer = AA_ONE_TO_THREE.get(aa_one, aa_one)
        question = (
            f"What amino acid is encoded at position {position} in the protein "
            f"translated from the longest open reading frame (considering all six "
            f"reading frames on both strands) of the following sequence?\n"
            f"Sequence: {seq}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=answer,
                subtask_base="orf_aa_position",
                source=PLASMID_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_AMINO_ACID,
                answer_regex=ANSWER_REGEX_AMINO_ACID,
                validator_params={
                    "sequence": seq,
                    "position": position,
                    "strands": "both",
                    "longest_orf_tie_break": "longest_aa",
                },
            )
        )
    return examples


def generate_orf_aa_sequence(
    n: int,
    seed: int,
    max_aa_len: int = DEFAULT_ORF_AA_SEQUENCE_MAX_LEN,
) -> list[dict[str, object]]:
    """Generate full translated-sequence questions for the longest ORF."""
    if max_aa_len < ORF_AA_SEQUENCE_MIN_LEN:
        raise ValueError(
            f"max_aa_len ({max_aa_len}) must be >= {ORF_AA_SEQUENCE_MIN_LEN}"
        )
    rng = random.Random(seed)
    sequences = load_long_orf_candidate_sequences(
        n * 12,
        min_aa_len=ORF_AA_SEQUENCE_MIN_LEN,
        max_aa_len=max_aa_len,
        rng=rng,
    )
    examples = []
    seq_index = 0
    while len(examples) < n and seq_index < len(sequences):
        seq = sequences[seq_index]
        seq_index += 1
        orf = longest_orf(seq)
        if (
            orf is None
            or int(orf["length_aa"]) < ORF_AA_SEQUENCE_MIN_LEN
            or int(orf["length_aa"]) > max_aa_len
        ):
            continue
        aa_seq = str(orf["aa_seq"])
        question = (
            f"Translate the longest open reading frame (considering all six reading "
            f"frames on both strands) in the following DNA sequence.\n"
            f"Sequence: {seq}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=aa_seq,
                subtask_base="orf_aa_sequence",
                source=PLASMID_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_AA_SEQUENCE,
                answer_regex=ANSWER_REGEX_AA_SEQUENCE,
                validator_params={
                    "sequence": seq,
                    "strands": "both",
                    "longest_orf_tie_break": "longest_aa",
                },
            )
        )
    return examples


def generate_orf_count_over_threshold(
    n: int,
    seed: int,
) -> list[dict[str, object]]:
    """Generate questions about ORF counts above a protein-length threshold."""
    rng = random.Random(seed)
    sequences = load_plsdb_sequences(n * 4, min_len=50, max_len=5000, rng=rng)
    examples = []
    seq_index = 0
    while len(examples) < n and seq_index < len(sequences):
        seq = sequences[seq_index]
        seq_index += 1
        threshold = rng.choice([15, 20, 30, 45, 60])
        count = count_orfs_strictly_over_threshold(seq, threshold)
        if count < 1:
            continue
        question = (
            f"How many open reading frames (considering all six reading frames on both "
            f"strands) in the following sequence encode proteins longer than {threshold} "
            f"amino acids?\n"
            f"Sequence: {seq}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=str(count),
                subtask_base="orf_count_over_threshold",
                source=PLASMID_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_INTEGER,
                answer_regex=ANSWER_REGEX_INTEGER,
                validator_params={
                    "sequence": seq,
                    "threshold": threshold,
                    "strands": "both",
                },
            )
        )
    return examples


def generate_translation_upstream_aug_count(
    n: int,
    seed: int,
) -> list[dict[str, object]]:
    """Generate translation questions with deterministic upstream-AUG counts."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        upstream_aug_count = rng.randint(0, 3)
        leader_rna = generate_translation_leader_with_upstream_augs(rng, upstream_aug_count)
        cds_len = rng.randint(24, 34) * 3
        cds_dna = "ATG" + generate_random_dna(cds_len - 3, rng, gc_target=0.48)
        while "*" in str(Seq(cds_dna).translate())[:-1]:
            cds_dna = "ATG" + generate_random_dna(cds_len - 3, rng, gc_target=0.48)
        cds_rna = cds_dna.replace("T", "U")
        context = rng.choice(TRANSLATION_CONTEXTS)
        coding_context = context["kozak"] + cds_rna
        question = (
            "How many AUG codons are in the 5' leader of the following transcript, "
            "upstream of the annotated main start codon? The main start codon is the "
            "AUG immediately after the Kozak context.\n"
            f"5' leader: {leader_rna}\n"
            f"Main start context + coding sequence: {coding_context}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=str(count_upstream_aug_codons(leader_rna)),
                subtask_base=TRANSLATION_UPSTREAM_AUG_COUNT_SUBTASK,
                source=PROCEDURAL_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_INTEGER,
                answer_regex=ANSWER_REGEX_INTEGER,
                validator_params={
                    "leader_rna": leader_rna,
                },
            )
        )
    return examples


def kozak_tier(pre3: str, post1: str) -> int:
    """Return the Kozak strength tier (1=strong, 2=moderate, 3=weak).

    The two critical positions in the mammalian Kozak consensus are:
      -3 (first nt of *pre3*): purine (A/G) is strong, pyrimidine (C/U) is weak.
      +4 (*post1*): G is strong, anything else is weak.
    Tier 1: purine at -3 AND G at +4.
    Tier 2: purine at -3 OR G at +4, but not both.
    Tier 3: pyrimidine at -3 AND non-G at +4.
    """
    minus3_strong = pre3[0] in "AG"
    plus4_strong = post1 == "G"
    if minus3_strong and plus4_strong:
        return 1
    if minus3_strong or plus4_strong:
        return 2
    return 3


def _build_transcript_rna(
    pre3: str,
    post1: str,
    leader_prefix: str,
    cds_suffix: str,
) -> str:
    """Assemble a full RNA with a specified Kozak context around the main AUG."""
    return leader_prefix + pre3 + "AUG" + post1 + cds_suffix


def generate_translation_efficiency(
    n: int,
    seed: int,
) -> list[dict[str, object]]:
    """Generate Kozak-context selection questions (LAB-Bench-style).

    Presents several candidate transcripts that share the same leader and CDS
    but differ in their Kozak initiation context around the main AUG.  The
    model must identify which transcript has the strongest Kozak context and
    is therefore expected to translate most efficiently in a mammalian cell.
    """
    rng = random.Random(seed)
    num_candidates = TRANSLATION_EFFICIENCY_NUM_CANDIDATES
    examples = []

    for _ in range(n):
        # Build a shared leader prefix and CDS suffix.
        leader_len = rng.randint(30, 120)
        leader_codons = [
            rng.choice(TRANSLATION_SAFE_FILLER_CODONS)
            for _ in range(leader_len // 3 + 1)
        ]
        leader_prefix = "".join(leader_codons)[:leader_len]

        cds_dna_len = rng.randint(90, 300)
        cds_dna = generate_random_dna(cds_dna_len, rng, gc_target=0.48)
        cds_suffix = cds_dna.replace("T", "U")

        # Pick one tier-1 (strong) context and fill the rest from tiers 2-3,
        # ensuring no duplicate (pre3, post1) pairs.
        tier1 = [k for k in KOZAK_CANDIDATES if k["tier"] == 1]
        other = [k for k in KOZAK_CANDIDATES if k["tier"] > 1]
        strong_pick = rng.choice(tier1)
        rng.shuffle(other)
        weak_picks: list[dict[str, object]] = []
        seen = {(strong_pick["pre3"], strong_pick["post1"])}
        for candidate in other:
            pair = (candidate["pre3"], candidate["post1"])
            if pair not in seen:
                weak_picks.append(candidate)
                seen.add(pair)
            if len(weak_picks) == num_candidates - 1:
                break

        # Assign shuffled positions.
        all_picks = [strong_pick] + weak_picks
        rng.shuffle(all_picks)
        labels = TRANSCRIPT_LABELS[:num_candidates]
        correct_label = labels[all_picks.index(strong_pick)]

        # Build question listing each candidate transcript.
        lines = [
            "Below are several candidate mRNA transcripts that share the same "
            "coding sequence but differ in the Kozak initiation context around "
            "the main AUG codon. Which transcript is expected to have the "
            "highest translation efficiency in a mammalian cell?\n"
        ]
        candidate_sequences = []
        for label, pick in zip(labels, all_picks):
            seq = _build_transcript_rna(
                pick["pre3"], pick["post1"], leader_prefix, cds_suffix,
            )
            candidate_sequences.append(seq)
            lines.append(f"{label}: {seq}")
        question = "\n".join(lines)

        examples.append(
            _make_example(
                question=question,
                ideal=correct_label,
                subtask_base="translation_efficiency",
                source=PROCEDURAL_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_TRANSCRIPT_LABEL,
                answer_regex=ANSWER_REGEX_TRANSCRIPT_LABEL,
                validator_params={
                    "candidate_labels": labels,
                    "candidate_pre3": [p["pre3"] for p in all_picks],
                    "candidate_post1": [p["post1"] for p in all_picks],
                    "selection_rule": "strongest_kozak_tier",
                },
            )
        )
    return examples


def generate_restriction_clone_primer_design(
    n: int,
    seed: int,
) -> list[dict[str, object]]:
    """Generate primer-design questions for vector-specific restriction cloning."""
    rng = random.Random(seed)
    vectors = load_vector_catalog(rng=rng)
    inserts = load_insert_catalog(rng=rng)
    examples = []
    attempts = 0
    while len(examples) < n and attempts < n * 60:
        attempts += 1
        vector = rng.choice(vectors)
        insert = rng.choice(inserts)
        pairs = find_compatible_vector_enzyme_pairs(
            vector["sequence"],
            insert["sequence"],
            vector_enzymes(vector),
        )
        if not pairs:
            continue
        enzyme_5p, enzyme_3p = rng.choice(pairs)
        fwd, rev = design_re_primers(insert["sequence"], enzyme_5p, enzyme_3p)
        question = (
            f"Design restriction cloning primers to insert {insert['label']} into "
            f"vector {vector['name']} (backbone: {vector['backbone']}) using {enzyme_5p} "
            f"(forward) and {enzyme_3p} (reverse). Each primer should have "
            f"{RE_PRIMER_PAD_LEN} leading G bases, then the restriction site, then "
            f"{RE_PRIMER_BINDING_LEN} nt of insert-binding sequence.\n"
            f"Insert sequence: {insert['sequence']}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=f"{fwd}, {rev}",
                subtask_base="restriction_clone_primer_design",
                source=f"{VECTOR_SOURCE_DATASET_ID}+{insert['source']}",
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_PRIMER_PAIR,
                answer_regex=ANSWER_REGEX_PRIMER_PAIR,
                validator_params={
                    "insert_sequence": insert["sequence"],
                    "enzyme_5p": enzyme_5p,
                    "enzyme_3p": enzyme_3p,
                    "pad_len": RE_PRIMER_PAD_LEN,
                    "binding_len": RE_PRIMER_BINDING_LEN,
                },
            )
        )
    if len(examples) < n:
        logger.warning(
            "Requested %d restriction-cloning examples but only generated %d.",
            n,
            len(examples),
        )
    return examples


def generate_amplicon_target_primers(n: int, seed: int) -> list[dict[str, object]]:
    """Generate questions asking for primers that amplify a target amplicon."""
    rng = random.Random(seed)
    templates = load_plsdb_sequences(n * 3, min_len=50, max_len=5000, rng=rng)
    examples = []
    template_index = 0
    while len(examples) < n and template_index < len(templates):
        template = templates[template_index]
        template_index += 1
        binding_len = PCR_PRIMER_BINDING_LEN
        max_amplicon = min(520, len(template) - 100)
        if max_amplicon < 120:
            continue
        amplicon_len = rng.randint(120, max_amplicon)
        if len(template) - amplicon_len - 40 < 40:
            continue
        amplicon_start = rng.randint(40, len(template) - amplicon_len - 40)
        amplicon = template[amplicon_start : amplicon_start + amplicon_len]
        fwd = template[amplicon_start : amplicon_start + binding_len]
        rev = str(
            Seq(
                template[
                    amplicon_start + amplicon_len - binding_len : amplicon_start + amplicon_len
                ]
            ).reverse_complement()
        )
        if count_exact_occurrences(template, amplicon) != 1:
            continue
        if not primer_pair_binds_uniquely(template, fwd, rev):
            continue
        question = (
            f"Design {binding_len} nt primers to amplify the target region from "
            f"the template below.\n"
            f"Target: {amplicon}\n"
            f"Template: {template}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=f"{fwd}, {rev}",
                subtask_base="amplicon_target_primers",
                source=PLASMID_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_PRIMER_PAIR,
                answer_regex=ANSWER_REGEX_PRIMER_PAIR,
                validator_params={
                    "template": template,
                    "target": amplicon,
                    "primer_length": binding_len,
                    "binding_rule": "exact_full_length_unique",
                },
            )
        )
    return examples


def generate_amplicon_length_primers(n: int, seed: int) -> list[dict[str, object]]:
    """Generate questions asking for primers that yield a target amplicon length."""
    rng = random.Random(seed)
    templates = load_plsdb_sequences(n * 8, min_len=50, max_len=5000, rng=rng)
    examples = []
    template_index = 0
    while len(examples) < n and template_index < len(templates):
        template = templates[template_index]
        template_index += 1
        binding_len = PCR_PRIMER_BINDING_LEN
        candidate = find_leftmost_amplicon_length_candidate(
            template,
            120,
            min(620, len(template) - 100),
            rng,
            binding_len=binding_len,
        )
        if candidate is None:
            continue
        amplicon_len, (fwd, rev) = candidate
        question = (
            f"Design {binding_len} nt primers to amplify a {amplicon_len} bp region "
            f"from the template below. Each primer must bind exactly once. If multiple "
            f"valid pairs exist, choose the earliest forward primer position.\n"
            f"Template: {template}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=f"{fwd}, {rev}",
                subtask_base="amplicon_length_primers",
                source=PLASMID_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_PRIMER_PAIR,
                answer_regex=ANSWER_REGEX_PRIMER_PAIR,
                validator_params={
                    "template": template,
                    "amplicon_length": amplicon_len,
                    "primer_length": binding_len,
                    "binding_rule": "exact_full_length_unique",
                    "tie_break": "earliest_forward_primer",
                },
            )
        )
    return examples


def generate_primer_pair_amplicon_length(
    n: int,
    seed: int,
) -> list[dict[str, object]]:
    """Generate questions asking for amplicon length from a primer pair."""
    rng = random.Random(seed)
    templates = load_plsdb_sequences(n * 3, min_len=50, max_len=5000, rng=rng)
    examples = []
    template_index = 0
    while len(examples) < n and template_index < len(templates):
        template = templates[template_index]
        template_index += 1
        binding_len = 20
        max_amplicon = min(620, len(template) - 100)
        if max_amplicon < 120:
            continue
        amplicon_len = rng.randint(120, max_amplicon)
        if len(template) - amplicon_len - 40 < 40:
            continue
        amplicon_start = rng.randint(40, len(template) - amplicon_len - 40)
        fwd = template[amplicon_start : amplicon_start + binding_len]
        rev = str(
            Seq(
                template[
                    amplicon_start + amplicon_len - binding_len : amplicon_start + amplicon_len
                ]
            ).reverse_complement()
        )
        computed_length = compute_amplicon_length(template, fwd, rev)
        if computed_length is None:
            continue
        question = (
            f"What is the expected amplicon length from primers {fwd} and {rev} "
            f"on the following template?\n"
            f"Template: {template}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=str(computed_length),
                subtask_base="primer_pair_amplicon_length",
                source=PLASMID_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_INTEGER,
                answer_regex=ANSWER_REGEX_INTEGER,
                validator_params={
                    "template": template,
                    "forward_primer": fwd,
                    "reverse_primer": rev,
                    "binding_rule": "exact_full_length",
                },
            )
        )
    return examples


def generate_amplicon_sequence(n: int, seed: int) -> list[dict[str, object]]:
    """Generate long-answer PCR tasks requiring the exact amplicon sequence."""
    rng = random.Random(seed)
    templates = load_long_plsdb_sequences(
        n * 12,
        min_len=LONG_AMPLICON_SEQUENCE_MIN_LEN + (2 * LONG_AMPLICON_SEQUENCE_FLANK_LEN),
        max_len=LONG_AMPLICON_SEQUENCE_MAX_LEN + (2 * LONG_AMPLICON_SEQUENCE_FLANK_LEN),
        rng=rng,
    )
    examples = []
    template_index = 0
    while len(examples) < n and template_index < len(templates):
        template = templates[template_index]
        template_index += 1
        candidate = find_random_amplicon_candidate(
            template,
            LONG_AMPLICON_SEQUENCE_MIN_LEN,
            LONG_AMPLICON_SEQUENCE_MAX_LEN,
            rng,
            binding_len=PCR_PRIMER_BINDING_LEN,
            edge_buffer=LONG_AMPLICON_SEQUENCE_FLANK_LEN,
        )
        if candidate is None:
            continue
        amplicon_len, fwd, rev, amplicon = candidate
        question = (
            f"What is the exact 5' to 3' DNA sequence of the PCR amplicon produced by "
            f"primers {fwd} and {rev} on the template below? Each primer binds exactly "
            f"once, and the expected amplicon length is {amplicon_len} bp.\n"
            f"Template: {template}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=amplicon,
                subtask_base="amplicon_sequence",
                source=PLASMID_SOURCE_DATASET_ID,
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_DNA_SEQUENCE,
                answer_regex=ANSWER_REGEX_DNA_SEQUENCE,
                validator_params={
                    "template": template,
                    "forward_primer": fwd,
                    "reverse_primer": rev,
                    "amplicon_length": amplicon_len,
                    "binding_rule": "exact_full_length_unique",
                },
            )
        )
    if len(examples) < n:
        logger.warning(
            "Requested %d amplicon-sequence examples but only generated %d.",
            n,
            len(examples),
        )
    return examples


def generate_vector_insert_compatibility(
    n: int,
    seed: int,
) -> list[dict[str, object]]:
    """Generate enzyme-pair selection questions for vector/insert compatibility."""
    rng = random.Random(seed)
    vectors = load_vector_catalog(rng=rng)
    inserts = load_insert_catalog(rng=rng)
    examples = []
    attempts = 0
    while len(examples) < n and attempts < n * 60:
        attempts += 1
        vector = rng.choice(vectors)
        insert = rng.choice(inserts)
        enzymes = vector_enzymes(vector)
        compatible = compatible_vector_enzymes(
            vector["sequence"],
            insert["sequence"],
            enzymes,
        )
        if len(enzymes) < VECTOR_COMPATIBILITY_LIST_SIZE or len(compatible) < 2:
            continue
        required_compatible = rng.sample(sorted(compatible), 2)
        remaining_enzymes = [enzyme for enzyme in enzymes if enzyme not in required_compatible]
        if len(remaining_enzymes) < VECTOR_COMPATIBILITY_LIST_SIZE - len(required_compatible):
            continue
        listed_enzymes = sorted(
            [
                *required_compatible,
                *rng.sample(
                    remaining_enzymes,
                    VECTOR_COMPATIBILITY_LIST_SIZE - len(required_compatible),
                ),
            ]
        )
        valid_pairs = find_unique_compatible_vector_enzyme_pairs(
            vector["sequence"],
            insert["sequence"],
            listed_enzymes,
        )
        if not valid_pairs:
            continue
        correct_pair = valid_pairs[0]
        single_cut_summary = ", ".join(listed_enzymes)
        question = (
            f"Vector {vector['name']} (backbone: {vector['backbone']}) has single-cut "
            f"sites for {single_cut_summary}. Which pair of these enzymes can be used "
            f"for directional cloning while leaving the insert intact? Report the "
            f"alphabetically first valid pair.\n"
            f"Insert sequence: {insert['sequence']}"
        )
        correct_answer = ", ".join(correct_pair)
        examples.append(
            _make_example(
                question=question,
                ideal=correct_answer,
                subtask_base="vector_insert_compatibility",
                source=f"{VECTOR_SOURCE_DATASET_ID}+{insert['source']}",
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_ENZYME_PAIR,
                answer_regex=ANSWER_REGEX_ENZYME_PAIR,
                validator_params={
                    "vector_sequence": vector["sequence"],
                    "insert_sequence": insert["sequence"],
                    "listed_enzymes": listed_enzymes,
                    "compatibility_rule": "single_cut_in_vector_and_no_internal_cut_in_insert",
                    "tie_break": "alphabetically_first",
                },
            )
        )
    if len(examples) < n:
        logger.warning(
            "Requested %d vector-compatibility examples but only generated %d.",
            n,
            len(examples),
        )
    return examples


def generate_gibson_primer_design(n: int, seed: int) -> list[dict[str, object]]:
    """Generate Gibson primer-design questions across mixed vectors."""
    rng = random.Random(seed)
    vectors = load_vector_catalog(rng=rng)
    inserts = load_insert_catalog(rng=rng)
    examples = []
    attempts = 0
    while len(examples) < n and attempts < n * 60:
        attempts += 1
        vector = rng.choice(vectors)
        insert = rng.choice(inserts)
        enzyme_name = rng.choice(vector_enzymes(vector))
        flanks = circular_flanks(vector["sequence"], enzyme_name)
        if flanks is None:
            continue
        left_flank, right_flank = flanks
        fwd, rev = design_gibson_primers(insert["sequence"], left_flank, right_flank)
        question = (
            f"Design Gibson assembly primers (with {GIBSON_HOMOLOGY_LEN} bp overlaps) "
            f"to insert the sequence below into vector {vector['name']} "
            f"(backbone: {vector['backbone']}) linearized with {enzyme_name}. "
            f"Each primer should have {GIBSON_HOMOLOGY_LEN} nt of vector homology "
            f"followed by {GIBSON_INSERT_BINDING_LEN} nt of insert-binding sequence.\n"
            f"Left junction flank: {left_flank}\n"
            f"Right junction flank: {right_flank}\n"
            f"Insert sequence: {insert['sequence']}"
        )
        examples.append(
            _make_example(
                question=question,
                ideal=f"{fwd}, {rev}",
                subtask_base="gibson_primer_design",
                source=f"{VECTOR_SOURCE_DATASET_ID}+{insert['source']}",
                rng=rng,
                prompt_suffix=PROMPT_SUFFIX_PRIMER_PAIR,
                answer_regex=ANSWER_REGEX_PRIMER_PAIR,
                validator_params={
                    "left_flank": left_flank,
                    "right_flank": right_flank,
                    "insert_sequence": insert["sequence"],
                    "overlap_len": GIBSON_HOMOLOGY_LEN,
                    "binding_len": GIBSON_INSERT_BINDING_LEN,
                },
            )
        )
    if len(examples) < n:
        logger.warning(
            "Requested %d Gibson-primer examples but only generated %d.",
            n,
            len(examples),
        )
    return examples


SUBTASK_GENERATORS: dict[str, Any] = {
    "seq_gc_pct": generate_seq_gc_pct,
    "restriction_fragment_count": generate_restriction_fragment_count,
    "restriction_fragment_lengths": generate_restriction_fragment_lengths,
    "orf_aa_position": generate_orf_aa_position,
    "orf_aa_sequence": generate_orf_aa_sequence,
    "orf_count_over_threshold": generate_orf_count_over_threshold,
    TRANSLATION_UPSTREAM_AUG_COUNT_SUBTASK: generate_translation_upstream_aug_count,
    "translation_efficiency": generate_translation_efficiency,
    "restriction_clone_primer_design": generate_restriction_clone_primer_design,
    "amplicon_target_primers": generate_amplicon_target_primers,
    "amplicon_length_primers": generate_amplicon_length_primers,
    "primer_pair_amplicon_length": generate_primer_pair_amplicon_length,
    "amplicon_sequence": generate_amplicon_sequence,
    "vector_insert_compatibility": generate_vector_insert_compatibility,
    "gibson_primer_design": generate_gibson_primer_design,
}


def stable_subtask_seed(seed: int, subtask_base: str) -> int:
    """Derive a deterministic per-subtask seed independent of PYTHONHASHSEED."""
    digest = hashlib.sha256(subtask_base.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % (2**31)
    return seed + offset


def generate_all(
    n_per_subtask: int = 1000,
    seed: int = 42,
    subtasks: list[str] | None = None,
    show_progress: bool = True,
    generator_kwargs_by_subtask: dict[str, dict[str, Any]] | None = None,
    length_distribution: str | None = None,
) -> list[dict[str, object]]:
    """Generate training examples for all or selected subtasks."""
    global _active_progress, _default_length_distribution  # noqa: PLW0603

    if length_distribution is not None:
        _default_length_distribution = length_distribution

    targets = subtasks or list(SUBTASK_GENERATORS)
    progress = None
    previous_progress = _active_progress
    all_examples: list[dict[str, object]] = []
    if show_progress and sys.stderr.isatty():
        progress = tqdm(
            total=len(targets) * n_per_subtask,
            desc="SeqQA",
            unit="example",
            dynamic_ncols=True,
        )
    _active_progress = progress
    try:
        for subtask_base in targets:
            if subtask_base not in SUBTASK_GENERATORS:
                raise ValueError(
                    f"Unknown subtask {subtask_base!r}. Valid: {sorted(SUBTASK_GENERATORS)}"
                )
            generator = SUBTASK_GENERATORS[subtask_base]
            subtask_seed = stable_subtask_seed(seed, subtask_base)
            if progress is not None:
                progress.set_description(subtask_base)
            logger.info(
                "Generating %d examples for %s (seed=%d) ...",
                n_per_subtask,
                subtask_base,
                subtask_seed,
            )
            generator_kwargs = {}
            if generator_kwargs_by_subtask is not None:
                generator_kwargs = generator_kwargs_by_subtask.get(subtask_base, {})
            examples = generator(
                n=n_per_subtask,
                seed=subtask_seed,
                **generator_kwargs,
            )
            logger.info("  -> generated %d examples.", len(examples))
            if progress is not None and len(examples) < n_per_subtask:
                progress.total -= n_per_subtask - len(examples)
                progress.refresh()
            all_examples.extend(examples)
    finally:
        _active_progress = previous_progress
        if progress is not None:
            progress.set_description("SeqQA")
            progress.close()
    return all_examples


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def push_examples_to_hub(
    examples: list[dict[str, object]],
    hub_repo: str,
    hub_config: str,
) -> None:
    """Push generated examples to the Hub."""
    from datasets import Dataset

    dataset = Dataset.from_list(examples)
    dataset.push_to_hub(
        hub_repo,
        config_name=hub_config,
        split="train",
        commit_message=f"Upload training subset {hub_config} ({len(examples)} examples)",
    )
    logger.info("Pushed %d examples to %s / %s", len(examples), hub_repo, hub_config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sequence-reasoning training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-per-subtask",
        type=int,
        default=1000,
        help="Number of examples per subtask (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--subtasks",
        nargs="+",
        default=None,
        help="Generate only these subtasks. Default: all training subtasks.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSONL output path. Omit to avoid local dataset files.",
    )
    parser.add_argument(
        "--max-orf-aa-seq-len",
        type=int,
        default=DEFAULT_ORF_AA_SEQUENCE_MAX_LEN,
        help=(
            "Maximum translated amino-acid length allowed for the orf_aa_sequence "
            f"subtask (default: {DEFAULT_ORF_AA_SEQUENCE_MAX_LEN})."
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push generated examples to the Hub.",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default="hf-carbon/seqqa-synth",
        help="Hub dataset repo for --push-to-hub.",
    )
    parser.add_argument(
        "--hub-config",
        type=str,
        default="v0",
        help="Hub config name for --push-to-hub.",
    )
    parser.add_argument(
        "--length-distribution",
        choices=["uniform", "normal"],
        default="uniform",
        help=(
            "How to sample DNA sequence lengths within each subtask's range. "
            "'uniform' samples uniformly; 'normal' uses a truncated normal "
            "centred on the midpoint (default: uniform)."
        ),
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=None,
        help="Override minimum DNA sequence length for all subtasks.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Override maximum DNA sequence length for all subtasks.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the interactive progress bar.",
    )
    args = parser.parse_args()

    if not args.output and not args.push_to_hub:
        parser.error("Specify --output, --push-to-hub, or both.")
    if args.max_orf_aa_seq_len < ORF_AA_SEQUENCE_MIN_LEN:
        parser.error(
            "--max-orf-aa-seq-len must be >= "
            f"{ORF_AA_SEQUENCE_MIN_LEN}."
        )
    if (
        args.min_len is not None
        and args.max_len is not None
        and args.max_len < args.min_len
    ):
        parser.error("--max-len must be >= --min-len.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    global _default_length_distribution, _default_min_len, _default_max_len  # noqa: PLW0603
    _default_length_distribution = args.length_distribution
    if args.min_len is not None:
        _default_min_len = args.min_len
    if args.max_len is not None:
        _default_max_len = args.max_len

    examples = generate_all(
        n_per_subtask=args.n_per_subtask,
        seed=args.seed,
        subtasks=args.subtasks,
        show_progress=not args.no_progress,
        generator_kwargs_by_subtask={
            "orf_aa_sequence": {
                "max_aa_len": args.max_orf_aa_seq_len,
            }
        },
    )

    if args.output:
        output_path = Path(args.output)
        write_jsonl(output_path, examples)
        logger.info("Wrote %d examples to %s", len(examples), output_path)

    if args.push_to_hub:
        push_examples_to_hub(examples, args.hub_repo, args.hub_config)

    print(f"Generated {len(examples)} examples across subtasks.")
    for subtask, count in sorted(Counter(row["subtask"] for row in examples).items()):
        print(f"  {subtask}: {count}")


if __name__ == "__main__":
    main()
