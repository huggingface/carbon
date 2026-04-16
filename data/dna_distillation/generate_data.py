#!/usr/bin/env python3
"""Generate a balanced DNA distillation subset from eukaryote pretraining data.

Example:
    uv run --project data --script data/dna_distillation/generate_data.py \
        --num-samples-per-species 1024 \
        --prompt-len 6144 \
        --min-completion-len 30 \
        --max-completion-len 960 \
        --no-tags-frac 0.5 \
        --both-tags-frac 0.16666666666666666 \
        --species-only-frac 0.16666666666666666 \
        --gene-only-frac 0.16666666666666666 \
        --shuffle \
        --dataset-id hf-carbon/dna-distillation \
        --dataset-config default
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import heapq
import logging
import math
import time
from typing import Any

from datasets import Dataset
from huggingface_hub import list_repo_files
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SOURCE_DATASET_ID = "GenerTeam/pretrain_data_eukaryote"
SOURCE_COLUMN = "source"
TYPE_COLUMN = "type"
MAPPED_SPECIES_TAG_COLUMN = "mapped_species_tag"
MAPPED_GENE_TAG_COLUMN = "mapped_gene_tag"
TAG_MODE_COLUMN = "tag_mode"
PROMPT_COLUMN = "prompt"
COMPLETION_COLUMN = "completion"
PROMPT_LEN_COLUMN = "prompt_len"
COMPLETION_LEN_COLUMN = "completion_len"
SEQUENCE_COLUMN = "sequence"
DNA_PREFIX = "<dna>"
EXPECTED_SPECIES = (
    "fungi",
    "invertebrate",
    "plant",
    "protozoa",
    "vertebrate_mammalian",
    "vertebrate_other",
)
RAW_SPECIES_TAG_MAP = {
    "<mam>": "<mammalian_species>",
    "<vrt>": "<vertebrate_non_mammalian_species>",
    "<fng>": "<fungi_species>",
    "<pln>": "<plant_species>",
    "<prt>": "<protozoa_species>",
    "<inv>": "<invertebrate_species>",
}
RAW_GENE_TAG_MAP = {
    "<cds>": "<protein_coding_region>",
    "<pseudo>": "<pseudo_gene>",
    "<tRNA>": "<transfer_rna>",
    "<tmRNA>": "<transfer_messenger_rna>",
    "<ncRNA>": "<non_coding_rna>",
    "<misc_RNA>": "<miscellaneous_rna>",
    "<rRNA>": "<ribosomal_rna>",
}
TAG_MODE_NO_TAGS = "no_tags"
TAG_MODE_BOTH = "both"
TAG_MODE_SPECIES_ONLY = "species_only"
TAG_MODE_GENE_ONLY = "gene_only"
DEFAULT_METADATA_NO_TAGS_FRAC = 0.5
DEFAULT_METADATA_OTHER_FRAC = 1.0 / 6.0
DEFAULT_MIN_COMPLETION_LEN = 30
DEFAULT_MAX_COMPLETION_LEN = 960


@dataclass
class SamplingResult:
    species: str
    num_files: int
    rows_seen: int
    eligible_rows: int
    skipped_too_short: int
    skipped_no_context: int
    samples: list[dict[str, Any]]


@dataclass
class FileCandidateResult:
    species: str
    relative_path: str
    rows_seen: int
    eligible_rows: int
    skipped_too_short: int
    skipped_no_context: int
    candidate_entries: list[tuple[int, int]]


@dataclass(frozen=True, order=True)
class SelectedRowRef:
    priority: int
    relative_path: str
    row_index: int


def positive_int(value: str) -> int:
    """Parse a positive integer argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Subsample balanced species data from GenerTeam/pretrain_data_eukaryote "
            "and build prompt/completion pairs with mixed metadata prefixes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-samples-per-species",
        type=positive_int,
        required=True,
        help="Number of rows to sample for each of the six species folders.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Optional Hub dataset repo ID to push the sampled dataset to.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional Hub dataset config name for the pushed sampled dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sampling and optional shuffling.",
    )
    parser.add_argument(
        "--num-proc",
        type=positive_int,
        default=16,
        help="Requested process count for file-level sampling and prompt post-processing.",
    )
    parser.add_argument(
        "--prompt-len",
        type=positive_int,
        default=6144,
        help=(
            "Maximum DNA context length in characters / base pairs to keep in the prompt, "
            "excluding the <species_tag><dna> prefix."
        ),
    )
    parser.add_argument(
        "--min-completion-len",
        type=positive_int,
        default=DEFAULT_MIN_COMPLETION_LEN,
        help=(
            "Minimum DNA completion length in characters / base pairs. The sampled length is "
            "drawn uniformly per row from the inclusive range bounded by this value and "
            "--max-completion-len."
        ),
    )
    parser.add_argument(
        "--max-completion-len",
        type=positive_int,
        default=DEFAULT_MAX_COMPLETION_LEN,
        help=(
            "Maximum DNA completion length in characters / base pairs. If this equals "
            "--min-completion-len, all rows use a constant completion length."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the concatenated dataset after sampling all species.",
    )
    parser.add_argument(
        "--no-tags-frac",
        dest="metadata_no_tags_frac",
        type=float,
        metavar="FRAC",
        default=DEFAULT_METADATA_NO_TAGS_FRAC,
        help="Fraction of rows that use no metadata prefix before <dna>.",
    )
    parser.add_argument(
        "--both-tags-frac",
        dest="metadata_both_tags_frac",
        type=float,
        metavar="FRAC",
        default=DEFAULT_METADATA_OTHER_FRAC,
        help="Fraction of rows that use both species and gene metadata prefixes.",
    )
    parser.add_argument(
        "--species-only-frac",
        dest="metadata_species_only_frac",
        type=float,
        metavar="FRAC",
        default=DEFAULT_METADATA_OTHER_FRAC,
        help="Fraction of rows that use only the species metadata prefix.",
    )
    parser.add_argument(
        "--gene-only-frac",
        dest="metadata_gene_only_frac",
        type=float,
        metavar="FRAC",
        default=DEFAULT_METADATA_OTHER_FRAC,
        help="Fraction of rows that use only the gene metadata prefix.",
    )
    return parser.parse_args()


def validate_push_args(args: argparse.Namespace) -> None:
    """Ensure Hub push arguments are either both present or both absent."""
    has_dataset_id = bool(args.dataset_id)
    has_dataset_config = bool(args.dataset_config)
    if has_dataset_id != has_dataset_config:
        raise ValueError("--dataset-id and --dataset-config must be provided together.")


def validate_completion_length_args(args: argparse.Namespace) -> None:
    """Ensure the completion-length range is valid."""
    if args.min_completion_len > args.max_completion_len:
        raise ValueError(
            "--min-completion-len must be less than or equal to --max-completion-len."
        )


def validate_metadata_fraction_args(args: argparse.Namespace) -> None:
    """Ensure metadata fractions form a valid probability distribution."""
    metadata_fracs = {
        TAG_MODE_NO_TAGS: args.metadata_no_tags_frac,
        TAG_MODE_BOTH: args.metadata_both_tags_frac,
        TAG_MODE_SPECIES_ONLY: args.metadata_species_only_frac,
        TAG_MODE_GENE_ONLY: args.metadata_gene_only_frac,
    }
    invalid = {name: value for name, value in metadata_fracs.items() if value < 0.0 or value > 1.0}
    if invalid:
        details = ", ".join(f"{name}={value}" for name, value in sorted(invalid.items()))
        raise ValueError(f"Metadata fractions must be between 0 and 1 inclusive: {details}")

    total = sum(metadata_fracs.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        details = ", ".join(f"{name}={value}" for name, value in sorted(metadata_fracs.items()))
        raise ValueError(
            f"Metadata fractions must sum to 1.0. Got total={total} with values: {details}"
        )


def build_metadata_thresholds(args: argparse.Namespace) -> tuple[tuple[str, float], ...]:
    """Build cumulative thresholds for deterministic metadata mode assignment."""
    ordered = (
        (TAG_MODE_NO_TAGS, args.metadata_no_tags_frac),
        (TAG_MODE_BOTH, args.metadata_both_tags_frac),
        (TAG_MODE_SPECIES_ONLY, args.metadata_species_only_frac),
        (TAG_MODE_GENE_ONLY, args.metadata_gene_only_frac),
    )
    cumulative = 0.0
    thresholds: list[tuple[str, float]] = []
    for mode, fraction in ordered:
        cumulative += fraction
        thresholds.append((mode, cumulative))
    return tuple(thresholds)


def stable_row_value(base_seed: int, row: dict[str, Any], namespace: str) -> int:
    """Derive a deterministic integer from stable row identity and a namespace."""
    payload = (
        f"{namespace}:{base_seed}:{row.get('record_id', '')}:{row.get('start', '')}:{row.get('end', '')}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def stable_row_fraction(base_seed: int, row: dict[str, Any], namespace: str = "fraction") -> float:
    """Derive a deterministic unit-interval value from stable row identity."""
    return stable_row_value(base_seed, row, namespace) / 2**64


def sample_completion_length(
    row: dict[str, Any],
    seed: int,
    min_completion_len: int,
    max_completion_len: int,
) -> int:
    """Sample a deterministic completion length for one row."""
    if min_completion_len == max_completion_len:
        return min_completion_len

    span = max_completion_len - min_completion_len + 1
    draw = stable_row_value(seed, row, "completion_len")
    return min_completion_len + (draw % span)


def to_hf_path(repo_id: str, relative_path: str) -> str:
    """Convert a dataset-relative path into an hf:// parquet path."""
    return f"hf://datasets/{repo_id}/{relative_path}"


def stable_row_priority(
    base_seed: int,
    relative_path: str,
    row_index: int,
    row: dict[str, Any],
) -> int:
    """Derive a deterministic priority used for exact sampling without replacement."""
    payload = (
        f"{base_seed}:{relative_path}:{row_index}:"
        f"{row.get('record_id', '')}:{row.get('start', '')}:{row.get('end', '')}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def compute_usable_context_length(sequence: str, prompt_len: int, completion_len: int) -> int:
    """Return the usable DNA context length after reserving the completion suffix."""
    raw_context_length = len(sequence) - completion_len
    if raw_context_length < 6:
        return 0
    return (min(raw_context_length, prompt_len) // 6) * 6


def iter_parquet_batches(
    repo_id: str,
    relative_path: str,
    columns: list[str] | None = None,
    batch_size: int = 4096,
):
    """Iterate parquet batches from the Hub without loading the full file into memory."""
    parquet_file = pq.ParquetFile(to_hf_path(repo_id, relative_path))
    yield from parquet_file.iter_batches(batch_size=batch_size, columns=columns)


def map_metadata_tag(
    raw_value: Any,
    mapping: dict[str, str],
    column_name: str,
    row: dict[str, Any],
) -> str:
    """Map a raw metadata code to the tokenizer tag used in prompt construction."""
    if raw_value not in mapping:
        raise ValueError(
            f"Unsupported {column_name} value {raw_value!r} for record_id={row.get('record_id')!r} "
            f"start={row.get('start')!r} end={row.get('end')!r}."
        )
    return mapping[raw_value]


def validate_metadata_tags(row: dict[str, Any]) -> None:
    """Fail fast if the raw metadata codes cannot be mapped to tokenizer tags."""
    map_metadata_tag(
        row.get("species_type"),
        RAW_SPECIES_TAG_MAP,
        "species_type",
        row,
    )
    map_metadata_tag(
        row.get("gene_type"),
        RAW_GENE_TAG_MAP,
        "gene_type",
        row,
    )


def assign_tag_mode(
    row: dict[str, Any],
    seed: int,
    metadata_thresholds: tuple[tuple[str, float], ...],
) -> str:
    """Assign a deterministic metadata mode for one row."""
    draw = stable_row_fraction(seed, row, namespace="tag_mode")
    for mode, threshold in metadata_thresholds:
        if draw < threshold:
            return mode
    return metadata_thresholds[-1][0]


def classify_row_eligibility(
    row: dict[str, Any],
    prompt_len: int,
    seed: int,
    min_completion_len: int,
    max_completion_len: int,
) -> tuple[str | None, int]:
    """Return the skip reason and sampled completion length for one row."""
    completion_len = sample_completion_length(
        row=row,
        seed=seed,
        min_completion_len=min_completion_len,
        max_completion_len=max_completion_len,
    )
    sequence = row.get(SEQUENCE_COLUMN)
    if not isinstance(sequence, str) or len(sequence) < completion_len + 6:
        return "too_short", completion_len

    usable_context_length = compute_usable_context_length(
        sequence=sequence,
        prompt_len=prompt_len,
        completion_len=completion_len,
    )
    if usable_context_length == 0:
        return "no_context", completion_len
    return None, completion_len


def prepare_sampled_row(
    row: dict[str, Any],
    repo_id: str,
    species: str,
    prompt_len: int,
    seed: int,
    min_completion_len: int,
    max_completion_len: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Attach metadata and filter out rows that cannot form a valid prompt/completion pair."""
    skip_reason, completion_len = classify_row_eligibility(
        row=row,
        prompt_len=prompt_len,
        seed=seed,
        min_completion_len=min_completion_len,
        max_completion_len=max_completion_len,
    )
    if skip_reason is not None:
        return None, skip_reason

    row_dict = dict(row)
    row_dict[MAPPED_SPECIES_TAG_COLUMN] = map_metadata_tag(
        row_dict.get("species_type"),
        RAW_SPECIES_TAG_MAP,
        "species_type",
        row_dict,
    )
    row_dict[MAPPED_GENE_TAG_COLUMN] = map_metadata_tag(
        row_dict.get("gene_type"),
        RAW_GENE_TAG_MAP,
        "gene_type",
        row_dict,
    )
    row_dict[SOURCE_COLUMN] = repo_id
    row_dict[TYPE_COLUMN] = species
    row_dict[COMPLETION_LEN_COLUMN] = completion_len
    return row_dict, None


def discover_species_files(repo_id: str) -> dict[str, list[str]]:
    """Group all source parquet files by top-level species folder."""
    grouped: dict[str, list[str]] = {species: [] for species in EXPECTED_SPECIES}
    extras: dict[str, list[str]] = {}

    for path in list_repo_files(repo_id, repo_type="dataset"):
        if not path.endswith(".parq"):
            continue
        species, _, _ = path.partition("/")
        if species in grouped:
            grouped[species].append(path)
        else:
            extras.setdefault(species, []).append(path)

    missing = [species for species in EXPECTED_SPECIES if not grouped[species]]
    if missing:
        raise ValueError(f"Missing expected species folders: {', '.join(sorted(missing))}")
    if extras:
        raise ValueError(
            "Found unexpected species folders: "
            + ", ".join(sorted(extras))
        )

    for species in EXPECTED_SPECIES:
        grouped[species].sort()
    return grouped


def _maybe_add_file_candidate(
    heap: list[tuple[int, int]],
    sample_size: int,
    priority: int,
    row_index: int,
) -> None:
    """Keep the best sample_size candidate priorities for one file."""
    entry = (-priority, -row_index)
    if len(heap) < sample_size:
        heapq.heappush(heap, entry)
        return

    worst_priority = -heap[0][0]
    worst_row_index = -heap[0][1]
    if (priority, row_index) < (worst_priority, worst_row_index):
        heapq.heapreplace(heap, entry)


def scan_file_candidates(
    repo_id: str,
    species: str,
    relative_path: str,
    sample_size: int,
    seed: int,
    prompt_len: int,
    min_completion_len: int,
    max_completion_len: int,
) -> FileCandidateResult:
    """Scan one parquet file and keep its file-local best candidates."""
    rows_seen = 0
    eligible_rows = 0
    skipped_too_short = 0
    skipped_no_context = 0
    candidate_heap: list[tuple[int, int]] = []
    columns = [SEQUENCE_COLUMN, "species_type", "gene_type", "record_id", "start", "end"]

    for batch in iter_parquet_batches(repo_id, relative_path, columns=columns):
        batch_rows = batch.to_pydict()
        for offset in range(batch.num_rows):
            row_index = rows_seen
            row = {column: values[offset] for column, values in batch_rows.items()}
            rows_seen += 1

            skip_reason, _ = classify_row_eligibility(
                row=row,
                prompt_len=prompt_len,
                seed=seed,
                min_completion_len=min_completion_len,
                max_completion_len=max_completion_len,
            )
            if skip_reason == "too_short":
                skipped_too_short += 1
                continue
            if skip_reason == "no_context":
                skipped_no_context += 1
                continue

            validate_metadata_tags(row)
            eligible_rows += 1
            priority = stable_row_priority(
                base_seed=seed,
                relative_path=relative_path,
                row_index=row_index,
                row=row,
            )
            _maybe_add_file_candidate(
                heap=candidate_heap,
                sample_size=sample_size,
                priority=priority,
                row_index=row_index,
            )

    candidate_entries = sorted((-priority, -row_index) for priority, row_index in candidate_heap)
    return FileCandidateResult(
        species=species,
        relative_path=relative_path,
        rows_seen=rows_seen,
        eligible_rows=eligible_rows,
        skipped_too_short=skipped_too_short,
        skipped_no_context=skipped_no_context,
        candidate_entries=candidate_entries,
    )


def scan_all_file_candidates(
    repo_id: str,
    species_files: dict[str, list[str]],
    sample_size: int,
    seed: int,
    num_proc: int,
    prompt_len: int,
    min_completion_len: int,
    max_completion_len: int,
) -> list[FileCandidateResult]:
    """Scan every parquet file independently and keep file-local candidate rows."""
    file_tasks = [
        (species, relative_path)
        for species in EXPECTED_SPECIES
        for relative_path in species_files[species]
    ]
    max_workers = min(num_proc, max(1, len(file_tasks)))
    if max_workers == 1:
        return [
            scan_file_candidates(
                repo_id=repo_id,
                species=species,
                relative_path=relative_path,
                sample_size=sample_size,
                seed=seed,
                prompt_len=prompt_len,
                min_completion_len=min_completion_len,
                max_completion_len=max_completion_len,
            )
            for species, relative_path in file_tasks
        ]

    results: list[FileCandidateResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                scan_file_candidates,
                repo_id,
                species,
                relative_path,
                sample_size,
                seed,
                prompt_len,
                min_completion_len,
                max_completion_len,
            ): (species, relative_path)
            for species, relative_path in file_tasks
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def aggregate_file_candidates(
    file_results: list[FileCandidateResult],
    sample_size: int,
) -> tuple[dict[str, SamplingResult], dict[str, list[SelectedRowRef]]]:
    """Merge file-local candidates into exact global per-species selections."""
    species_heaps: dict[str, list[tuple[int, str, int]]] = {
        species: [] for species in EXPECTED_SPECIES
    }
    stats_by_species = {
        species: SamplingResult(
            species=species,
            num_files=0,
            rows_seen=0,
            eligible_rows=0,
            skipped_too_short=0,
            skipped_no_context=0,
            samples=[],
        )
        for species in EXPECTED_SPECIES
    }

    for file_result in file_results:
        stats = stats_by_species[file_result.species]
        stats.num_files += 1
        stats.rows_seen += file_result.rows_seen
        stats.eligible_rows += file_result.eligible_rows
        stats.skipped_too_short += file_result.skipped_too_short
        stats.skipped_no_context += file_result.skipped_no_context

        species_heap = species_heaps[file_result.species]
        for priority, row_index in file_result.candidate_entries:
            entry = (-priority, file_result.relative_path, -row_index)
            if len(species_heap) < sample_size:
                heapq.heappush(species_heap, entry)
                continue

            worst_priority = -species_heap[0][0]
            worst_row_index = -species_heap[0][2]
            if (priority, row_index) < (worst_priority, worst_row_index):
                heapq.heapreplace(species_heap, entry)

    selected_refs_by_species: dict[str, list[SelectedRowRef]] = {}
    for species in EXPECTED_SPECIES:
        stats = stats_by_species[species]
        if stats.eligible_rows < sample_size:
            raise ValueError(
                f"Species {species} has only {stats.eligible_rows} eligible rows, fewer than "
                f"requested {sample_size}."
            )

        refs = [
            SelectedRowRef(
                priority=-priority,
                relative_path=relative_path,
                row_index=-row_index,
            )
            for priority, relative_path, row_index in species_heaps[species]
        ]
        refs.sort()
        selected_refs_by_species[species] = refs

    return stats_by_species, selected_refs_by_species


def materialize_file_rows(
    repo_id: str,
    species: str,
    relative_path: str,
    row_indexes: list[int],
    prompt_len: int,
    seed: int,
    min_completion_len: int,
    max_completion_len: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Load the selected rows for one file and attach the sampled-row metadata."""
    remaining = set(row_indexes)
    found_rows: dict[int, dict[str, Any]] = {}
    row_index_base = 0

    for batch in iter_parquet_batches(repo_id, relative_path, columns=None):
        if not remaining:
            break

        batch_rows = batch.to_pydict()
        for offset in range(batch.num_rows):
            row_index = row_index_base + offset
            if row_index not in remaining:
                continue

            row = {column: values[offset] for column, values in batch_rows.items()}
            row_dict, skip_reason = prepare_sampled_row(
                row=row,
                repo_id=repo_id,
                species=species,
                prompt_len=prompt_len,
                seed=seed,
                min_completion_len=min_completion_len,
                max_completion_len=max_completion_len,
            )
            if row_dict is None:
                raise ValueError(
                    f"Selected row {relative_path}:{row_index} became invalid during materialization "
                    f"(skip_reason={skip_reason})."
                )

            found_rows[row_index] = row_dict
            remaining.remove(row_index)
            if not remaining:
                break

        row_index_base += batch.num_rows

    if remaining:
        missing = ", ".join(str(index) for index in sorted(remaining))
        raise ValueError(
            f"Failed to materialize selected rows from {relative_path}. Missing row indexes: {missing}"
        )

    return [(row_index, found_rows[row_index]) for row_index in row_indexes]


def materialize_selected_rows(
    repo_id: str,
    selected_refs_by_species: dict[str, list[SelectedRowRef]],
    prompt_len: int,
    seed: int,
    min_completion_len: int,
    max_completion_len: int,
    num_proc: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Materialize only the selected rows after exact sampling is complete."""
    refs_by_file: dict[str, list[int]] = {}
    species_by_file: dict[str, str] = {}
    for species, refs in selected_refs_by_species.items():
        for ref in refs:
            refs_by_file.setdefault(ref.relative_path, []).append(ref.row_index)
            species_by_file[ref.relative_path] = species

    file_tasks = sorted(refs_by_file)
    if not file_tasks:
        return {}

    max_workers = min(num_proc, max(1, len(file_tasks)))
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}

    if max_workers == 1:
        for relative_path in file_tasks:
            materialized_rows = materialize_file_rows(
                repo_id=repo_id,
                species=species_by_file[relative_path],
                relative_path=relative_path,
                row_indexes=sorted(refs_by_file[relative_path]),
                prompt_len=prompt_len,
                seed=seed,
                min_completion_len=min_completion_len,
                max_completion_len=max_completion_len,
            )
            for row_index, row in materialized_rows:
                rows_by_key[(relative_path, row_index)] = row
        return rows_by_key

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                materialize_file_rows,
                repo_id,
                species_by_file[relative_path],
                relative_path,
                sorted(refs_by_file[relative_path]),
                prompt_len,
                seed,
                min_completion_len,
                max_completion_len,
            ): relative_path
            for relative_path in file_tasks
        }
        for future in as_completed(futures):
            relative_path = futures[future]
            for row_index, row in future.result():
                rows_by_key[(relative_path, row_index)] = row

    return rows_by_key


def sample_all_species(
    repo_id: str,
    species_files: dict[str, list[str]],
    sample_size: int,
    seed: int,
    num_proc: int,
    prompt_len: int,
    min_completion_len: int,
    max_completion_len: int,
) -> list[SamplingResult]:
    """Sample each species independently, using file-level multiprocessing."""
    phase_start = time.monotonic()
    file_results = scan_all_file_candidates(
        repo_id=repo_id,
        species_files=species_files,
        sample_size=sample_size,
        seed=seed,
        num_proc=num_proc,
        prompt_len=prompt_len,
        min_completion_len=min_completion_len,
        max_completion_len=max_completion_len,
    )
    stats_by_species, selected_refs_by_species = aggregate_file_candidates(
        file_results=file_results,
        sample_size=sample_size,
    )
    logger.info("Phase sample_candidates completed in %.2fs", time.monotonic() - phase_start)

    phase_start = time.monotonic()
    rows_by_key = materialize_selected_rows(
        repo_id=repo_id,
        selected_refs_by_species=selected_refs_by_species,
        prompt_len=prompt_len,
        seed=seed,
        min_completion_len=min_completion_len,
        max_completion_len=max_completion_len,
        num_proc=num_proc,
    )
    logger.info(
        "Phase materialize_selected_rows completed in %.2fs",
        time.monotonic() - phase_start,
    )

    sampling_results: list[SamplingResult] = []
    for species in EXPECTED_SPECIES:
        refs = selected_refs_by_species[species]
        stats = stats_by_species[species]
        stats.samples = [rows_by_key[(ref.relative_path, ref.row_index)] for ref in refs]
        sampling_results.append(stats)
    return sampling_results


def add_prompt_completion_fields(
    row: dict[str, Any],
    prompt_len: int,
    seed: int,
    metadata_thresholds: tuple[tuple[str, float], ...],
) -> dict[str, Any]:
    """Add eval-aligned prompt and completion fields to a sampled row."""
    completion_len = row[COMPLETION_LEN_COLUMN]
    sequence = row[SEQUENCE_COLUMN]
    raw_context = sequence[:-completion_len]
    completion = sequence[-completion_len:]
    usable_context_length = compute_usable_context_length(
        sequence=sequence,
        prompt_len=prompt_len,
        completion_len=completion_len,
    )
    prompt_dna = raw_context[-usable_context_length:]
    tag_mode = assign_tag_mode(row, seed=seed, metadata_thresholds=metadata_thresholds)
    mapped_species_tag = row[MAPPED_SPECIES_TAG_COLUMN]
    mapped_gene_tag = row[MAPPED_GENE_TAG_COLUMN]
    # Keep the metadata-prefix cases aligned with the upstream Qwen3 hybrid
    # tokenization script, while splitting prompt and completion into separate
    # fields instead of emitting a single <dna>...</dna> string:
    # https://github.com/huggingface/moe-lab/blob/59d8eb63b671d7ea6a52e6bd04fef7503594b1b8/examples/carbon/tokenization_scripts/qwen3_hybrid_tokenizer/tokenize_generdata_metadata_megatron_qwen3h.py#L92
    if tag_mode == TAG_MODE_NO_TAGS:
        prompt_prefix = DNA_PREFIX
    elif tag_mode == TAG_MODE_BOTH:
        prompt_prefix = mapped_species_tag + mapped_gene_tag + DNA_PREFIX
    elif tag_mode == TAG_MODE_SPECIES_ONLY:
        prompt_prefix = mapped_species_tag + DNA_PREFIX
    elif tag_mode == TAG_MODE_GENE_ONLY:
        prompt_prefix = mapped_gene_tag + DNA_PREFIX
    else:
        raise ValueError(f"Unsupported tag_mode {tag_mode!r}")
    return {
        TAG_MODE_COLUMN: tag_mode,
        PROMPT_COLUMN: prompt_prefix + prompt_dna,
        COMPLETION_COLUMN: completion,
        PROMPT_LEN_COLUMN: len(prompt_dna),
        COMPLETION_LEN_COLUMN: len(completion),
    }


def build_dataset(
    sampling_results: list[SamplingResult],
    shuffle: bool,
    seed: int,
    prompt_len: int,
    num_proc: int,
    metadata_thresholds: tuple[tuple[str, float], ...],
) -> Dataset:
    """Create the final combined dataset from per-species samples."""
    rows: list[dict[str, Any]] = []
    for result in sampling_results:
        rows.extend(result.samples)

    dataset = Dataset.from_list(rows)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    map_num_proc = min(num_proc, max(1, len(dataset)))
    map_kwargs: dict[str, Any] = {
        "fn_kwargs": {
            "prompt_len": prompt_len,
            "seed": seed,
            "metadata_thresholds": metadata_thresholds,
        },
        "desc": "Deriving prompt/completion fields",
    }
    if map_num_proc > 1:
        map_kwargs["num_proc"] = map_num_proc
    dataset = dataset.map(add_prompt_completion_fields, **map_kwargs)
    return dataset


def push_dataset(dataset: Dataset, args: argparse.Namespace) -> None:
    """Push the final dataset to the Hub."""
    dataset.push_to_hub(
        args.dataset_id,
        config_name=args.dataset_config,
        split="train",
        commit_message=(
            f"Upload balanced DNA distillation subset "
            f"({len(dataset)} rows, {args.num_samples_per_species} per species)"
        ),
    )
    logger.info(
        "Pushed %d rows to %s / %s",
        len(dataset),
        args.dataset_id,
        args.dataset_config,
    )


def log_sampling_summary(
    species_files: dict[str, list[str]],
    sampling_results: list[SamplingResult],
) -> None:
    """Log source file counts and final sampling stats."""
    for species in EXPECTED_SPECIES:
        logger.info("Discovered %d parquet files for %s", len(species_files[species]), species)
    for result in sampling_results:
        logger.info(
            "Sampled %d rows from %s across %d eligible rows (%d seen, %d too short, %d no-context) in %d files",
            len(result.samples),
            result.species,
            result.eligible_rows,
            result.rows_seen,
            result.skipped_too_short,
            result.skipped_no_context,
            result.num_files,
        )


def run(args: argparse.Namespace) -> int:
    """Execute the sampling workflow."""
    validate_push_args(args)
    validate_completion_length_args(args)
    validate_metadata_fraction_args(args)
    metadata_thresholds = build_metadata_thresholds(args)

    phase_start = time.monotonic()
    species_files = discover_species_files(SOURCE_DATASET_ID)
    logger.info("Phase discover_files completed in %.2fs", time.monotonic() - phase_start)

    sampling_results = sample_all_species(
        repo_id=SOURCE_DATASET_ID,
        species_files=species_files,
        sample_size=args.num_samples_per_species,
        seed=args.seed,
        num_proc=args.num_proc,
        prompt_len=args.prompt_len,
        min_completion_len=args.min_completion_len,
        max_completion_len=args.max_completion_len,
    )
    log_sampling_summary(species_files, sampling_results)

    phase_start = time.monotonic()
    dataset = build_dataset(
        sampling_results=sampling_results,
        shuffle=args.shuffle,
        seed=args.seed,
        prompt_len=args.prompt_len,
        num_proc=args.num_proc,
        metadata_thresholds=metadata_thresholds,
    )
    logger.info("Phase build_dataset completed in %.2fs", time.monotonic() - phase_start)
    logger.info("Built dataset with %d total rows", len(dataset))

    if args.dataset_id and args.dataset_config:
        phase_start = time.monotonic()
        push_dataset(dataset, args)
        logger.info("Phase push_dataset completed in %.2fs", time.monotonic() - phase_start)
    else:
        logger.info("Dry run only; not pushing to the Hub.")

    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
