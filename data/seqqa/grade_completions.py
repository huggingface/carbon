#!/usr/bin/env python3
"""Grade the processed SeqQA completion subset and push it as a graded config.

Usage::

    uv run --project data python data/seqqa/grade_completions.py \
        --source-repo hf-carbon/seqqa-sft_gpt-oss-120b \
"""

import argparse
from collections import Counter
import logging
from pathlib import Path
import re
import sys
from typing import Any

from datasets import Dataset, load_dataset

try:
    from seqqa import generate_data
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT / "data"))
    from seqqa import generate_data  # noqa: E402

logger = logging.getLogger(__name__)
GRADED_CONFIG = "graded"
GRADED_CORRECT_CONFIG = "graded_correct"
GRADED_SPLIT = "train"
OPTIONAL_GRADED_KEYS = {
    "validator_type",
    "answer_regex",
}
GRADED_MAP_COLUMNS = (
    "id",
    "question",
    "ideal",
    "prompt_suffix",
    "prompt",
    "subtask",
    "source",
    "validator_type",
    "validator_params",
    "answer_regex",
    "question_len",
    "ideal_len",
    "messages",
    "model_completion",
    "model_completion_tokens",
    "model_finish_reason",
    "model_answer",
    "grading_method",
    "grading_reference_source",
    "grading_reference_answer",
    "grading_source_ideal_matches_reference",
    "grading_is_correct",
    "answer_extraction_method",
)

MIN_DNA_TOKEN_LEN = 12
MIN_DNA_SEQUENCE_LEN = 20
MIN_AA_SEQUENCE_LEN = generate_data.ORF_AA_SEQUENCE_MIN_LEN
ALL_SUBTASKS = {
    "amplicon_length_primers",
    "amplicon_sequence",
    "amplicon_target_primers",
    "gibson_primer_design",
    "orf_aa_position",
    "orf_aa_sequence",
    "orf_count_over_threshold",
    "primer_pair_amplicon_length",
    "restriction_clone_primer_design",
    "restriction_fragment_count",
    "restriction_fragment_lengths",
    "seq_gc_pct",
    "translation_efficiency",
    "translation_upstream_aug_count",
    "vector_insert_compatibility",
}
ENZYME_RE = re.compile(
    r"\b(?:" + "|".join(generate_data.TRAINING_CLONING_ENZYME_NAMES) + r")\b",
    re.IGNORECASE,
)
CANONICAL_ENZYME_NAMES = {
    enzyme_name.lower(): enzyme_name
    for enzyme_name in generate_data.TRAINING_CLONING_ENZYME_NAMES
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Load a processed SeqQA completion subset, grade each row by "
            "recomputing the expected answer from validator_params, and "
            "push the graded subset back to the same dataset repo."
        )
    )
    parser.add_argument(
        "--source-repo",
        required=True,
        help="Source dataset repo id.",
    )
    parser.add_argument(
        "--source-config",
        default="processed",
        help="Source config to read (default: processed).",
    )
    parser.add_argument(
        "--source-split",
        default="train",
        help="Source split to read (default: train).",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=12,
        help="Worker count for parallel grading via datasets.map (default: 12).",
    )
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push the graded subset to the same source repo (default: enabled).",
    )
    return parser.parse_args()


def extract_answer_from_completion(completion: str) -> tuple[str, str]:
    """Extract the model answer from the completion text."""
    closed_matches = list(
        re.finditer(r"<answer\b[^>]*>(.*?)</answer>", completion, flags=re.IGNORECASE | re.DOTALL)
    )
    if closed_matches:
        return _clean_extracted_answer(closed_matches[-1].group(1)), "xml_closed"

    open_matches = list(re.finditer(r"<answer\b[^>]*>", completion, flags=re.IGNORECASE))
    if open_matches:
        tail = completion[open_matches[-1].end() :]
        if "<" in tail:
            tail = tail.split("<", 1)[0]
        return _clean_extracted_answer(tail), "xml_open"

    return _clean_extracted_answer(completion), "fallback_full_completion"


def _clean_extracted_answer(answer: str) -> str:
    """Trim common markdown wrappers from an extracted answer."""
    cleaned = answer.strip()
    while cleaned.startswith("```"):
        newline = cleaned.find("\n")
        if newline < 0:
            cleaned = cleaned[3:].strip()
            break
        cleaned = cleaned[newline + 1 :].strip()
    while cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()
    return cleaned.strip()


def canonicalize_integer(answer: str) -> int | None:
    """Extract the first integer from an answer."""
    match = re.search(r"-?\d+", answer.replace(",", ""))
    return int(match.group(0)) if match else None


def canonicalize_integer_list(answer: str) -> tuple[int, ...] | None:
    """Extract an ordered integer tuple from an answer."""
    values = re.findall(r"-?\d+", answer)
    return tuple(int(value) for value in values) if values else None


def _sanitize_dna_token(text: str) -> str | None:
    token = re.sub(r"[^ACGTU]", "", text.upper())
    if len(token) >= MIN_DNA_TOKEN_LEN:
        return token
    return None


def canonicalize_dna_pair(answer: str) -> tuple[str, str] | None:
    """Extract an ordered DNA pair from a primer-style answer."""
    forward = re.search(
        r"forward primer(?:\s*\([^)]*\))?\s*[:=]\s*([^\n;]+)",
        answer,
        flags=re.IGNORECASE,
    )
    reverse = re.search(
        r"reverse primer(?:\s*\([^)]*\))?\s*[:=]\s*([^\n;]+)",
        answer,
        flags=re.IGNORECASE,
    )
    if forward and reverse:
        forward_seq = _sanitize_dna_token(forward.group(1))
        reverse_seq = _sanitize_dna_token(reverse.group(1))
        if forward_seq and reverse_seq:
            return forward_seq, reverse_seq

    candidates: list[str] = []
    for chunk in re.split(r"[,;/\n]", answer):
        token = _sanitize_dna_token(chunk)
        if token:
            candidates.append(token)
    if len(candidates) >= 2:
        return candidates[0], candidates[1]

    compact = re.sub(r"\s+", "", answer.upper())
    fallback = re.findall(rf"[ACGTU]{{{MIN_DNA_TOKEN_LEN},}}", compact)
    if len(fallback) >= 2:
        return fallback[0], fallback[1]

    return None


def canonicalize_enzyme_pair(answer: str) -> tuple[str, str] | None:
    """Extract a sorted enzyme pair from an answer."""
    seen: list[str] = []
    for match in ENZYME_RE.findall(answer):
        canonical = CANONICAL_ENZYME_NAMES.get(match.lower())
        if canonical is None:
            continue
        if canonical not in seen:
            seen.append(canonical)
    if len(seen) < 2:
        return None
    return tuple(sorted(seen[:2]))


def canonicalize_amino_acid_position(answer: str) -> str | None:
    """Normalize one-letter, three-letter, or full-name amino-acid answers."""
    stripped = answer.strip()
    if stripped in generate_data.AA_ONE_TO_THREE:
        return generate_data.AA_ONE_TO_THREE[stripped]

    for three_letter in generate_data.AA_ONE_TO_THREE.values():
        if stripped.lower() == three_letter.lower():
            return three_letter

    full_names = {
        "alanine": "Ala",
        "arginine": "Arg",
        "asparagine": "Asn",
        "aspartic acid": "Asp",
        "cysteine": "Cys",
        "glutamine": "Gln",
        "glutamic acid": "Glu",
        "glycine": "Gly",
        "histidine": "His",
        "isoleucine": "Ile",
        "leucine": "Leu",
        "lysine": "Lys",
        "methionine": "Met",
        "phenylalanine": "Phe",
        "proline": "Pro",
        "serine": "Ser",
        "threonine": "Thr",
        "tryptophan": "Trp",
        "tyrosine": "Tyr",
        "valine": "Val",
    }
    for full_name, three_letter in sorted(
        full_names.items(),
        key=lambda item: -len(item[0]),
    ):
        if re.search(rf"\b{re.escape(full_name)}\b", stripped, flags=re.IGNORECASE):
            return three_letter

    return None


def canonicalize_rna_sequence(answer: str) -> str | None:
    """Strip non-RNA characters and return the cleaned sequence."""
    letters = re.sub(r"[^ACGU]", "", answer.upper())
    if len(letters) < 10:
        return None
    return letters


def canonicalize_dna_sequence(answer: str) -> str | None:
    """Strip non-DNA characters and normalize U to T."""
    letters = re.sub(r"[^ACGTU]", "", answer.upper()).replace("U", "T")
    if len(letters) < MIN_DNA_SEQUENCE_LEN:
        return None
    return letters


def canonicalize_transcript_label(answer: str) -> str | None:
    """Extract a transcript label like 'Transcript A' from an answer."""
    match = re.search(r"Transcript\s+([A-Da-d])", answer, flags=re.IGNORECASE)
    if not match:
        return None
    return f"Transcript {match.group(1).upper()}"


def canonicalize_amino_acid_sequence(answer: str) -> str | None:
    """Strip non-amino-acid characters and a terminal stop marker."""
    letters = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY*]", "", answer.upper()).rstrip("*")
    if len(letters) < MIN_AA_SEQUENCE_LEN:
        return None
    return letters


CANONICALIZERS = {
    "amplicon_length_primers": canonicalize_dna_pair,
    "amplicon_sequence": canonicalize_dna_sequence,
    "amplicon_target_primers": canonicalize_dna_pair,
    "gibson_primer_design": canonicalize_dna_pair,
    "orf_aa_position": canonicalize_amino_acid_position,
    "orf_aa_sequence": canonicalize_amino_acid_sequence,
    "orf_count_over_threshold": canonicalize_integer,
    "primer_pair_amplicon_length": canonicalize_integer,
    "restriction_clone_primer_design": canonicalize_dna_pair,
    "restriction_fragment_count": canonicalize_integer,
    "restriction_fragment_lengths": canonicalize_integer_list,
    "seq_gc_pct": canonicalize_integer,
    "translation_efficiency": canonicalize_transcript_label,
    "translation_upstream_aug_count": canonicalize_integer,
    "vector_insert_compatibility": canonicalize_enzyme_pair,
}
# ---------------------------------------------------------------------------
# Validator-params-based recomputation
# ---------------------------------------------------------------------------


def _require_stop(params: dict[str, Any]) -> bool:
    """Return whether ORF recomputation should require a terminal stop codon."""
    return bool(params.get("require_stop", True))


def _require_unique_longest_orf(params: dict[str, Any]) -> bool:
    """Return whether longest-ORF tasks require a unique maximum-length ORF."""
    return params.get("longest_orf_requirement", "unique_longest_aa") == "unique_longest_aa"


def _recompute_from_params(subtask: str, params: dict[str, Any]) -> str:
    """Recompute the expected answer from validator_params metadata."""
    if subtask == "seq_gc_pct":
        return str(generate_data.gc_percent(params["sequence"]))

    if subtask == "amplicon_sequence":
        amplicon = generate_data.extract_amplicon_sequence(
            params["template"], params["forward_primer"], params["reverse_primer"]
        )
        if amplicon is None:
            raise ValueError("Could not compute amplicon sequence from validator params.")
        return amplicon

    if subtask == "restriction_fragment_count":
        fragments = generate_data.digest(params["sequence"], params["enzymes"])
        return str(len(fragments))

    if subtask == "restriction_fragment_lengths":
        fragments = generate_data.digest(params["sequence"], params["enzymes"])
        if params.get("ordering") == "sorted_ascending":
            fragments = sorted(fragments)
        return ", ".join(str(length) for length in fragments)

    if subtask == "orf_aa_position":
        orf = generate_data.longest_orf(
            params["sequence"],
            require_stop=_require_stop(params),
            require_unique=_require_unique_longest_orf(params),
        )
        if orf is None:
            raise ValueError("No ORF found when recomputing amino-acid position.")
        aa_seq = str(orf["aa_seq"])
        position = int(params["position"])
        if position < 1 or position > len(aa_seq):
            raise ValueError("Requested amino-acid position is out of range.")
        return generate_data.AA_ONE_TO_THREE[aa_seq[position - 1]]

    if subtask == "orf_aa_sequence":
        orf = generate_data.longest_orf(
            params["sequence"],
            require_stop=_require_stop(params),
            require_unique=_require_unique_longest_orf(params),
        )
        if orf is None:
            raise ValueError("No ORF found when recomputing amino-acid sequence.")
        return str(orf["aa_seq"])

    if subtask == "orf_count_over_threshold":
        return str(
            generate_data.count_orfs_strictly_over_threshold(
                params["sequence"],
                int(params["threshold"]),
                require_stop=_require_stop(params),
            )
        )

    if subtask == "translation_upstream_aug_count":
        return str(generate_data.count_upstream_aug_codons(params["leader_rna"]))

    if subtask == "translation_efficiency":
        labels = params["candidate_labels"]
        pre3s = params["candidate_pre3"]
        post1s = params["candidate_post1"]
        tiers = [
            generate_data.kozak_tier(pre3, post1)
            for pre3, post1 in zip(pre3s, post1s)
        ]
        best_idx = tiers.index(min(tiers))
        return labels[best_idx]

    if subtask == "primer_pair_amplicon_length":
        length = generate_data.compute_amplicon_length(
            params["template"], params["forward_primer"], params["reverse_primer"]
        )
        if length is None:
            raise ValueError("Could not compute amplicon length from validator params.")
        return str(length)

    if subtask == "restriction_clone_primer_design":
        fwd, rev = generate_data.design_re_primers(
            params["insert_sequence"],
            params["enzyme_5p"],
            params["enzyme_3p"],
            binding_len=int(params["binding_len"]),
            pad_len=int(params["pad_len"]),
        )
        return f"{fwd}, {rev}"

    if subtask == "amplicon_target_primers":
        target = params["target"]
        primer_length = int(params["primer_length"])
        fwd = target[:primer_length]
        from Bio.Seq import Seq as _Seq

        rev = str(_Seq(target[-primer_length:]).reverse_complement())
        return f"{fwd}, {rev}"

    if subtask == "amplicon_length_primers":
        candidates = generate_data.enumerate_amplicon_primer_pair_candidates(
            params["template"],
            int(params["amplicon_length"]),
            binding_len=int(params["primer_length"]),
        )
        if not candidates:
            raise ValueError("No valid primer pair candidates found.")
        _, fwd, rev = candidates[0]
        return f"{fwd}, {rev}"

    if subtask == "vector_insert_compatibility":
        valid_pairs = generate_data.find_unique_compatible_vector_enzyme_pairs(
            params["vector_sequence"],
            params["insert_sequence"],
            params["listed_enzymes"],
        )
        if not valid_pairs:
            raise ValueError("No compatible enzyme pairs found.")
        return ", ".join(valid_pairs[0])

    if subtask == "gibson_primer_design":
        fwd, rev = generate_data.design_gibson_primers(
            params["insert_sequence"],
            params["left_flank"],
            params["right_flank"],
            overlap_len=int(params["overlap_len"]),
            binding_len=int(params["binding_len"]),
        )
        return f"{fwd}, {rev}"

    raise ValueError(f"Unsupported subtask {subtask!r} for validator-params recomputation.")


def recompute_expected_answer(
    subtask: str,
    validator_params: dict[str, Any],
) -> str:
    """Recompute the expected answer from validator_params metadata."""
    return _recompute_from_params(subtask, validator_params)


def grade_row(row: dict[str, Any]) -> dict[str, Any]:
    """Grade one processed completion row."""
    subtask = str(row["subtask"])
    if subtask not in ALL_SUBTASKS:
        raise ValueError(f"Unsupported subtask {subtask!r}.")

    question = str(row["question"])
    ideal = str(row["ideal"])
    model_completion = str(row.get("model_completion") or "")
    model_answer = str(row.get("model_answer") or "").strip()
    extraction_method = "existing_model_answer"
    if not model_answer:
        model_answer, extraction_method = extract_answer_from_completion(model_completion)
    if not model_answer:
        raise ValueError("Model answer is empty after extraction.")

    validator_params: dict[str, Any] = row["validator_params"]

    canonicalizer = CANONICALIZERS[subtask]
    model_canonical = canonicalizer(model_answer)
    reference_answer = recompute_expected_answer(subtask, validator_params)
    reference_source = "recomputed_from_validator_params"
    ideal_canonical = canonicalizer(ideal)
    reference_canonical = canonicalizer(reference_answer)
    if ideal_canonical is None or reference_canonical is None:
        raise ValueError("Could not canonicalize recomputed comparison values.")
    source_matches_reference = ideal_canonical == reference_canonical
    if not source_matches_reference:
        raise ValueError("Stored ideal does not match recomputed answer.")
    is_correct = model_canonical is not None and reference_canonical == model_canonical
    grading_method = "recomputed"

    prompt_suffix = str(row.get("prompt_suffix") or "")
    prompt = str(row.get("prompt") or "")
    if not prompt:
        prompt = f"{question}\n\n{prompt_suffix}" if prompt_suffix else question

    graded_row: dict[str, Any] = {
        "id": str(row["id"]),
        "question": question,
        "ideal": ideal,
        "prompt_suffix": prompt_suffix,
        "prompt": prompt,
        "subtask": subtask,
        "source": str(row.get("source") or ""),
    }

    # Preserve generator metadata in the same order as generate_data.py.
    if row.get("validator_type"):
        graded_row["validator_type"] = row["validator_type"]
    graded_row["validator_params"] = validator_params
    if row.get("answer_regex"):
        graded_row["answer_regex"] = row["answer_regex"]

    graded_row["question_len"] = int(row.get("question_len") or len(question))
    graded_row["ideal_len"] = int(row.get("ideal_len") or len(ideal))
    graded_row["messages"] = list(row.get("messages") or [])
    graded_row["model_completion"] = model_completion
    graded_row["model_completion_tokens"] = row.get("model_completion_tokens")
    graded_row["model_finish_reason"] = row.get("model_finish_reason")
    graded_row["model_answer"] = model_answer

    graded_row["grading_method"] = grading_method
    graded_row["grading_reference_source"] = reference_source
    graded_row["grading_reference_answer"] = reference_answer
    graded_row["grading_source_ideal_matches_reference"] = source_matches_reference
    graded_row["grading_is_correct"] = is_correct
    graded_row["answer_extraction_method"] = extraction_method

    return graded_row


def _safe_grade_row(row: dict[str, Any]) -> dict[str, Any]:
    """Grade one row without aborting the entire map on exceptions."""
    result = {key: None for key in GRADED_MAP_COLUMNS}
    result["_grading_ok"] = False
    result["_grading_error"] = ""
    result["_grading_error_type"] = ""
    try:
        graded_row = grade_row(dict(row))
        result.update(graded_row)
        result["_grading_ok"] = True
        return result
    except Exception as error:  # noqa: BLE001
        result["id"] = str(row.get("id") or "")
        result["_grading_error"] = str(error)
        result["_grading_error_type"] = type(error).__name__
        return result


def process_rows(dataset: Dataset, num_proc: int | None = None) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Grade all rows in a processed completion dataset."""
    stats: Counter[str] = Counter()
    graded_rows: list[dict[str, Any]] = []

    graded_dataset = dataset.map(
        _safe_grade_row,
        remove_columns=list(dataset.column_names),
        num_proc=num_proc,
        desc="Grading completions",
    )

    for row in graded_dataset:
        if not row["_grading_ok"]:
            stats["skipped"] += 1
            stats[f"skipped:{row['_grading_error_type']}"] += 1
            logger.warning("Skipping row %s: %s", row.get("id"), row["_grading_error"])
            continue
        graded_row = {
            key: value
            for key, value in row.items()
            if not key.startswith("_grading_")
            and (key not in OPTIONAL_GRADED_KEYS or value is not None)
        }
        graded_rows.append(graded_row)
        stats["graded"] += 1
        stats[f"graded:{graded_row['subtask']}"] += 1
        stats[f"correct:{graded_row['subtask']}"] += int(graded_row["grading_is_correct"])
    return graded_rows, stats


def filter_correct_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only rows graded as correct."""
    return [row for row in rows if row.get("grading_is_correct")]


def push_rows(rows: list[dict[str, Any]], args: argparse.Namespace, *, config_name: str) -> None:
    """Push graded rows back to the same source dataset repo."""
    dataset = Dataset.from_list(rows)
    dataset.push_to_hub(
        args.source_repo,
        config_name=config_name,
        split=GRADED_SPLIT,
        commit_message=(
            f"Add {config_name} subset for {args.source_config} "
            f"({len(rows)} rows)"
        ),
    )
    logger.info(
        "Pushed %d rows to %s / %s / %s",
        len(rows),
        args.source_repo,
        config_name,
        GRADED_SPLIT,
    )


def run(args: argparse.Namespace) -> int:
    """Execute the grading workflow."""
    dataset = load_dataset(
        args.source_repo,
        args.source_config,
        split=args.source_split,
    )
    logger.info(
        "Loaded %d rows from %s / %s / %s",
        len(dataset),
        args.source_repo,
        args.source_config,
        args.source_split,
    )

    graded_rows, stats = process_rows(dataset, num_proc=args.num_proc)
    logger.info("Graded %d rows; skipped %d rows", stats["graded"], stats["skipped"])

    for subtask in sorted({row["subtask"] for row in graded_rows}):
        graded = stats[f"graded:{subtask}"]
        correct = stats[f"correct:{subtask}"]
        logger.info(
            "%s: %d graded, %d correct (%.3f)",
            subtask,
            graded,
            correct,
            correct / graded if graded else 0.0,
        )

    graded_correct_rows = filter_correct_rows(graded_rows)
    logger.info("Prepared %d correct graded rows", len(graded_correct_rows))

    if args.push:
        push_rows(graded_rows, args, config_name=GRADED_CONFIG)
        push_rows(graded_correct_rows, args, config_name=GRADED_CORRECT_CONFIG)
    else:
        print(
            f"Dry run: prepared {len(graded_rows)} graded rows for "
            f"{args.source_repo} / {GRADED_CONFIG} / {GRADED_SPLIT} "
            f"and {len(graded_correct_rows)} graded rows for "
            f"{args.source_repo} / {GRADED_CORRECT_CONFIG} / {GRADED_SPLIT}"
        )
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
