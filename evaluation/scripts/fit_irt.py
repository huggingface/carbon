"""Fit a 2PL IRT model for supported evaluation profiles.

Usage:
    uv run --directory evaluation python scripts/fit_irt.py \
        --profile seqqa \
        --collection hf-carbon/lighteval-outputs \
        --config lab_bench_seqqa_mcf_all_0 \
        --split latest \
        --device cpu

    uv run --directory evaluation python scripts/fit_irt.py \
        --profile basic_dna \
        --device cpu
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypedDict

import numpy as np
import pandas as pd
import pyro
import torch
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from huggingface_hub import HfApi

from _irt_2pl import IrtDataset, TwoParamIrtTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
# SeqQA metadata is joined against its canonical source dataset; basic_dna metadata
# comes directly from the local parquet details files, so it does not need analogous constants.
SEQQA_REPO_ID = "hf-carbon/lab-bench"
SEQQA_SUBSET = "SeqQA"
DETAILS_ROOT = REPO_ROOT / "details"
SEQQA_EXPECTED_ROWS = 600
BASIC_DNA_EXPECTED_ROWS = 200
SEQQA_ALWAYS_EXCLUDED_SUBSTRINGS = ("fsx",)


class IrtBestParams(TypedDict):
    item_ids: dict[int, str]
    subject_ids: dict[int, str]
    disc: list[float] | np.ndarray
    diff: list[float] | np.ndarray
    ability: list[float] | np.ndarray


@dataclass
class ManifestRow:
    status: str
    reason: str = ""
    model_org: str | None = None
    model_name: str | None = None
    row_count: int | None = None
    observed_accuracy: float | None = None
    repo_id: str | None = None
    item_type: str | None = None
    subject_id: str | None = None
    timestamp: str | None = None
    file_path: str | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "reason": self.reason,
            "model_org": self.model_org,
            "model_name": self.model_name,
            "row_count": self.row_count,
            "observed_accuracy": self.observed_accuracy,
        }
        if self.repo_id is not None:
            record["repo_id"] = self.repo_id
        if self.item_type is not None:
            record["item_type"] = self.item_type
        if self.subject_id is not None:
            record["subject_id"] = self.subject_id
        if self.timestamp is not None:
            record["timestamp"] = self.timestamp
        if self.file_path is not None:
            record["file_path"] = self.file_path
        return record


@dataclass(frozen=True)
class ReferenceItem:
    item_id: str
    query: str
    choices: list[str]
    gold_index: int
    task_name: str | None = None


@dataclass
class SubjectRow:
    subject_id: str
    model_org: str | None
    model_name: str | None
    responses: dict[str, int]
    observed_accuracy: float
    row_count: int
    timestamp: str | None = None


@dataclass
class LoadedSubject:
    manifest_key: str | tuple[str, str]
    subject_row: SubjectRow
    items: dict[str, ReferenceItem]


@dataclass(frozen=True)
class ProfileSpec:
    apply_defaults: Callable[[argparse.Namespace], None]
    validate_args: Callable[[argparse.Namespace], None]
    discover: Callable[[argparse.Namespace], tuple[list[object], list[ManifestRow]]]
    load_subject_rows: Callable[
        [argparse.Namespace, list[object], list[ManifestRow]],
        tuple[list[SubjectRow], dict[str, ReferenceItem]],
    ]
    build_metadata: Callable[[dict[str, ReferenceItem]], pd.DataFrame]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def validate_device(device: str) -> None:
    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid --device value: {device!r}") from exc
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested --device {device}, but CUDA is not available.")


def ensure_positive(value: int | None, flag_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{flag_name} must be positive")


def dataset_model_parts(dataset_repo_id: str) -> tuple[str, str]:
    _, separator, repo_name = dataset_repo_id.partition("/")
    if not separator or not repo_name:
        raise ValueError(f"Expected a dataset repo id like org/name, got: {dataset_repo_id!r}")

    model_stub = repo_name
    if model_stub.startswith("details_"):
        model_stub = model_stub[len("details_") :]
    for suffix in ("_private", "_public"):
        if model_stub.endswith(suffix):
            model_stub = model_stub[: -len(suffix)]
            break

    parts = model_stub.split("__")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            "Expected dataset repo name to look like "
            "details_{org}__{model_name}[_private|_public] with exactly one '__' separator, "
            f"got: {dataset_repo_id!r}"
        )
    return parts[0], parts[1]


def normalize_gold_index(value: object) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected a single gold index, got {value!r}")
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def normalize_choices(values: Sequence[object]) -> list[str]:
    return [str(value).strip() for value in values]


def model_parts(path: Path) -> tuple[str, str]:
    return path.parts[-4], path.parts[-3]


def model_id(path: Path) -> str:
    model_org, model_name = model_parts(path)
    return f"{model_org}/{model_name}"


def extract_doc_metric(
    row: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    doc = row["doc"]
    metric = row["metric"]
    if not isinstance(doc, Mapping) or not isinstance(metric, Mapping):
        raise RuntimeError("Expected row['doc'] and row['metric'] to be mappings")
    return doc, metric


def extract_binary_accuracy(
    metric: Mapping[str, object], *, subject_id: str, row_position: int
) -> int:
    if "acc" not in metric:
        raise RuntimeError(f"{subject_id} is missing metric.acc")
    acc = int(metric["acc"])
    if acc not in {0, 1}:
        raise RuntimeError(
            f"{subject_id} has a non-binary metric.acc value at row {row_position}: {acc}"
        )
    return acc


def build_seqqa_reference_item(doc: Mapping[str, object]) -> ReferenceItem:
    return ReferenceItem(
        item_id=str(doc["id"]),
        query=str(doc["query"]),
        choices=[str(choice) for choice in doc["choices"]],
        gold_index=normalize_gold_index(doc["gold_index"]),
    )


def build_basic_dna_reference_item(doc: Mapping[str, object]) -> ReferenceItem:
    choices_value = doc["choices"]
    if hasattr(choices_value, "tolist"):
        choices_value = choices_value.tolist()
    return ReferenceItem(
        item_id=str(doc["id"]),
        task_name=str(doc["task_name"]),
        query=str(doc["query"]),
        choices=[str(choice) for choice in choices_value],
        gold_index=normalize_gold_index(doc["gold_index"]),
    )


def build_loaded_subject(
    *,
    manifest_key: str | tuple[str, str],
    subject_id: str,
    model_org: str | None,
    model_name: str | None,
    records: Sequence[object],
    expected_row_count: int,
    extract_row: Callable[[Mapping[str, object]], tuple[Mapping[str, object], Mapping[str, object]]],
    build_reference_item: Callable[[Mapping[str, object]], ReferenceItem],
    timestamp: str | None = None,
) -> LoadedSubject:
    if len(records) != expected_row_count:
        raise RuntimeError(
            f"{subject_id} returned {len(records)} rows instead of {expected_row_count}"
        )

    responses: dict[str, int] = {}
    current_items: dict[str, ReferenceItem] = {}
    total_correct = 0

    for row_position, record in enumerate(records):
        doc, metric = extract_row(record)
        acc = extract_binary_accuracy(metric, subject_id=subject_id, row_position=row_position)
        item = build_reference_item(doc)
        if item.item_id in responses:
            raise RuntimeError(f"{subject_id} repeats doc.id={item.item_id}")
        responses[item.item_id] = acc
        current_items[item.item_id] = item
        total_correct += acc

    if not responses:
        raise RuntimeError(f"{subject_id} did not produce any responses")

    return LoadedSubject(
        manifest_key=manifest_key,
        subject_row=SubjectRow(
            subject_id=subject_id,
            model_org=model_org,
            model_name=model_name,
            responses=responses,
            observed_accuracy=total_correct / len(responses),
            row_count=len(responses),
            timestamp=timestamp,
        ),
        items=current_items,
    )


def validate_reference_items(
    *,
    subject_id: str,
    current_items: dict[str, ReferenceItem],
    reference_items: dict[str, ReferenceItem],
    compare_fields: Sequence[str],
) -> None:
    if set(current_items) != set(reference_items):
        missing_ids = sorted(set(reference_items) - set(current_items))
        extra_ids = sorted(set(current_items) - set(reference_items))
        raise RuntimeError(
            f"{subject_id} item ids differ from the reference set. "
            f"missing={missing_ids[:5]} extra={extra_ids[:5]}"
        )

    for item_id, current_item in current_items.items():
        expected_item = reference_items[item_id]
        for field_name in compare_fields:
            if getattr(current_item, field_name) != getattr(expected_item, field_name):
                raise RuntimeError(
                    f"{subject_id} mismatched {field_name} for item_id={item_id} "
                    "relative to the reference dataset"
                )


def finalize_loaded_subjects(
    loaded_subjects: list[LoadedSubject],
    *,
    manifest_by_key: dict[str | tuple[str, str], ManifestRow],
    compare_fields: Sequence[str],
) -> tuple[list[SubjectRow], dict[str, ReferenceItem]]:
    loaded_subjects.sort(key=lambda loaded: loaded.subject_row.subject_id)
    subject_rows: list[SubjectRow] = []
    reference_items: dict[str, ReferenceItem] = {}

    for loaded_subject in loaded_subjects:
        subject_row = loaded_subject.subject_row
        if not reference_items:
            reference_items = loaded_subject.items
        else:
            validate_reference_items(
                subject_id=subject_row.subject_id,
                current_items=loaded_subject.items,
                reference_items=reference_items,
                compare_fields=compare_fields,
            )

        if len(subject_row.responses) != len(reference_items):
            raise RuntimeError(
                f"{subject_row.subject_id} produced {len(subject_row.responses)} unique responses, "
                f"expected {len(reference_items)}"
            )

        subject_rows.append(subject_row)
        manifest_row = manifest_by_key[loaded_subject.manifest_key]
        manifest_row.row_count = subject_row.row_count
        manifest_row.observed_accuracy = subject_row.observed_accuracy

    if not subject_rows:
        raise RuntimeError("All discovered datasets failed to load.")

    final_keys = {loaded_subject.manifest_key for loaded_subject in loaded_subjects}
    for key, row in manifest_by_key.items():
        if row.status == "included" and key not in final_keys:
            row.status = "excluded"

    return subject_rows, reference_items


def discover_seqqa_datasets(
    *,
    collection_id: str,
    config: str,
    split: str,
    exclude_substrings: list[str],
    limit_datasets: int | None,
    max_workers: int,
) -> tuple[list[str], list[ManifestRow]]:
    api = HfApi()
    collection = api.get_collection(collection_id)
    items = sorted(collection.items, key=lambda entry: entry.item_id)
    effective_exclude_substrings = list(SEQQA_ALWAYS_EXCLUDED_SUBSTRINGS) + list(
        exclude_substrings
    )

    def inspect_item(item) -> ManifestRow:
        repo_id = item.item_id
        item_type = item.item_type
        try:
            model_org, model_name = dataset_model_parts(repo_id)
        except ValueError:
            model_org = None
            model_name = None

        def make_row(*, status: str, reason: str = "") -> ManifestRow:
            return ManifestRow(
                repo_id=repo_id,
                item_type=item_type,
                status=status,
                reason=reason,
                model_org=model_org,
                model_name=model_name,
            )

        if item_type != "dataset":
            return make_row(status="excluded", reason="not_dataset")
        if model_org is None or model_name is None:
            return make_row(status="excluded", reason="invalid_repo_name")

        repo_id_lower = repo_id.lower()
        matched_substring = next(
            (
                substring
                for substring in effective_exclude_substrings
                if substring.lower() in repo_id_lower
            ),
            None,
        )
        if matched_substring is not None:
            return make_row(
                status="excluded",
                reason=f"excluded_by_name:{matched_substring}",
            )

        try:
            config_names = set(get_dataset_config_names(repo_id))
        except Exception as exc:
            return make_row(
                status="excluded",
                reason=f"config_lookup_failed:{type(exc).__name__}",
            )
        if config not in config_names:
            return make_row(status="excluded", reason="missing_config")

        try:
            split_names = set(get_dataset_split_names(repo_id, config))
        except Exception as exc:
            return make_row(
                status="excluded",
                reason=f"split_lookup_failed:{type(exc).__name__}",
            )
        if split not in split_names:
            return make_row(status="excluded", reason="missing_split")

        return make_row(status="included")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        manifest_rows = list(executor.map(inspect_item, items))

    included_repo_ids = [row.repo_id for row in manifest_rows if row.status == "included"]
    if limit_datasets is not None:
        kept = set(included_repo_ids[:limit_datasets])
        for row in manifest_rows:
            if row.status == "included" and row.repo_id not in kept:
                row.status = "excluded"
                row.reason = "trimmed_by_limit"
        included_repo_ids = included_repo_ids[:limit_datasets]

    if not included_repo_ids:
        raise RuntimeError(
            "No datasets matched the requested collection/config/split filters."
        )

    return included_repo_ids, manifest_rows


def join_seqqa_reference_metadata(
    reference_items: dict[str, ReferenceItem],
) -> pd.DataFrame:
    source_dataset = load_dataset(SEQQA_REPO_ID, SEQQA_SUBSET, split="train")
    rows: list[dict[str, object]] = []
    expected_size = len(reference_items)
    if len(source_dataset) != expected_size:
        raise RuntimeError(
            f"SeqQA source dataset has {len(source_dataset)} rows, expected {expected_size}"
        )

    for item_id in sorted(reference_items, key=lambda value: int(value)):
        item = reference_items[item_id]
        try:
            row_index = int(item_id)
        except ValueError as exc:
            raise RuntimeError(f"doc.id is not an integer row index: {item_id}") from exc
        if row_index < 0 or row_index >= len(source_dataset):
            raise RuntimeError(f"doc.id={item_id} is out of range for SeqQA/train")

        source_row = source_dataset[row_index]
        if source_row["question"] not in item.query:
            raise RuntimeError(
                f"SeqQA question text did not match details query for item_id={item_id}"
            )
        if int(source_row["answer_index"]) != item.gold_index:
            raise RuntimeError(
                f"SeqQA gold index did not match details gold index for item_id={item_id}"
            )

        expected_choice_labels = [
            chr(ord("A") + offset) for offset in range(len(source_row["options"]))
        ]
        if normalize_choices(item.choices) != expected_choice_labels:
            raise RuntimeError(
                f"SeqQA choice labels did not match the expected MCF labels for item_id={item_id}"
            )

        rows.append(
            {
                "item_id": item_id,
                "seqqa_row_index": row_index,
                "seqqa_uuid": source_row["id"],
                "subtask": source_row["subtask"],
                "question": source_row["question"],
                "options": json.dumps(source_row["options"]),
                "answer_index": int(source_row["answer_index"]),
                "n_choices": len(source_row["options"]),
            }
        )

    return pd.DataFrame(rows)


def discover_basic_dna_datasets(
    *,
    details_glob: str,
    limit_models: int | None,
) -> tuple[list[Path], list[ManifestRow]]:
    all_paths = sorted(DETAILS_ROOT.glob(details_glob))
    if not all_paths:
        raise RuntimeError(f"No detail parquet files matched details/{details_glob}")

    latest_by_model: dict[str, Path] = {}
    for path in all_paths:
        subject_id = model_id(path)
        current = latest_by_model.get(subject_id)
        if current is None or path.parts[-2] > current.parts[-2]:
            latest_by_model[subject_id] = path

    included_paths = sorted(
        latest_by_model.values(),
        key=lambda path: (path.parts[-4].lower(), path.parts[-3].lower()),
    )
    if limit_models is not None:
        included_paths = included_paths[:limit_models]

    included_path_set = set(included_paths)
    manifest_rows: list[ManifestRow] = []
    for path in all_paths:
        subject_id = model_id(path)
        model_org, model_name = model_parts(path)
        included = path in included_path_set
        manifest_rows.append(
            ManifestRow(
                subject_id=subject_id,
                model_org=model_org,
                model_name=model_name,
                timestamp=path.parts[-2],
                status="included" if included else "excluded",
                reason="" if included else "superseded_by_newer_run",
                file_path=str(path.relative_to(REPO_ROOT)),
            )
        )

    return included_paths, manifest_rows


def build_basic_dna_metadata(
    reference_items: dict[str, ReferenceItem],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item_id in sorted(reference_items, key=lambda value: int(value)):
        item = reference_items[item_id]
        rows.append(
            {
                "item_id": item_id,
                "task_name": item.task_name,
                "query": item.query,
                "choices": json.dumps(item.choices),
                "gold_index": item.gold_index,
                "n_choices": len(item.choices),
            }
        )
    return pd.DataFrame(rows)


def write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def build_py_irt_rows(subject_rows: Sequence[SubjectRow]) -> list[dict[str, object]]:
    return [{"subject_id": row.subject_id, "responses": row.responses} for row in subject_rows]


def train_irt(
    *, jsonl_path: Path, epochs: int, device: str, log_every: int
) -> IrtBestParams:
    dataset = IrtDataset.from_jsonlines(jsonl_path)
    trainer = TwoParamIrtTrainer(dataset, priors="hierarchical")
    return trainer.train(epochs=epochs, device=device, log_every=log_every)


def build_item_params_dataframe(best_params: IrtBestParams) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": list(best_params["item_ids"].values()),
            "discrimination_a": best_params["disc"],
            "difficulty_b": best_params["diff"],
        }
    )


def build_subject_summary_dataframe(
    best_params: IrtBestParams, subject_rows: list[SubjectRow]
) -> pd.DataFrame:
    summary_by_subject = {row.subject_id: row for row in subject_rows}
    rows: list[dict[str, object]] = []

    for index, subject_id in enumerate(best_params["subject_ids"].values()):
        source_row = summary_by_subject[subject_id]
        row: dict[str, object] = {
            "subject_id": subject_id,
            "model_org": source_row.model_org,
            "model_name": source_row.model_name,
            "ability_theta": best_params["ability"][index],
            "observed_accuracy": source_row.observed_accuracy,
            "answered_items": source_row.row_count,
        }
        if source_row.timestamp is not None:
            row["timestamp"] = source_row.timestamp
        rows.append(row)

    return pd.DataFrame(rows)


def build_item_difficulty_dataframe(
    item_params_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    subject_rows: list[SubjectRow],
) -> pd.DataFrame:
    item_ids = list(item_params_df["item_id"])
    item_mean_accuracy = {
        item_id: float(np.mean([row.responses[item_id] for row in subject_rows]))
        for item_id in item_ids
    }
    stats_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "mean_accuracy": [item_mean_accuracy[item_id] for item_id in item_ids],
            "n_subjects": [len(subject_rows)] * len(item_ids),
        }
    )
    difficulty_df = item_params_df.merge(stats_df, on="item_id", how="inner").merge(
        metadata_df, on="item_id", how="inner"
    )
    difficulty_df = difficulty_df.sort_values(
        by=["difficulty_b", "item_id"], ascending=[False, True]
    ).reset_index(drop=True)
    difficulty_df.insert(1, "difficulty_rank", np.arange(1, len(difficulty_df) + 1))
    return difficulty_df


def save_outputs(
    *,
    output_dir: Path,
    manifest_rows: list[ManifestRow],
    item_params_df: pd.DataFrame,
    item_difficulty_df: pd.DataFrame,
    subject_summary_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.DataFrame(row.to_record() for row in manifest_rows)
    sort_columns = [
        column
        for column in ("status", "repo_id", "subject_id", "timestamp")
        if column in manifest_df.columns
    ]
    if sort_columns:
        manifest_df = manifest_df.sort_values(sort_columns)

    manifest_df.to_csv(output_dir / "dataset_manifest.csv", index=False)
    item_params_df.to_csv(output_dir / "irt_item_params.csv", index=False)
    item_difficulty_df.to_csv(output_dir / "irt_item_difficulty.csv", index=False)
    subject_summary_df.to_csv(output_dir / "subject_summary.csv", index=False)


def validate_seqqa_args(args: argparse.Namespace) -> None:
    ensure_positive(args.limit_datasets, "--limit-datasets")
    ensure_positive(args.max_workers, "--max-workers")


def apply_seqqa_defaults(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT
            / "scratch"
            / "seqqa_irt"
            / "lab_bench_seqqa_mcf_all_0"
            / "latest"
            / "non_abl"
        )
    if args.collection is None:
        args.collection = "hf-carbon/lighteval-outputs"
    if args.config is None:
        args.config = "lab_bench_seqqa_mcf_all_0"
    if args.split is None:
        args.split = "latest"
    if args.exclude_substring is None:
        args.exclude_substring = ["abl"]
    if args.max_workers is None:
        args.max_workers = 8


def discover_seqqa_profile(
    args: argparse.Namespace,
) -> tuple[list[object], list[ManifestRow]]:
    return discover_seqqa_datasets(
        collection_id=args.collection,
        config=args.config,
        split=args.split,
        exclude_substrings=args.exclude_substring,
        limit_datasets=args.limit_datasets,
        max_workers=args.max_workers,
    )


def load_seqqa_profile(
    args: argparse.Namespace,
    sources: list[object],
    manifest_rows: list[ManifestRow],
) -> tuple[list[SubjectRow], dict[str, ReferenceItem]]:
    repo_ids = [str(source) for source in sources]
    manifest_by_repo_id = {
        row.repo_id: row for row in manifest_rows if row.status == "included" and row.repo_id
    }

    def load_subject_dataset(repo_id: str) -> LoadedSubject:
        try:
            dataset = load_dataset(repo_id, args.config, split=args.split)
        except Exception as exc:
            raise RuntimeError(f"load_failed:{repo_id}:{type(exc).__name__}") from exc

        try:
            model_org, model_name = dataset_model_parts(repo_id)
        except ValueError as exc:
            raise RuntimeError(f"invalid_repo_name:{repo_id}") from exc
        return build_loaded_subject(
            manifest_key=repo_id,
            subject_id=repo_id,
            model_org=model_org,
            model_name=model_name,
            records=dataset,
            expected_row_count=SEQQA_EXPECTED_ROWS,
            extract_row=extract_doc_metric,
            build_reference_item=build_seqqa_reference_item,
        )

    loaded_subjects: list[LoadedSubject] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(load_subject_dataset, repo_id): repo_id for repo_id in repo_ids}
        for future in as_completed(futures):
            repo_id = futures[future]
            try:
                loaded_subjects.append(future.result())
            except RuntimeError as exc:
                manifest_by_repo_id[repo_id].status = "excluded"
                manifest_by_repo_id[repo_id].reason = str(exc)

    return finalize_loaded_subjects(
        loaded_subjects,
        manifest_by_key=manifest_by_repo_id,
        compare_fields=("query", "choices", "gold_index"),
    )


def validate_basic_dna_args(args: argparse.Namespace) -> None:
    ensure_positive(args.limit_models, "--limit-models")


def apply_basic_dna_defaults(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT
            / "scratch"
            / "basic_dna_irt"
            / "basic_dna_mcf_0"
            / "latest"
            / "per_model_latest"
        )
    if args.details_glob is None:
        args.details_glob = "*/*/*/details_basic_dna_mcf|0_*.parquet"


def discover_basic_dna_profile(
    args: argparse.Namespace,
) -> tuple[list[object], list[ManifestRow]]:
    return discover_basic_dna_datasets(
        details_glob=args.details_glob,
        limit_models=args.limit_models,
    )


def load_basic_dna_profile(
    _args: argparse.Namespace,
    sources: list[object],
    manifest_rows: list[ManifestRow],
) -> tuple[list[SubjectRow], dict[str, ReferenceItem]]:
    paths = [Path(source) for source in sources]
    manifest_by_key = {
        (row.subject_id, row.timestamp): row
        for row in manifest_rows
        if row.status == "included" and row.subject_id is not None and row.timestamp is not None
    }

    def load_subject_dataset(path: Path) -> LoadedSubject:
        subject_id = model_id(path)
        model_org, model_name = model_parts(path)
        try:
            dataframe = pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(f"load_failed:{subject_id}:{type(exc).__name__}") from exc
        timestamp = path.parts[-2]
        return build_loaded_subject(
            manifest_key=(subject_id, timestamp),
            subject_id=subject_id,
            model_org=model_org,
            model_name=model_name,
            records=dataframe.to_dict(orient="records"),
            expected_row_count=BASIC_DNA_EXPECTED_ROWS,
            extract_row=extract_doc_metric,
            build_reference_item=build_basic_dna_reference_item,
            timestamp=timestamp,
        )

    loaded_subjects: list[LoadedSubject] = []
    for path in paths:
        key = (model_id(path), path.parts[-2])
        try:
            loaded_subjects.append(load_subject_dataset(path))
        except RuntimeError as exc:
            manifest_by_key[key].status = "excluded"
            manifest_by_key[key].reason = str(exc)

    return finalize_loaded_subjects(
        loaded_subjects,
        manifest_by_key=manifest_by_key,
        compare_fields=("task_name", "query", "choices", "gold_index"),
    )


PROFILES = {
    "seqqa": ProfileSpec(
        apply_defaults=apply_seqqa_defaults,
        validate_args=validate_seqqa_args,
        discover=discover_seqqa_profile,
        load_subject_rows=load_seqqa_profile,
        build_metadata=join_seqqa_reference_metadata,
    ),
    "basic_dna": ProfileSpec(
        apply_defaults=apply_basic_dna_defaults,
        validate_args=validate_basic_dna_args,
        discover=discover_basic_dna_profile,
        load_subject_rows=load_basic_dna_profile,
        build_metadata=build_basic_dna_metadata,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a 2PL IRT model for a supported evaluation profile."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        required=True,
        help="IRT dataset profile to fit.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of optimization epochs for the 2PL fit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for NumPy, Torch, and Pyro.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use for fitting. Accepts values like 'cpu', 'cuda', or 'cuda:0'.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="How often to print training progress.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for manifests, JSONL input, and fitted outputs.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="SeqQA profile: HF collection id used to discover details datasets.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="SeqQA profile: HF dataset config name to load from each details repo.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="SeqQA profile: HF dataset split name to load from each details repo.",
    )
    parser.add_argument(
        "--exclude-substring",
        action="append",
        default=None,
        help="SeqQA profile: dataset repo ids containing this substring are excluded. Repeatable.",
    )
    parser.add_argument(
        "--limit-datasets",
        type=int,
        default=None,
        help="SeqQA profile: optional cap on the number of included datasets after filtering.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="SeqQA profile: maximum number of parallel workers for collection inspection and dataset loads.",
    )
    parser.add_argument(
        "--details-glob",
        default=None,
        help="Basic DNA profile: glob pattern under details/ used to find parquet detail files.",
    )
    parser.add_argument(
        "--limit-models",
        type=int,
        default=None,
        help="Basic DNA profile: optional cap on the number of included models after sorting.",
    )

    args = parser.parse_args()
    PROFILES[args.profile].apply_defaults(args)
    return args


def main() -> None:
    args = parse_args()
    validate_device(args.device)
    set_random_seed(args.seed)

    profile = PROFILES[args.profile]
    profile.validate_args(args)
    included_sources, manifest_rows = profile.discover(args)
    subject_rows, reference_items = profile.load_subject_rows(
        args, included_sources, manifest_rows
    )
    metadata_df = profile.build_metadata(reference_items)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "py_irt_input.jsonl"
    write_jsonl(input_path, build_py_irt_rows(subject_rows))

    best_params = train_irt(
        jsonl_path=input_path,
        epochs=args.epochs,
        device=args.device,
        log_every=args.log_every,
    )

    item_params_df = build_item_params_dataframe(best_params)
    subject_summary_df = build_subject_summary_dataframe(best_params, subject_rows)
    item_difficulty_df = build_item_difficulty_dataframe(
        item_params_df, metadata_df, subject_rows
    )
    save_outputs(
        output_dir=output_dir,
        manifest_rows=manifest_rows,
        item_params_df=item_params_df,
        item_difficulty_df=item_difficulty_df,
        subject_summary_df=subject_summary_df,
    )

    included_count = sum(1 for row in manifest_rows if row.status == "included")
    excluded_count = len(manifest_rows) - included_count
    print(f"Included datasets: {included_count}")
    print(f"Excluded datasets: {excluded_count}")
    print(f"Items: {len(reference_items)}")
    print(f"Wrote manifest to {output_dir / 'dataset_manifest.csv'}")
    print(f"Wrote raw responses to {input_path}")
    print(f"Wrote item parameters to {output_dir / 'irt_item_params.csv'}")
    print(f"Wrote item difficulty table to {output_dir / 'irt_item_difficulty.csv'}")
    print(f"Wrote subject summary to {output_dir / 'subject_summary.csv'}")


if __name__ == "__main__":
    main()
