"""Push IRT output files to a Hub dataset, one subset per file.

Usage:
    uv run --directory evaluation python scripts/push_irt_outputs.py --profile seqqa
    uv run --directory evaluation python scripts/push_irt_outputs.py --profile basic_dna
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET_FILES = (
    "dataset_manifest.csv",
    "py_irt_input.jsonl",
    "irt_item_params.csv",
    "irt_item_difficulty.csv",
    "subject_summary.csv",
)

PROFILES = {
    "seqqa": {
        "default_output_dir": REPO_ROOT / "scratch" / "seqqa_irt" / "lab_bench_seqqa_mcf_all_0" / "latest" / "non_abl",
        "default_hub_repo_id": "hf-carbon/seqqa-irt-difficulty",
        "fit_script": "scripts/fit_irt.py",
        "fit_args": ["--profile", "seqqa", "--config", "lab_bench_seqqa_mcf_all_0", "--split", "latest"],
        "pretty_name": "SeqQA IRT Difficulty",
        "tags": ["biology", "lab-bench", "irt", "evaluation"],
        "summary_item_label": "SeqQA items",
        "summary_subject_label": "included model datasets",
        "fallback_summary": "SeqQA IRT outputs",
        "description": "Upload SeqQA IRT outputs to a Hugging Face dataset repo, using one dataset config per output file.",
        "output_help": "Directory containing the output files from fit_irt.py for the SeqQA profile.",
        "intro": "This dataset stores outputs produced by `evaluation/scripts/fit_irt.py --profile seqqa` in the Carbon repo.",
        "source_lines": [
            "- Collection: `hf-carbon/lighteval-outputs`",
            "- Task config: `lab_bench_seqqa_mcf_all_0`",
            "- Split: `latest`",
            "- Reference dataset: `hf-carbon/lab-bench`, subset `SeqQA`, split `train`",
        ],
        "subset_descriptions": {
            "dataset_manifest": "Discovery manifest for collection members considered during filtering.",
            "py_irt_input": "Binary subject-by-item response data used as the direct input to the 2PL fit.",
            "irt_item_params": "Raw fitted 2PL item parameters.",
            "irt_item_difficulty": "Raw item parameters joined to SeqQA metadata and empirical accuracy summaries.",
            "subject_summary": "Per-dataset subject metadata, observed accuracy, and fitted ability.",
        },
        "column_descriptions": {
            "dataset_manifest": [
                ("repo_id", "Hugging Face dataset repo id found in the collection."),
                ("item_type", "Collection item type returned by the Hub API."),
                ("status", "Whether the repo was included in or excluded from the fit."),
                ("reason", "Empty for included repos; otherwise the exclusion reason."),
                ("model_org", "Parsed model organization from the details dataset repo name when available."),
                ("model_name", "Parsed model name from the details dataset repo name when available."),
                ("row_count", "Number of SeqQA rows loaded from the repo for the requested config and split."),
                ("observed_accuracy", "Mean `metric.acc` over the included SeqQA items for that repo."),
            ],
            "py_irt_input": [
                ("subject_id", "Dataset repo id used as the IRT subject identifier."),
                ("responses", "Dictionary keyed by `item_id`; each value is the binary `metric.acc` response for that subject."),
            ],
            "irt_item_params": [
                ("item_id", "SeqQA problem identifier from the details datasets."),
                ("discrimination_a", "Fitted 2PL discrimination parameter; larger values separate stronger and weaker models more sharply."),
                ("difficulty_b", "Fitted 2PL difficulty parameter; larger values indicate harder items."),
            ],
            "irt_item_difficulty": [
                ("item_id", "SeqQA problem identifier from the details datasets."),
                ("difficulty_rank", "Rank after sorting by `difficulty_b` descending; `1` is the hardest item."),
                ("discrimination_a", "Fitted 2PL discrimination parameter."),
                ("difficulty_b", "Fitted 2PL difficulty parameter; larger values indicate harder problems."),
                ("mean_accuracy", "Empirical mean `metric.acc` across all included subjects for this item."),
                ("n_subjects", "Number of included subjects contributing responses to this item."),
                ("seqqa_row_index", "Integer row index in `hf-carbon/lab-bench`, subset `SeqQA`, split `train`."),
                ("seqqa_uuid", "Original UUID-style item id from the reference SeqQA dataset."),
                ("subtask", "SeqQA subtask label."),
                ("question", "Original SeqQA question text."),
                ("options", "JSON-encoded list of the original answer options from the reference SeqQA dataset."),
                ("answer_index", "Zero-based index of the correct answer in `options`."),
                ("n_choices", "Number of answer choices for the problem."),
            ],
            "subject_summary": [
                ("subject_id", "Dataset repo id used as the IRT subject identifier."),
                ("model_org", "Parsed model organization from the details dataset repo name when available."),
                ("model_name", "Parsed model name from the details dataset repo name when available."),
                ("ability_theta", "Fitted 2PL subject ability parameter."),
                ("observed_accuracy", "Mean empirical `metric.acc` for this subject across all SeqQA items."),
                ("answered_items", "Number of items answered by this subject in the fitted matrix."),
            ],
        },
    },
    "basic_dna": {
        "default_output_dir": REPO_ROOT / "scratch" / "basic_dna_irt" / "basic_dna_mcf_0" / "latest" / "per_model_latest",
        "default_hub_repo_id": "hf-carbon/basic-dna-irt-difficulty",
        "fit_script": "scripts/fit_irt.py",
        "fit_args": ["--profile", "basic_dna"],
        "pretty_name": "Basic DNA IRT Difficulty",
        "tags": ["biology", "dna", "irt", "evaluation"],
        "summary_item_label": "basic_dna_mcf|0 items",
        "summary_subject_label": "included model runs",
        "fallback_summary": "basic_dna_mcf|0 IRT outputs",
        "description": "Upload basic_dna_mcf|0 IRT outputs to a Hugging Face dataset repo, using one dataset config per output file.",
        "output_help": "Directory containing the output files from fit_irt.py for the basic DNA profile.",
        "intro": "This dataset stores outputs produced by `evaluation/scripts/fit_irt.py --profile basic_dna` in the Carbon repo.",
        "source_lines": [
            "- Task: `basic_dna_mcf|0`",
            "- Input source: local `details/*/*/*/details_basic_dna_mcf|0_*.parquet`",
            "- Inclusion rule: latest timestamped run per model",
        ],
        "subset_descriptions": {
            "dataset_manifest": "Manifest of local detail parquet files considered during fitting.",
            "py_irt_input": "Binary subject-by-item response data used as the direct input to the 2PL fit.",
            "irt_item_params": "Raw fitted 2PL item parameters.",
            "irt_item_difficulty": "Raw item parameters joined to basic_dna_mcf|0 item text and empirical accuracy summaries.",
            "subject_summary": "Per-model subject metadata, observed accuracy, and fitted ability.",
        },
        "column_descriptions": {
            "dataset_manifest": [
                ("subject_id", "Model identifier in the form `{org}/{model_name}`."),
                ("model_org", "Model organization from the local details path."),
                ("model_name", "Model name from the local details path."),
                ("timestamp", "Timestamp directory of the evaluated local run."),
                ("status", "Whether the run was included in or excluded from the fit."),
                ("reason", "Empty for included runs; otherwise the exclusion reason."),
                ("file_path", "Repo-relative path to the source parquet detail file."),
                ("row_count", "Number of basic_dna_mcf|0 rows loaded from the detail file."),
                ("observed_accuracy", "Mean `metric.acc` over the included items for that subject."),
            ],
            "py_irt_input": [
                ("subject_id", "Model identifier used as the IRT subject identifier."),
                ("responses", "Dictionary keyed by `item_id`; each value is the binary `metric.acc` response for that subject."),
            ],
            "irt_item_params": [
                ("item_id", "basic_dna_mcf|0 problem identifier from the detail parquet files."),
                ("discrimination_a", "Fitted 2PL discrimination parameter; larger values separate stronger and weaker models more sharply."),
                ("difficulty_b", "Fitted 2PL difficulty parameter; larger values indicate harder items."),
            ],
            "irt_item_difficulty": [
                ("item_id", "basic_dna_mcf|0 problem identifier from the detail parquet files."),
                ("difficulty_rank", "Rank after sorting by `difficulty_b` descending; `1` is the hardest item."),
                ("discrimination_a", "Fitted 2PL discrimination parameter."),
                ("difficulty_b", "Fitted 2PL difficulty parameter; larger values indicate harder items."),
                ("mean_accuracy", "Empirical mean `metric.acc` across all included subjects for this item."),
                ("n_subjects", "Number of included subjects contributing responses to this item."),
                ("task_name", "Task name recorded in the source detail files."),
                ("query", "Rendered multiple-choice prompt from the source detail files."),
                ("choices", "JSON-encoded list of answer choice labels from the source detail files."),
                ("gold_index", "Zero-based index of the correct answer in `choices`."),
                ("n_choices", "Number of answer choices for the problem."),
            ],
            "subject_summary": [
                ("subject_id", "Model identifier used as the IRT subject identifier."),
                ("model_org", "Model organization from the local details path."),
                ("model_name", "Model name from the local details path."),
                ("timestamp", "Timestamp directory of the evaluated local run."),
                ("ability_theta", "Fitted 2PL subject ability parameter."),
                ("observed_accuracy", "Mean empirical `metric.acc` for this subject across all items."),
                ("answered_items", "Number of items answered by this subject in the fitted matrix."),
            ],
        },
    },
}


def parse_args(default_profile: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload IRT outputs to a Hugging Face dataset repo, using one dataset config per output file."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=default_profile,
        help="IRT dataset profile to publish.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory containing the fitted output files.",
    )
    parser.add_argument(
        "--hub-repo-id",
        default=None,
        help="Dataset repo id to push to.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not exist.",
    )
    args = parser.parse_args()
    if args.profile is None:
        parser.error("--profile is required unless using a profile-specific wrapper script.")
    return args


def load_output_dataset(path: Path) -> Dataset:
    if path.suffix == ".csv":
        dataframe = pd.read_csv(path)
        return Dataset.from_pandas(dataframe, preserve_index=False)
    if path.suffix == ".jsonl":
        return load_dataset("json", data_files=str(path), split="train")
    raise ValueError(f"Unsupported output file type: {path}")


def render_create_dataset_py(profile_name: str, profile: dict[str, object], hub_repo_id: str, output_dir: Path) -> str:
    fit_args = "\n".join(f'            "{value}",' for value in profile["fit_args"])
    fit_args_block = f"\n{fit_args}" if fit_args else ""
    return f'''"""Show how the `{hub_repo_id}` dataset was created."""

from pathlib import Path
import subprocess

OUTPUT_DIR = Path("{output_dir}")


def main() -> None:
    subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            "evaluation",
            "python",
            "{profile["fit_script"]}",{fit_args_block}
            "--output-dir",
            str(OUTPUT_DIR),
        ],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            "evaluation",
            "python",
            "scripts/push_irt_outputs.py",
            "--profile",
            "{profile_name}",
            "--output-dir",
            str(OUTPUT_DIR),
            "--hub-repo-id",
            "{hub_repo_id}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
'''


def render_readme(profile: dict[str, object], hub_repo_id: str, output_dir: Path, subset_names: list[str]) -> str:
    subject_summary_path = output_dir / "subject_summary.csv"
    difficulty_path = output_dir / "irt_item_difficulty.csv"
    subject_count = None
    item_count = None
    if subject_summary_path.exists():
        subject_count = len(pd.read_csv(subject_summary_path))
    if difficulty_path.exists():
        item_count = len(pd.read_csv(difficulty_path))

    subset_descriptions = profile["subset_descriptions"]
    column_descriptions = profile["column_descriptions"]

    subset_lines = "\n".join(
        f"- `{name}`: {subset_descriptions.get(name, 'Uploaded output subset.')}" for name in subset_names
    )
    summary_bits = []
    if item_count is not None:
        summary_bits.append(f"{item_count} {profile['summary_item_label']}")
    if subject_count is not None:
        summary_bits.append(f"{subject_count} {profile['summary_subject_label']}")
    summary = ", ".join(summary_bits) if summary_bits else profile["fallback_summary"]

    subset_sections = []
    for subset_name in subset_names:
        columns = column_descriptions.get(subset_name, [])
        column_lines = "\n".join(f"- `{column}`: {description}" for column, description in columns)
        subset_sections.append(
            f"### `{subset_name}`\n\n"
            f"{subset_descriptions.get(subset_name, 'Uploaded output subset.')}\n\n"
            f"Columns:\n\n{column_lines}"
        )
    subset_details = "\n\n".join(subset_sections)

    configs_yaml = "\n".join(
        (
            f"  - config_name: {name}\n"
            f"    data_files:\n"
            f"      - split: train\n"
            f"        path: {name}/train-*.parquet"
        )
        for name in subset_names
    )
    tags_yaml = "\n".join(f"- {tag}" for tag in profile["tags"])
    source_block = "\n".join(profile["source_lines"])

    return f"""---
pretty_name: {profile["pretty_name"]}
task_categories:
- text-classification
language:
- en
tags:
{tags_yaml}
configs:
{configs_yaml}
---

# {hub_repo_id}

{profile["intro"]}
The current pushed run contains {summary}.

## Subsets

Each output file is published as its own dataset config:

{subset_lines}

All subsets use a single `train` split.

## Subset details

{subset_details}

## Source

{source_block}

## Usage

```py
from datasets import load_dataset

difficulty = load_dataset("{hub_repo_id}", "irt_item_difficulty", split="train")
subjects = load_dataset("{hub_repo_id}", "subject_summary", split="train")
```

## Reproduction

See `create_dataset.py` in this dataset repo for the exact local commands used to regenerate and upload these outputs.
"""


def main(default_profile: str | None = None) -> None:
    args = parse_args(default_profile=default_profile)
    profile = PROFILES[args.profile]

    output_dir = args.output_dir or profile["default_output_dir"]
    hub_repo_id = args.hub_repo_id or profile["default_hub_repo_id"]
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    files = [output_dir / file_name for file_name in SUBSET_FILES]
    missing = [path.name for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected output files in {output_dir}: {missing}")

    api = HfApi()
    api.create_repo(
        repo_id=hub_repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )

    subset_names = []
    for path in files:
        subset_name = path.stem
        subset_names.append(subset_name)
        dataset = load_output_dataset(path)
        dataset.push_to_hub(
            hub_repo_id,
            config_name=subset_name,
            split="train",
            set_default=(subset_name == "irt_item_difficulty"),
            commit_message=f"Upload {subset_name} subset",
        )
        print(f"Pushed {path.name} to {hub_repo_id}::{subset_name}")

    readme_bytes = render_readme(profile, hub_repo_id, output_dir, subset_names).encode("utf-8")
    create_dataset_bytes = render_create_dataset_py(args.profile, profile, hub_repo_id, output_dir).encode("utf-8")
    api.upload_file(
        path_or_fileobj=readme_bytes,
        path_in_repo="README.md",
        repo_id=hub_repo_id,
        repo_type="dataset",
        commit_message="Upload dataset card",
    )
    api.upload_file(
        path_or_fileobj=create_dataset_bytes,
        path_in_repo="create_dataset.py",
        repo_id=hub_repo_id,
        repo_type="dataset",
        commit_message="Upload create_dataset.py",
    )
    print(f"Uploaded dataset card and create_dataset.py to {hub_repo_id}")


if __name__ == "__main__":
    main()
