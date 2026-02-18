---
pretty_name: MMLU-Pro Biology
language:
- en
license: other
size_categories:
- 1K<n<10K
source_datasets:
- TIGER-Lab/MMLU-Pro
---

# MMLU-Pro Biology

This dataset is a filtered subset of `TIGER-Lab/MMLU-Pro` containing only rows where `category == "biology"`.

## Source

- Original dataset: `TIGER-Lab/MMLU-Pro`
- Filter applied: `category == "biology"`

Please refer to the original dataset card for full license and citation details. This subset is provided for convenience and inherits the original dataset's terms.

## Data fields

Each split contains the following columns (inherited from the original dataset):

- `question_id`
- `question`
- `options`
- `answer`
- `answer_index`
- `cot_content`
- `category`
- `src`

## Splits

The dataset includes the same split names as the source dataset, filtered by category:

- `test`
- `validation`

Note: The expected random-guess accuracy on the `test` split (accounting for varying numbers of options per question) is about 11.08% (`0.110794`).

## Usage

```py
from datasets import load_dataset

ds = load_dataset("hf-carbon/mmlu-pro-biology")
print(ds)
print(ds["test"][0])
```

## Creation script

See `create_dataset.py` for the exact filtering and push logic.
