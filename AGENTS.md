## Plans

* Store plans in the `scratch/plans` folder with an instructive name.

## Formatting

* When writing code snippets in Markdown, use ```sh code snippet ``` instead of ```bash code snippet ```

## Development / debugging

* Use the `scratch` folder at the root of the repo to store local test scripts or files.
* Never commit data files to Git

## Evaluation

* For evals, use the virtual env stored at evaluation/.venv

## Hugging Face Hub

* Check you are logged in with `hf auth whoami`. If you are not logged in, check you have activated a virtual environment

### Datasets

* To download datasets, use `datasets.load_dataset`
* To push datasets, use the `push_to_hub()` functionality of the `datasets` library
* When you create a new dataset and push it to the Hub, include a `create_dataset.py` script which shows how the dataset was created.
* When you create a new dataset and push it to the Hub, include an informative dataset card on what the sources are and how to use the dataset.
* In dataset cards, use ```py code snippet ``` for Python usage examples.
