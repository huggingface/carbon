# Evals

Scripts to eval all the things.

## Setup

First create the virtual env:

```sh
uv venv --python 3.12
```

Then install the remaining dependencies:

```sh
uv sync
```

Some evals require Evo2 support and can be installed as follows:

```sh
uv sync --extra evo2
```

## Pure DNA models

### Sequence Recovery 
We provide a standalone eval script that runs Sequence Recovery after training against a **Hub model + revision** and saves results to Parquet (plus a JSON summary). Sequence Recovery is a **training-free generative eval**: given a fixed-length DNA context, the model generates the next segment and we score **exact-base recovery accuracy** (overall + type-wise).
`--data_type` selects the dataset split: `eukaryote`, `bacteria`, or `others`.

CLI:
```
python evaluation/sequence_recovery_eval.py \
  --model /path/to/carbon/model-or-hub-repo \
  --revision checkpoint-10000 \
  --data_type eukaryote \
  --output_dir ./eval_results/sequence_recovery \
  --bf16
```
For official Evo2 weights, add `--use_evo2` (and optionally `--gen_len_bp 30`).

SLURM:
```
sbatch --export=MODEL=/path/to/carbon/model-or-hub-repo,REVISION=checkpoint-10000,DATA_TYPE=eukaryote evaluation/sequence_recovery_eval.slurm
```

Optional upload:
```
python evaluation/sequence_recovery_eval.py \
  --model /path/to/carbon/model-or-hub-repo \
  --revision checkpoint-10000 \
  --data_type eukaryote \
  --output_dir ./eval_results/sequence_recovery \
  --bf16 \
  --push_to_hub \
  --hub_repo_id hf-carbon/seq-recovery-results
```

### ClinVar VEP (post-training)
ClinVar VEP evaluates variant effect prediction by scoring **ref vs alt alleles** in long genomic context and reporting **AUROC/AUPRC**.

CLI:
```
python evaluation/clinvar_vep_eval.py \
  --model /path/to/carbon/model-or-hub-repo \
  --revision checkpoint-10000 \
  --context_length 96000 \
  --output_dir ./eval_results/clinvar_vep \
  --bf16
```
For official Evo2 weights, add `--use_evo2`.

SLURM:
```
sbatch --export=MODEL=/path/to/carbon/model-or-hub-repo,REVISION=checkpoint-10000,CONTEXT_LEN=96000 evaluation/clinvar_vep_eval.slurm
```

### CDS half-shuffle discrimination (post-training)
This task evaluates whether a model assigns higher likelihood to **real CDS sequences** vs **half-shuffled negatives** (first half fixed, second half shuffled). The dataset lives at `hf-carbon/carbon_tasks` and uses fixed column names (`original_sequence`, `input`).
For the current dataset, `original_sequence` is the real CDS and `input` is the half-shuffled control.

CLI:
```
python evaluation/cds_half_shuffle_eval.py \
  --model /path/to/carbon/model-or-hub-repo \
  --revision checkpoint-10000 \
  --dataset hf-carbon/carbon_tasks \
  --split train \
  --output_dir ./eval_results/cds_half_shuffle \
  --bf16
```
Dataset columns example:
```
record_id, taxonomy, gene_type, species_type, original_sequence, length, label, input, __index_level_0__
```
In this dataset, `original_sequence` is the real CDS and `input` is the half-shuffled control.
The dataset currently provides a single `train` split with ~30K rows (config: `default`).
For official Evo2 weights, add `--use_evo2` and pass the Evo2 model name (e.g., `evo2_1b_base`).

SLURM:
```
sbatch --export=MODEL=/path/to/carbon/model-or-hub-repo,REVISION=checkpoint-10000 evaluation/cds_half_shuffle_eval.slurm
```

### DART-Eval Task 1: Prioritizing Known Regulatory Elements (post-training)
Zero-shot likelihood evaluation from [DART-Eval](https://github.com/kundajelab/DART-Eval). Compares model log-likelihoods on real ENCODE cCRE elements vs dinucleotide-shuffled controls, reporting accuracy and Wilcoxon signed-rank test. Data is auto-downloaded from [hf-carbon/dart-eval-task1](https://huggingface.co/datasets/hf-carbon/dart-eval-task1) (private).

Extra dependencies: `pip install pyfaidx polars`

CLI:
```
python evaluation/dart_eval_task1.py \
  --model GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --dart_work_dir /path/to/dart_work \
  --batch_size 512 \
  --bf16
```

SLURM:
```
sbatch --export=MODEL=GenerTeam/GENERator-v2-eukaryote-1.2b-base evaluation/dart_task1_zero_shot.slurm
```

### KEGG DNA-only classifier (post-training)
This matches the BioReason **DNA-only Evo2** setup: we train a lightweight classifier head on `wanglab/kegg` **with the backbone frozen** and evaluate accuracy/F1 on **val and test splits**.

CLI:
```
python evaluation/kegg_dna_classifier_train.py \
  --model /path/to/carbon/model-or-hub-repo \
  --revision checkpoint-10000 \
  --max_epochs 5 \
  --batch_size 1 \
  --max_length 2048 \
  --truncate_dna_per_side 1024 \
  --merge_val_test_set \
  --bf16
```

SLURM:
```
sbatch --export=MODEL=/path/to/carbon/model-or-hub-repo,REVISION=checkpoint-10000 kegg_dna_classifier_train.slurm
```

Optional upload:
```
python evaluation/kegg_dna_classifier_train.py \
  --model /path/to/carbon/model-or-hub-repo \
  --revision checkpoint-10000 \
  --max_epochs 5 \
  --batch_size 1 \
  --max_length 2048 \
  --truncate_dna_per_side 1024 \
  --merge_val_test_set \
  --bf16 \
  --push_to_hub \
  --hub_repo_id hf-carbon/kegg-dna-classifier-results
```

## Base / pretrained models

For pretrained models, we use logit-based scoring on multiple-choice questions. In addition to tasks that are natively supported in LightEval, we support custom tasks for biology in the [`tasks.py`](evaluation/lighteval_tasks/tasks.py) module. These tasks can be run as follows from the root of the repo:

```sh
uv run lighteval vllm \
  "model_name=Qwen/Qwen3.5-0.8B-Base,dtype=bfloat16,override_chat_template=false,trust_remote_code=True,tensor_parallel_size=2,max_model_length=8192,max_num_batched_tokens=8192,generation_parameters={temperature:0},gpu_memory_utilization=0.85,max_num_seqs=16" \
  "mmlu_pro_biology_cf" \
  --custom-tasks evaluation/lighteval_tasks/tasks.py \
  --output-dir . \
  --save-details \
  --push-to-hub \
  --results-org hf-carbon
```

Or submit via Slurm:

```sh
sbatch evaluation/launch_lighteval.slurm \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision main \
  --tp 2 \
  --dp 1 \
  --task "mmlu_pro_biology_cf"
```

The results will be pushed to the [`hf-carbon`](https://huggingface.co/hf-carbon) org and later added to [this collection](https://huggingface.co/collections/hf-carbon/eval-outputs) by a daily cron job. To run a curated list of tasks in series, use:

```sh
uv run lighteval vllm \
  "model_name=Qwen/Qwen3.5-0.8B-Base,dtype=bfloat16,override_chat_template=false,trust_remote_code=True,tensor_parallel_size=2,max_model_length=8192,max_num_batched_tokens=8192,generation_parameters={temperature:0},gpu_memory_utilization=0.85,max_num_seqs=16" \
  "lighteval_tasks/{en,bio}.txt" \
  --custom-tasks evaluation/lighteval_tasks/tasks.py \
  --output-dir . \
  --save-details \
  --push-to-hub \
  --results-org hf-carbon
```

### Supported tasks
The table below covers the curated biology task list in [`lighteval_tasks/bio.txt`](evaluation/lighteval_tasks/bio.txt). Dataset links point to the eval dataset split defined in [`lighteval_tasks/tasks.py`](evaluation/lighteval_tasks/tasks.py), while `Samples` and prompt-token stats are computed from the `latest` split in `hf-carbon/details_Qwen__Qwen3-4B-Base_private` using the `Qwen/Qwen3-4B-Base` tokenizer.

| Task | Description | Dataset | Samples | Choices | Prompt tokens (min/mean/max) |
| --- | --- | --- | ---: | ---: | ---: |
| `basic_dna_mcf` | Short multiple-choice questions on foundational DNA sequence and molecular biology concepts. | [dataset](https://huggingface.co/datasets/hf-carbon/basic-dna/viewer/default/train) | 200 | 4 | 32/80.8/203 |
| `gpqa_biology_mcq_mcf` | Graduate-level biology questions from GPQA designed to be difficult even for domain experts. | [dataset](https://huggingface.co/datasets/hf-carbon/gpqa-biology-mcq/viewer/gpqa_main/train) | 78 | 4 | 75/269.6/2770 |
| `hle_gold_bio_mcf` | Expert-validated biology questions from the gold subset of Humanity's Last Exam. | [dataset](https://huggingface.co/datasets/hf-carbon/hle-gold-bio/viewer/multiple_choice_formatted/train) | 77 | 6 | 37/341.3/2173 |
| `lab_bench_cloningscenarios_mcf` | LAB-Bench cloning problems that require reasoning over plasmids, enzymes, and construct design. | [dataset](https://huggingface.co/datasets/hf-carbon/lab-bench/viewer/CloningScenarios/train) | 33 | 4 | 756/5926.8/26334 |
| `lab_bench_seqqa_mcf` | LAB-Bench sequence-analysis questions covering ORFs, primers, restriction digests, and related tasks. | [dataset](https://huggingface.co/datasets/hf-carbon/lab-bench/viewer/SeqQA/train) | 600 | 4 | 72/717.8/4410 |
| `mmlu_mcf:college_biology` | College-level biology knowledge questions from MMLU. | [dataset](https://huggingface.co/datasets/cais/mmlu/viewer/college_biology/test) | 144 | 4 | 38/78.5/203 |
| `mmlu_mcf:high_school_biology` | High-school biology knowledge questions from MMLU. | [dataset](https://huggingface.co/datasets/cais/mmlu/viewer/high_school_biology/test) | 310 | 4 | 32/79.4/185 |
| `mmlu_mcf:medical_genetics` | Medical genetics multiple-choice questions from MMLU. | [dataset](https://huggingface.co/datasets/cais/mmlu/viewer/medical_genetics/test) | 100 | 4 | 33/56.6/99 |
| `mmlu_pro_biology_mcf` | Harder biology questions from MMLU-Pro with more answer choices and stronger reasoning demands. | [dataset](https://huggingface.co/datasets/hf-carbon/mmlu-pro-biology/viewer/default/test) | 717 | 10 | 40/200.3/1096 |
| `mmlu_redux_mcf:college_biology` | Cleaned and ambiguity-reduced college biology questions from MMLU-Redux. | [dataset](https://huggingface.co/datasets/hf-carbon/mmlu-redux-2.0-biology/viewer/college_biology/train) | 98 | 4 | 40/79.7/203 |
| `mmlu_redux_mcf:high_school_biology` | Cleaned and ambiguity-reduced high-school biology questions from MMLU-Redux. | [dataset](https://huggingface.co/datasets/hf-carbon/mmlu-redux-2.0-biology/viewer/high_school_biology/train) | 95 | 4 | 36/82.0/183 |
| `mmlu_redux_mcf:medical_genetics` | Cleaned and ambiguity-reduced medical genetics questions from MMLU-Redux. | [dataset](https://huggingface.co/datasets/hf-carbon/mmlu-redux-2.0-biology/viewer/medical_genetics/train) | 100 | 4 | 33/56.6/99 |
| `scieval_mcq_genetics_mcf` | Genetics-focused multiple-choice questions from the SciEval scientific benchmark. | [dataset](https://huggingface.co/datasets/hf-carbon/scieval-biology/viewer/mcq_genetics/test_mcq) | 593 | 4 | 34/127.3/937 |
| `scieval_mcq_mcf` | Biology and biomedical multiple-choice questions from SciEval spanning scientific understanding and problem solving. | [dataset](https://huggingface.co/datasets/hf-carbon/scieval-biology/viewer/mcq/validation_mcq) | 400 | 4 | 35/123.7/465 |
| `sciknoweval_mcq_mcf` | Biology questions from SciKnowEval aimed at measuring scientific knowledge and comprehension. | [dataset](https://huggingface.co/datasets/hf-carbon/sciknoweval-biology/viewer/mcq-4-choices-formatted/test) | 3935 | 4 | 35/122.8/3267 |
| `wmdp_bio_mcf` | Biosecurity-focused questions from WMDP about potentially hazardous biological knowledge. | [dataset](https://huggingface.co/datasets/hf-carbon/wmdp/viewer/wmdp-bio/test) | 1273 | 4 | 33/91.5/757 |

## Post-trained models

For post-trained models, we use the excellent [Inspect](https://inspect.aisi.org.uk) framework, which includes LAB-Bench [v1](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/lab_bench) and [v2](https://github.com/EdisonScientific/labbench2) as an open [issue](https://github.com/UKGovernmentBEIS/inspect_evals/issues/1204).


### LAB-Bench

See the [Inspect instructions](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/lab_bench#running-evaluations) on how to run the eval. Here's a quick example for the SeqQA subtask:

```sh
uv run inspect eval inspect_evals/lab_bench_seqqa \
  --model vllm/Qwen/Qwen3.5-4B \
  -M tensor_parallel_size=2 \
  -M gpu_memory_utilization=0.85 \
  -M max_model_len=32768 \
  --max-connections 64 \
  --max-samples 64 \
  --log-dir ./results/inspect-logs/ \
  --display plain
```

You can then sync the results to the Hub as follows:

```sh
uv run inspect view bundle \
  --log-dir ./results/inspect-logs \
  --output-dir hf/hf-carbon/inspect-evals
```