# Carbon

A project to build an English and DNA LLM + a plasmid generation model as an application. Current training proposal: [here](https://docs.google.com/document/d/1YdHmR8-MdgGOBApA76KvgcZyKbnngmRnXhzwT_UntjM/edit?usp=sharing)

## Datasets

### DNA Datasets
- [OpenGenome2](https://huggingface.co/datasets/arcinstitute/opengenome2): used to train the [Evo2](https://huggingface.co/collections/arcinstitute/evo) family of 1B, 7B, and 40B biological foundation models. [Paper](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1), [GitHub](https://github.com/ArcInstitute/evo2).

### Natural Plasmids
Natural plasmids found in bacteria, not synthetically engineered.
- **PLSDB:** 66k bacterial plasmid sequences collected from GenBank and RefSeq isolate genomes. Includes metadata on host taxonomy, geographical location, antibiotic resistance genes, virulence factors, and MOB/replicon typing.
- **IMG/PR:** An environmental database containing 693k plasmid sequences automatically identified from isolate genomes, metagenomes, and metatranscriptomes. Data analysis notebook available at [`data/imgpr_data_analysis.ipynb`](data/imgpr_data_analysis.ipynb).
- **PIPdb:** A pathogen-focused database with 761k plasmid sequences from pathogenic bacteria, including annotations for virulence factors and antimicrobial resistance genes, as well as a risk-scoring system.
- **PlasmidScope**: Includes both PLSDB and IMG/PR plus additional plasmids from GenBank, RefSeq, COMPASS.. with standardized annotations and quality filtering + deduplication. Total of 852k plasmids. (The sequences are being added to the hub)

Datasets are available on the hub: [hf-carbon/natural-plasmids](https://huggingface.co/datasets/hf-carbon/natural-plasmids) (and [HuggingFaceTB/carbon-raw-data](https://huggingface.co/datasets/HuggingFaceTB/carbon-raw-data) for the viewer)

### Synthetic Plasmids from Labs
- 160k Plasmids from Addgene at [hf-carbon/addgene](https://huggingface.co/datasets/hf-carbon/AddGene)

## Model Training

- TRL training under [`trl_training`](trl_training/), use nanotron instead which is faster. 
- nanotron training under [`nanotron_training`](nanotron_training/) with code for tokenization using `datatrove`

For large scale pretraining we use [nanotron](https://github.com/huggingface/nanotron/tree/main) library.

## Evaluation

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
