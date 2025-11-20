# Carbon

A project to build a plasmid generation model. Current training proposal: [here](https://docs.google.com/document/d/1Q-OXMESw0p61RyS1J8Na6GSR8gys4BKJFYoJClp9Iv4/edit?usp=sharing)

## Datasets

### DNA Datasets
- [OpenGenome2](https://huggingface.co/datasets/arcinstitute/opengenome2): 9T tokens used to train the [Evo2](https://huggingface.co/collections/arcinstitute/evo) family of 1B, 7B, and 40B biological foundation models. Uses the StripedHyena2 architecture with 131k context length (a hybrid transformer, not integrated into transformers). [Paper](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1), [GitHub](https://github.com/ArcInstitute/evo2).
- [OpenGenome](https://huggingface.co/datasets/LongSafari/open-genome): 300B tokens used to train [Evo1](https://huggingface.co/togethercomputer/evo-1-131k-base), a 7B biological foundation model trained on DNA. [Paper](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v1.full).

### Natural Plasmids
- **PLSDB:** 66k bacterial plasmid sequences collected from GenBank and RefSeq isolate genomes. Includes metadata on host taxonomy, geographical location, antibiotic resistance genes, virulence factors, and MOB/replicon typing.
- **IMG/PR:** An environmental database containing 693k plasmid sequences automatically identified from isolate genomes, metagenomes, and metatranscriptomes. Data analysis notebook available at [`data/imgpr_data_analysis.ipynb`](data/imgpr_data_analysis.ipynb).
- **PIPdb:** A pathogen-focused database with 761k plasmid sequences from pathogenic bacteria, including annotations for virulence factors and antimicrobial resistance genes, as well as a risk-scoring system.

Datasets are available on the hub: [https://huggingface.co/datasets/HuggingFaceTB/carbon-raw-data](https://huggingface.co/datasets/HuggingFaceTB/carbon-raw-data)

### Synthetic Plasmids from Labs
- The request to [Addgene](https://www.addgene.org/browse/) (160k plasmids) is still being processed.

## Model Training

- TRL training under [`trl_training`](trl_training/), but the training loss has spikes, use nanotron instead. 
- nanotron training under [`nanotron_training`](nanotron_training/)

## Evaluation
This is a WIP. Script to compute token accuracy under [`evaluation/`](evaluation/).
