# DNA Annotation Pipeline

Gene-centric annotation pipeline that produces structured training data for DNA language models. For each gene it fetches the raw DNA sequence with randomized flanking, annotates it across 10 biological categories using 11 databases, then generates natural language descriptions and reasoning traces via an LLM.

![Pipeline Overview](https://huggingface.co/datasets/lvwerra/admin/resolve/main/pipeline_overview.png)

## Quick Start

### Setup

```bash
pip install biopython requests pandas
brew install samtools  # or: conda install -c bioconda samtools
```

### Download reference genome + gene annotations

```bash
# Human reference genome (~850 MB compressed, ~3.1 GB uncompressed)
wget https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
samtools faidx Homo_sapiens.GRCh38.dna.primary_assembly.fa

# Gene annotations
wget https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz

export REFERENCE_FASTA=Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

### Fetch gene list

```bash
python fetch_genes.py --gtf Homo_sapiens.GRCh38.113.gtf.gz
# → genes_human.json (62,041 genes: protein-coding, lncRNA, miRNA, tRNA, snoRNA, ...)
```

### Annotate a gene

```bash
# By gene symbol
python annotate_all.py --gene BRCA1 --seed 42

# Or by index from gene list
python annotate_all.py --genes-file genes_human.json --index 0

# Or by raw coordinates
python annotate_all.py 17 43094000 43094500
```

### Summarize + describe

```bash
python summarize.py BRCA1_output.json

export GEMINI_API_KEY=your_key
python describe.py BRCA1_output_concise.json
```

### Visualize

```bash
python report_visualize.py BRCA1_output_concise_described.json
# → BRCA1_output_concise_described.html
```

### Generate reports

```bash
python report_genes.py genes_human.json     # gene size statistics + histograms
python report_cost.py                        # multi-organism cost estimates
python report_pipeline.py                    # pipeline infographic
```
