#!/usr/bin/env python3
"""
Generate an HTML infographic showing the full annotation pipeline.

Usage:
    python pipeline_overview.py
"""


def generate_html():
    # Pipeline stages (compact: title, scripts, details)
    stages = [
        ("Input", "fetch_genes.py, common.py", [
            "62,041 human genes from Ensembl GTF (protein-coding, lncRNA, miRNA, tRNA, snoRNA, ...)",
            "Randomized flanking: Beta(6,2) × 4kb upstream of TSS, Beta(4,2) × 3kb downstream",
            "DNA fetched locally from indexed GRCh38 FASTA via samtools (no API needed)",
        ]),
        ("Annotate", "annotate_all.py → 01–10_*.py", [
            "10 scripts run sequentially, each querying databases and mapping to local coordinates",
            "Uniform output: Annotation(start, end, type, category, label, score, metadata)",
        ]),
        ("Summarize", "summarize.py", [
            "Drop regex noise (raw codons, GC windows, predicted hairpins)",
            "Deduplicate by rs_id + position across categories",
            "Collapse >5 per type → top 5 + aggregate stats (typical: 2,141 → 52, 98% reduction)",
        ]),
        ("Describe", "describe.py → Gemini 3.1 Pro", [
            "Description — minimal external facts not derivable from the sequence alone",
            "Reasoning — chain-of-thought as if a model discovers content from raw DNA",
        ]),
        ("Output", "visualize.py, gene_report.py, cost_report.py", [
            "Per-gene: {gene}_output.json → _concise.json → _described.json → .html",
            "Training format: DNA sequence → (description, reasoning trace)",
        ]),
    ]

    # Annotation scripts with per-database detail
    ann_scripts = [
        {
            "num": "01",
            "name": "Gene Structure",
            "file": "01_gene_structure.py",
            "sources": [
                ("Ensembl REST", "/overlap/region", "Genes, transcripts (38 isoforms for BRCA1), exons, CDS with protein IDs"),
                ("Local regex", "sequence scan", "Start/stop codons (ATG, TAA/TAG/TGA), GT-AG splice donor/acceptor consensus"),
            ],
        },
        {
            "num": "02",
            "name": "Regulatory Elements",
            "file": "02_regulatory_elements.py",
            "sources": [
                ("JASPAR REST", "/matrix/ + PFM scan", "TF binding site motifs (CTCF, SP1, CREB1) scored against sequence"),
                ("Biopython", "Restriction module", "Restriction enzyme sites (EcoRI, BamHI, HindIII, ... — 10 enzymes)"),
                ("Local regex", "motif patterns", "TATA box, CAAT box, GC box, Initiator elements"),
                ("Local algorithm", "sliding window", "CpG islands (GC ≥ 50%, obs/exp CpG ≥ 0.6)"),
            ],
        },
        {
            "num": "03",
            "name": "Functional Classification",
            "file": "03_functional_classification.py",
            "sources": [
                ("Ensembl REST", "/overlap/region (gene, exon, cds)", "Per-position classification: CDS > exon > UTR > intron > intergenic"),
                ("Ensembl REST", "/overlap/region (regulatory)", "Regulatory features overlaid on the classification"),
            ],
        },
        {
            "num": "04",
            "name": "Chromatin States",
            "file": "04_chromatin_states.py",
            "sources": [
                ("Ensembl REST", "/overlap/region (regulatory)", "Promoters, enhancers, CTCF sites, open chromatin from Regulatory Build"),
                ("ENCODE REST", "/search/ (Histone ChIP-seq)", "Available experiments per histone mark (H3K4me3, H3K27ac, H3K4me1, H3K36me3, H3K27me3, H3K9me3)"),
                ("Rule-based", "mark → state mapping", "Inferred states: Active_Promoter, Enhancer, CTCF_Insulator, etc."),
            ],
        },
        {
            "num": "05",
            "name": "Conservation",
            "file": "05_conservation.py",
            "sources": [
                ("Ensembl REST", "/overlap/region (constrained)", "GERP constrained elements with scores (e.g., score=37.5)"),
                ("UCSC REST", "phastConsElements100way track", "PhastCons conserved elements from 100-way vertebrate alignment"),
                ("Local", "sliding window", "GC content per 50bp window with CpG dinucleotide counts"),
            ],
        },
        {
            "num": "06",
            "name": "Variants",
            "file": "06_variants.py",
            "sources": [
                ("Ensembl REST", "/overlap/region (variation)", "All known variants with rs_id, alleles, consequence type, clinical significance"),
                ("Ensembl VEP", "/vep/human/region/", "Predicted functional effect: consequence terms, impact, SIFT, PolyPhen"),
            ],
        },
        {
            "num": "07",
            "name": "Expression / Epigenetic",
            "file": "07_expression_epigenetic.py",
            "sources": [
                ("Ensembl REST", "/overlap/region (regulatory)", "Regulatory features: open chromatin, promoters, enhancers"),
                ("ENCODE REST", "/search/ (experiments)", "DNase-seq, H3K27ac ChIP-seq, H3K4me3 ChIP-seq availability in K562"),
                ("GTEx REST", "/expression/medianGeneExpression", "Tissue-specific expression: median TPM across 54 tissues, top tissues"),
            ],
        },
        {
            "num": "08",
            "name": "Repetitive Elements",
            "file": "08_repetitive_elements.py",
            "sources": [
                ("UCSC REST", "rmsk track", "RepeatMasker: repeat name, class (SINE/Alu, LINE), family, divergence %"),
                ("Ensembl REST", "/overlap/region (repeat)", "Repeat features with descriptions"),
                ("Local algorithm", "tandem repeat finder", "Simple repeats: (unit)n for unit sizes 1–6bp, ≥3 copies"),
            ],
        },
        {
            "num": "09",
            "name": "ncRNA",
            "file": "09_mirna_regulatory_rna.py",
            "sources": [
                ("Ensembl REST", "/overlap/region (gene, transcript)", "ncRNA genes: miRNA, lncRNA, snoRNA, snRNA, tRNA, rRNA, scaRNA"),
                ("Local", "miRNA seed matching", "Reverse-complement seed match for 8 common miRNAs (let-7a, miR-21, miR-155, ...)"),
                ("Local algorithm", "stem-loop prediction", "Hairpin structures: stem ≥ 5bp, loop 3–8nt, ≥80% base pairing"),
            ],
        },
        {
            "num": "10",
            "name": "Disease / Clinical",
            "file": "10_disease_clinical.py",
            "sources": [
                ("Ensembl REST", "/overlap/region (variation)", "Variants filtered to those with clinical_significance (pathogenic, benign, VUS, ...)"),
                ("Ensembl REST", "/phenotype/gene/", "Phenotype associations per gene (Cancer Gene Census, MIM, Orphanet)"),
                ("OpenTargets", "GraphQL API", "Disease–gene associations with scores, top 5 diseases per gene"),
                ("GWAS Catalog", "REST /singleNucleotidePolymorphisms", "GWAS hits: rs_id, position, functional class"),
            ],
        },
    ]

    # Build annotation cards
    ann_cards = ""
    for script in ann_scripts:
        source_rows = ""
        for db_name, endpoint, what in script["sources"]:
            source_rows += f'''<tr>
                <td class="src-db">{db_name}</td>
                <td class="src-endpoint dim">{endpoint}</td>
                <td class="src-what">{what}</td>
            </tr>'''

        ann_cards += f'''<div class="ann-card">
            <div class="ann-header">
                <span class="ann-num">{script["num"]}</span>
                <span class="ann-name">{script["name"]}</span>
                <span class="ann-file dim">{script["file"]}</span>
            </div>
            <table class="src-table">{source_rows}</table>
        </div>'''

    # Build compact pipeline flow
    flow_items = ""
    for i, (title, scripts, details) in enumerate(stages):
        details_html = "".join(f"<li>{d}</li>" for d in details)
        arrow = '<div class="flow-arrow">▾</div>' if i < len(stages) - 1 else ""
        flow_items += f'''<div class="flow-stage">
            <div class="flow-header">
                <span class="flow-title">{title}</span>
                <span class="flow-scripts dim">{scripts}</span>
            </div>
            <ul class="flow-details">{details_html}</ul>
        </div>{arrow}'''

    # Database inventory
    db_entries = [
        ("Ensembl REST API", "rest.ensembl.org", "Gene structure, variants, regulatory, conservation, ncRNA, phenotypes"),
        ("Ensembl GTF", "ftp.ensembl.org", "Local gene annotations (62K genes, parsed offline)"),
        ("GRCh38 FASTA", "ftp.ensembl.org", "Local reference genome (3.1 GB, indexed with samtools)"),
        ("UCSC Genome Browser", "api.genome.ucsc.edu", "PhastCons elements, RepeatMasker tracks"),
        ("ENCODE", "encodeproject.org", "Histone ChIP-seq, DNase-seq experiment metadata"),
        ("GTEx", "gtexportal.org", "Tissue-specific gene expression (54 tissues, median TPM)"),
        ("JASPAR", "jaspar.elixir.no", "Transcription factor binding site PFMs (position frequency matrices)"),
        ("UniProt", "uniprot.org", "Protein function annotations, domains, active sites"),
        ("OpenTargets", "platform.opentargets.org", "Disease–gene associations via GraphQL"),
        ("GWAS Catalog", "ebi.ac.uk/gwas", "Genome-wide association study SNPs"),
        ("Gemini API", "generativelanguage.googleapis.com", "Natural language description + reasoning trace generation"),
    ]

    db_rows = ""
    for name, url, desc in db_entries:
        db_rows += f'<tr><td class="db-name">{name}</td><td class="dim">{url}</td><td>{desc}</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DNA Annotation Pipeline — Overview</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: "Inter", sans-serif;
    font-size: 12px;
    font-weight: 300;
    line-height: 1.6;
    color: #1a1a1a;
    background: #fafafa;
    max-width: 960px;
    margin: 40px auto;
    padding: 0 20px;
  }}
  h1 {{
    font-family: "JetBrains Mono", monospace;
    font-size: 16px;
    font-weight: 400;
    letter-spacing: 2px;
    border-bottom: 1px solid #333;
    padding-bottom: 8px;
    margin-bottom: 4px;
  }}
  .meta {{
    font-family: "JetBrains Mono", monospace;
    color: #888;
    font-size: 10px;
    font-weight: 300;
    letter-spacing: 0.5px;
    margin-bottom: 32px;
  }}
  details {{ margin-bottom: 4px; }}
  summary {{
    font-family: "JetBrains Mono", monospace;
    cursor: pointer;
    user-select: none;
    font-size: 11px;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #444;
    margin-top: 32px;
    margin-bottom: 8px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
    list-style: none;
  }}
  summary::before {{ content: "▸ "; color: #999; }}
  details[open] > summary::before {{ content: "▾ "; }}
  summary::-webkit-details-marker {{ display: none; }}

  /* Pipeline flow */
  .flow-stage {{
    border: 1px solid #ddd;
    background: #fff;
    padding: 10px 14px;
    margin: 0;
  }}
  .flow-header {{
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 4px;
  }}
  .flow-title {{
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
  }}
  .flow-scripts {{
    font-family: "JetBrains Mono", monospace;
    font-size: 9px;
  }}
  .flow-details {{
    list-style: none;
    padding: 0;
  }}
  .flow-details li {{
    font-size: 11px;
    color: #555;
    padding: 1px 0 1px 14px;
    position: relative;
  }}
  .flow-details li::before {{
    content: "—";
    position: absolute;
    left: 0;
    color: #ccc;
  }}
  .flow-arrow {{
    text-align: center;
    color: #bbb;
    font-size: 14px;
    line-height: 20px;
  }}

  /* Annotation cards */
  .ann-card {{
    border: 1px solid #ddd;
    background: #fff;
    padding: 10px 14px;
    margin-bottom: 8px;
  }}
  .ann-header {{
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 6px;
  }}
  .ann-num {{
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
    font-weight: 500;
    color: #999;
  }}
  .ann-name {{
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    font-weight: 500;
    color: #333;
  }}
  .ann-file {{
    font-family: "JetBrains Mono", monospace;
    font-size: 9px;
    margin-left: auto;
  }}
  .src-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 10px;
    margin: 0;
  }}
  .src-table td {{
    padding: 2px 8px 2px 0;
    border-bottom: 1px solid #f5f5f5;
    vertical-align: top;
  }}
  .src-table tr:last-child td {{
    border-bottom: none;
  }}
  .src-db {{
    font-family: "JetBrains Mono", monospace;
    font-weight: 500;
    white-space: nowrap;
    width: 120px;
    color: #555;
  }}
  .src-endpoint {{
    font-family: "JetBrains Mono", monospace;
    font-size: 9px;
    white-space: nowrap;
    width: 200px;
  }}
  .src-what {{
    color: #444;
  }}

  /* Shared */
  table {{
    font-family: "JetBrains Mono", monospace;
    width: 100%;
    border-collapse: collapse;
    font-size: 10px;
    font-weight: 300;
    margin: 8px 0 16px 0;
  }}
  th {{
    text-align: left;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 9px;
    color: #888;
    border-bottom: 1px solid #ccc;
    padding: 4px 8px 4px 0;
  }}
  td {{ padding: 4px 8px 4px 0; vertical-align: top; border-bottom: 1px solid #f0f0f0; }}
  .db-name {{ font-weight: 500; white-space: nowrap; }}
  .dim {{ color: #999; }}

  .file-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 6px;
    margin: 12px 0;
  }}
  .file-card {{
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
    border: 1px solid #e0e0e0;
    padding: 8px 10px;
    background: #fff;
  }}
  .file-card .fname {{ font-weight: 500; color: #333; margin-bottom: 2px; }}
  .file-card .fdesc {{ font-family: "Inter", sans-serif; font-weight: 300; color: #777; font-size: 10px; }}

  .highlight {{
    background: #f4f4f4;
    border: 1px solid #ddd;
    padding: 12px 16px;
    margin: 12px 0;
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    line-height: 1.8;
  }}

  @media print {{
    body {{ font-size: 10px; max-width: 100%; margin: 20px; }}
  }}
</style>
</head>
<body>

<h1>DNA Annotation Pipeline</h1>
<div class="meta">Gene-centric &middot; 10 annotation layers &middot; 11 databases &middot; LLM-powered reasoning</div>

<details open>
<summary>Pipeline</summary>
{flow_items}
</details>

<details open>
<summary>Annotation layers — per-database detail</summary>
{ann_cards}
</details>

<details open>
<summary>File inventory</summary>
<div class="ann-card">
  <div class="ann-header"><span class="ann-name">Core</span></div>
  <table class="src-table">
    <tr><td class="src-db">common.py</td><td class="src-what">Shared data model: GenomicRegion, Gene, Annotation, AnnotatedSequence dataclasses. Sequence fetching (local FASTA → NCBI → Ensembl fallback chain). Beta-distributed flanking sampler. Ensembl API retry logic.</td></tr>
    <tr><td class="src-db">fetch_genes.py</td><td class="src-what">Builds the gene list for a species. Parses Ensembl GTF locally (fastest), or falls back to BioMart / REST API. Filters by biotype (protein_coding, lncRNA, miRNA, tRNA, snoRNA, ...). Outputs genes_{{species}}.json.</td></tr>
    <tr><td class="src-db">annotate_all.py</td><td class="src-what">Orchestrator. Takes a gene name (--gene BRCA1) or gene list index (--genes-file + --index). Expands gene to region with randomized flanking, runs all 10 annotation scripts, merges into one AnnotatedSequence JSON.</td></tr>
  </table>
</div>
<div class="ann-card">
  <div class="ann-header"><span class="ann-name">Annotation scripts</span></div>
  <table class="src-table">
    <tr><td class="src-db">01_gene_structure.py</td><td class="src-what">Ensembl overlap → genes, transcripts, exons, CDS. Local regex → start/stop codons, splice signals.</td></tr>
    <tr><td class="src-db">02_regulatory_elements.py</td><td class="src-what">JASPAR PFM scan → TFBS. Biopython → restriction sites. Local → TATA/CAAT/GC boxes, CpG islands.</td></tr>
    <tr><td class="src-db">03_functional_classification.py</td><td class="src-what">Ensembl overlap → per-position CDS/exon/UTR/intron/intergenic classification + regulatory overlay.</td></tr>
    <tr><td class="src-db">04_chromatin_states.py</td><td class="src-what">Ensembl Regulatory Build → promoters, enhancers, CTCF. ENCODE → histone mark experiments. Rule-based state inference.</td></tr>
    <tr><td class="src-db">05_conservation.py</td><td class="src-what">Ensembl Compara → GERP constrained elements. UCSC → phastCons 100-way vertebrate elements. Local → GC content.</td></tr>
    <tr><td class="src-db">06_variants.py</td><td class="src-what">Ensembl variation → known SNPs with clinical significance. Ensembl VEP → consequence prediction, SIFT, PolyPhen.</td></tr>
    <tr><td class="src-db">07_expression_epigenetic.py</td><td class="src-what">Ensembl → regulatory features. ENCODE → DNase/ChIP-seq availability. GTEx → tissue expression (54 tissues, TPM).</td></tr>
    <tr><td class="src-db">08_repetitive_elements.py</td><td class="src-what">UCSC RepeatMasker → Alu, LINE, SINE with divergence. Ensembl → repeats. Local → tandem/simple repeats.</td></tr>
    <tr><td class="src-db">09_mirna_regulatory_rna.py</td><td class="src-what">Ensembl → ncRNA genes. Local → miRNA seed matching (8 common miRNAs), stem-loop structure prediction.</td></tr>
    <tr><td class="src-db">10_disease_clinical.py</td><td class="src-what">Ensembl → clinical variants + phenotypes. OpenTargets → disease scores. GWAS Catalog → association hits.</td></tr>
  </table>
</div>
<div class="ann-card">
  <div class="ann-header"><span class="ann-name">Post-processing</span></div>
  <table class="src-table">
    <tr><td class="src-db">summarize.py</td><td class="src-what">Reads raw annotation JSON. Drops regex noise (codons, GC windows, hairpins). Deduplicates by rs_id + position. Collapses dense types to top 5 + summary. Typical 98% reduction.</td></tr>
    <tr><td class="src-db">describe.py</td><td class="src-what">Sends concise annotations to Gemini 3.1 Pro. Two calls: (1) description — minimal external facts, (2) reasoning — chain-of-thought from raw DNA. Outputs _described.json.</td></tr>
  </table>
</div>
<div class="ann-card">
  <div class="ann-header"><span class="ann-name">Reporting</span></div>
  <table class="src-table">
    <tr><td class="src-db">visualize.py</td><td class="src-what">Reads a _described.json. Generates monochrome HTML with formatted sequence, annotations table, description, and reasoning. Collapsible sections.</td></tr>
    <tr><td class="src-db">gene_report.py</td><td class="src-what">Reads genes_{{species}}.json. Generates HTML with size distribution table, log-scale histograms (shared axes), chromosome bar chart.</td></tr>
    <tr><td class="src-db">cost_report.py</td><td class="src-what">Generates HTML cost estimates for multi-organism dataset generation across Gemini model tiers, with batch pricing.</td></tr>
    <tr><td class="src-db">pipeline_overview.py</td><td class="src-what">This file. Generates the pipeline infographic.</td></tr>
  </table>
</div>
</details>

<details open>
<summary>Database inventory ({len(db_entries)})</summary>
<table>
  <tr><th>Database</th><th>Endpoint</th><th>What it provides</th></tr>
  {db_rows}
</table>
</details>

<details>
<summary>Quick start</summary>
<div class="highlight">
# Setup<br>
pip install biopython requests pandas<br>
brew install samtools<br>
<br>
# Download reference genome + gene annotations<br>
wget https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz<br>
gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz<br>
samtools faidx Homo_sapiens.GRCh38.dna.primary_assembly.fa<br>
wget https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz<br>
<br>
export REFERENCE_FASTA=Homo_sapiens.GRCh38.dna.primary_assembly.fa<br>
<br>
# Fetch gene list<br>
python fetch_genes.py --gtf Homo_sapiens.GRCh38.113.gtf.gz<br>
<br>
# Annotate a single gene<br>
python annotate_all.py --gene BRCA1 --seed 42<br>
<br>
# Summarize + describe<br>
python summarize.py BRCA1_output.json<br>
export GEMINI_API_KEY=your_key<br>
python describe.py BRCA1_output_concise.json<br>
<br>
# Visualize<br>
python visualize.py BRCA1_output_concise_described.json
</div>
</details>

</body>
</html>"""


def main():
    html = generate_html()
    output = "pipeline_overview.html"
    with open(output, "w") as f:
        f.write(html)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
