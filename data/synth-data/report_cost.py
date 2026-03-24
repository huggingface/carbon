#!/usr/bin/env python3
"""
Generate an HTML cost report for the multi-organism annotation pipeline.
"""

import json

# ---------------------------------------------------------------------------
# Per-window constants (from our test run on 501bp BRCA1 region)
# ---------------------------------------------------------------------------
INPUT_TOKENS_PER_WINDOW = 4656 * 2     # two Gemini calls
OUTPUT_TOKENS_PER_WINDOW = 102 + 1235  # description + reasoning
DATASET_TOKENS_PER_WINDOW = 1838       # what we store (seq + desc + reasoning)
WINDOW_SIZE = 500

# ---------------------------------------------------------------------------
# Organisms
# ---------------------------------------------------------------------------
# (name, genome_bp, high_frac, medium_frac, annotation_depth)
ORGANISMS = [
    # Well-annotated model organisms
    ("Human (GRCh38)",           3_088_286_401, 0.105, 0.22, "10/10"),
    ("Mouse (mm39)",             2_728_222_451, 0.105, 0.22, "9/10"),
    ("Rat (rn7)",                2_647_000_000, 0.105, 0.22, "7/10"),
    ("Zebrafish (danRer11)",     1_345_101_688, 0.105, 0.22, "7/10"),
    ("Chicken (galGal6)",        1_065_365_434, 0.105, 0.20, "6/10"),
    ("Frog (xenTro10)",          1_451_000_000, 0.105, 0.20, "5/10"),
    ("Fruit fly (dm6)",            143_726_002, 0.35,  0.25, "7/10"),
    ("C. elegans (ce11)",          100_286_401, 0.35,  0.25, "7/10"),
    # Other vertebrates
    ("Dog (canFam6)",            2_410_976_875, 0.105, 0.22, "5/10"),
    ("Cat (felCat9)",            2_521_897_038, 0.105, 0.22, "4/10"),
    ("Pig (susScr11)",           2_501_912_388, 0.105, 0.22, "5/10"),
    ("Cow (bosTau9)",            2_715_853_792, 0.105, 0.22, "5/10"),
    ("Horse (equCab3)",          2_506_966_135, 0.105, 0.22, "4/10"),
    ("Sheep (oviAri4)",          2_615_516_299, 0.105, 0.22, "4/10"),
    ("Macaque (rheMac10)",       2_946_843_737, 0.105, 0.22, "5/10"),
    ("Gorilla (gorGor6)",        3_063_362_754, 0.105, 0.22, "4/10"),
    ("Chimpanzee (panTro6)",     3_050_398_073, 0.105, 0.22, "4/10"),
    # Fish
    ("Fugu (fr3)",                 391_485_651, 0.20,  0.20, "4/10"),
    ("Medaka (oryLat2)",           869_000_000, 0.105, 0.22, "4/10"),
    ("Stickleback (gasAcu1)",      463_354_048, 0.20,  0.20, "4/10"),
    # Plants
    ("Arabidopsis (TAIR10)",       119_668_634, 0.35,  0.25, "4/10"),
    ("Rice (IRGSP-1.0)",          373_245_519, 0.20,  0.20, "4/10"),
    ("Maize (Zm-B73-v5)",       2_135_083_061, 0.105, 0.22, "3/10"),
    # Fungi
    ("Yeast (sacCer3)",            12_157_105, 0.35,  0.30, "5/10"),
    ("Aspergillus (ASM)",          30_000_000, 0.35,  0.25, "3/10"),
    # Pathogens / other
    ("Plasmodium (ASM276v2)",      23_332_831, 0.35,  0.25, "3/10"),
    ("E. coli (K-12)",              4_641_652, 0.50,  0.30, "3/10"),
]

# ---------------------------------------------------------------------------
# Gemini models
# ---------------------------------------------------------------------------
# (name, input_$/M, output_$/M, batch_available, quality_tier)
MODELS = [
    ("Gemini 3.1 Pro Preview",    2.00, 12.00, True,  "Best"),
    ("Gemini 3 Flash Preview",    0.50,  3.00, True,  "Good"),
    ("Gemini 3.1 Flash-Lite",     0.25,  1.50, True,  "Basic"),
    ("Gemini 2.5 Pro",            1.25, 10.00, True,  "Best"),
    ("Gemini 2.5 Flash",          0.30,  2.50, True,  "Good"),
    ("Gemini 2.5 Flash-Lite",     0.10,  0.40, True,  "Basic"),
]

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def windows(genome_bp, frac):
    return int(genome_bp * frac / WINDOW_SIZE)

def gemini_cost(n_windows, inp_price, out_price):
    return (n_windows * INPUT_TOKENS_PER_WINDOW * inp_price +
            n_windows * OUTPUT_TOKENS_PER_WINDOW * out_price) / 1e6

def fmt_cost(v):
    if v == 0:
        return "Free*"
    if v < 1000:
        return f"${v:,.0f}"
    return f"${v:,.0f}"

def fmt_tokens(v):
    if v >= 1e9:
        return f"{v/1e9:.1f}B"
    if v >= 1e6:
        return f"{v/1e6:.0f}M"
    return f"{v:,.0f}"

# Per-organism data
org_data = []
for name, genome, high_f, med_f, depth in ORGANISMS:
    high_w = windows(genome, high_f)
    highmed_w = windows(genome, high_f + med_f)
    org_data.append({
        "name": name,
        "genome": genome,
        "high_windows": high_w,
        "highmed_windows": highmed_w,
        "high_tokens": high_w * DATASET_TOKENS_PER_WINDOW,
        "highmed_tokens": highmed_w * DATASET_TOKENS_PER_WINDOW,
        "depth": depth,
    })

total_high_w = sum(o["high_windows"] for o in org_data)
total_highmed_w = sum(o["highmed_windows"] for o in org_data)
total_high_tok = sum(o["high_tokens"] for o in org_data)
total_highmed_tok = sum(o["highmed_tokens"] for o in org_data)

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def build_html():
    # Organism table rows
    org_rows = ""
    for o in org_data:
        org_rows += f"""<tr>
            <td>{o['name']}</td>
            <td class="num">{o['genome']/1e9:.3f}</td>
            <td class="num">{o['high_windows']:,}</td>
            <td class="num">{fmt_tokens(o['high_tokens'])}</td>
            <td class="num">{o['highmed_windows']:,}</td>
            <td class="num">{fmt_tokens(o['highmed_tokens'])}</td>
            <td class="num">{o['depth']}</td>
        </tr>"""

    org_rows += f"""<tr class="total">
            <td>Total (27 organisms)</td>
            <td class="num"></td>
            <td class="num">{total_high_w:,}</td>
            <td class="num">{fmt_tokens(total_high_tok)}</td>
            <td class="num">{total_highmed_w:,}</td>
            <td class="num">{fmt_tokens(total_highmed_tok)}</td>
            <td class="num"></td>
        </tr>"""

    # Cost table rows (standard pricing)
    cost_rows = ""
    for mname, ip, op, batch, quality in MODELS:
        ch = gemini_cost(total_high_w, ip, op)
        chm = gemini_cost(total_highmed_w, ip, op)
        cost_rows += f"""<tr>
            <td>{mname}</td>
            <td class="num">${ip:.2f}</td>
            <td class="num">${op:.2f}</td>
            <td class="num">{fmt_cost(ch)}</td>
            <td class="num">{fmt_cost(chm)}</td>
            <td class="dim">{quality}</td>
        </tr>"""

    # Batch cost rows
    batch_rows = ""
    for mname, ip, op, batch, quality in MODELS:
        if not batch:
            batch_rows += f"""<tr>
                <td>{mname}</td>
                <td class="num" colspan="4">No batch API</td>
            </tr>"""
            continue
        ch = gemini_cost(total_high_w, ip*0.5, op*0.5)
        chm = gemini_cost(total_highmed_w, ip*0.5, op*0.5)
        batch_rows += f"""<tr>
            <td>{mname}</td>
            <td class="num">{fmt_cost(ch)}</td>
            <td class="num">{fmt_cost(chm)}</td>
            <td class="dim">{quality}</td>
        </tr>"""

    # Per-organism cost breakdown for sweet spot model (3 Flash batch)
    per_org_rows = ""
    for o in org_data:
        ch = gemini_cost(o["high_windows"], 0.25, 1.50)
        chm = gemini_cost(o["highmed_windows"], 0.25, 1.50)
        per_org_rows += f"""<tr>
            <td>{o['name']}</td>
            <td class="num">{o['high_windows']:,}</td>
            <td class="num">{fmt_cost(ch)}</td>
            <td class="num">{o['highmed_windows']:,}</td>
            <td class="num">{fmt_cost(chm)}</td>
        </tr>"""
    per_org_rows += f"""<tr class="total">
        <td>Total</td>
        <td class="num">{total_high_w:,}</td>
        <td class="num">{fmt_cost(gemini_cost(total_high_w, 0.25, 1.50))}</td>
        <td class="num">{total_highmed_w:,}</td>
        <td class="num">{fmt_cost(gemini_cost(total_highmed_w, 0.25, 1.50))}</td>
    </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DNA Annotation Pipeline — Cost Report</title>
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
  th.num {{ text-align: right; }}
  td {{ padding: 3px 8px 3px 0; vertical-align: top; border-bottom: 1px solid #f0f0f0; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.total {{ border-top: 1px solid #999; font-weight: 500; }}
  tr.total td {{ border-bottom: none; padding-top: 6px; }}
  .dim {{ color: #999; }}
  .note {{
    font-family: "Inter", sans-serif;
    font-size: 11px;
    color: #666;
    margin: 8px 0 16px 0;
    line-height: 1.6;
  }}
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
    details[open] {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>

<h1>DNA Annotation Pipeline — Cost Report</h1>
<div class="meta">27 organisms &middot; 10 annotation categories &middot; Gemini-powered description &amp; reasoning</div>

<div class="highlight">
  <strong>Pipeline:</strong> genomic region → local FASTA → 10 annotation scripts (Ensembl, UCSC, ENCODE, GTEx, OpenTargets, JASPAR) → summarize → Gemini (description + reasoning trace)<br>
  <strong>Window size:</strong> {WINDOW_SIZE} bp &middot;
  <strong>Per window:</strong> ~{INPUT_TOKENS_PER_WINDOW:,} input tokens, ~{OUTPUT_TOKENS_PER_WINDOW:,} output tokens →
  ~{DATASET_TOKENS_PER_WINDOW:,} tokens stored
</div>

<details open>
<summary>Dataset size by organism</summary>
<p class="note">
  HIGH = protein-coding exons, UTRs, conserved non-coding, promoters, enhancers (~10.5% of large genomes, ~35% of compact genomes).<br>
  HIGH+MED adds introns (splice sites, some regulatory elements).<br>
  Window = a fixed-length 500 bp chunk of the genome, annotated independently. Each window becomes one training example: 500 bases of DNA → description + reasoning trace.
</p>
<table>
  <tr>
    <th>Organism</th>
    <th class="num">Genome (Gb)</th>
    <th class="num">HIGH windows</th>
    <th class="num">HIGH tokens</th>
    <th class="num">HIGH+MED windows</th>
    <th class="num">HIGH+MED tokens</th>
    <th class="num">Depth</th>
  </tr>
  {org_rows}
</table>
</details>

<details open>
<summary>Gemini cost — standard pricing</summary>
<p class="note">Cost to generate description + reasoning for all {total_high_w:,} (HIGH) or {total_highmed_w:,} (HIGH+MED) windows across 27 organisms. * Free tier has rate limits (typically 15 RPM / 1M TPD).</p>
<table>
  <tr>
    <th>Model</th>
    <th class="num">In $/M</th>
    <th class="num">Out $/M</th>
    <th class="num">HIGH (all orgs)</th>
    <th class="num">HIGH+MED (all orgs)</th>
    <th>Quality</th>
  </tr>
  {cost_rows}
</table>
</details>

<details open>
<summary>Gemini cost — batch API (50% off)</summary>
<p class="note">Batch API processes requests asynchronously (up to 24h turnaround) at half price.</p>
<table>
  <tr>
    <th>Model</th>
    <th class="num">HIGH (all orgs)</th>
    <th class="num">HIGH+MED (all orgs)</th>
    <th>Quality</th>
  </tr>
  {batch_rows}
</table>
</details>

<details>
<summary>Per-organism cost breakdown (3 Flash + batch)</summary>
<p class="note">Gemini 3 Flash Preview with Batch API — best quality/cost ratio.</p>
<table>
  <tr>
    <th>Organism</th>
    <th class="num">HIGH windows</th>
    <th class="num">HIGH cost</th>
    <th class="num">HIGH+MED windows</th>
    <th class="num">HIGH+MED cost</th>
  </tr>
  {per_org_rows}
</table>
</details>

<details>
<summary>Genome composition reference</summary>
<table>
  <tr><th>Region type</th><th class="num">% of genome</th><th>Priority</th><th>Rationale</th></tr>
  <tr><td>Protein-coding exons</td><td class="num">1.5%</td><td>HIGH</td><td>Every base matters for protein function</td></tr>
  <tr><td>UTRs (5'+3')</td><td class="num">1.0%</td><td>HIGH</td><td>Translation regulation, miRNA targets</td></tr>
  <tr><td>Conserved non-coding</td><td class="num">3.0%</td><td>HIGH</td><td>Enhancers, CNEs under purifying selection</td></tr>
  <tr><td>Promoters + enhancers</td><td class="num">5.0%</td><td>HIGH</td><td>ENCODE-defined regulatory elements</td></tr>
  <tr><td>Introns</td><td class="num">22.0%</td><td>MEDIUM</td><td>Splice sites, some regulatory; mostly neutral</td></tr>
  <tr><td>Repetitive elements</td><td class="num">45.0%</td><td>LOW</td><td>Transposon fossils, satellite DNA</td></tr>
  <tr><td>Intergenic non-repetitive</td><td class="num">22.5%</td><td>LOW</td><td>Mostly neutral; sparse annotation</td></tr>
</table>
</details>

<details>
<summary>Annotation database coverage by organism</summary>
<table>
  <tr>
    <th>Database</th>
    <th>Human</th>
    <th>Mouse</th>
    <th>Fly / Worm / Fish</th>
    <th>Yeast</th>
    <th>Plants</th>
  </tr>
  <tr><td>Ensembl (genes, variants)</td><td>Full</td><td>Full</td><td>Full</td><td>Full</td><td>Full</td></tr>
  <tr><td>UCSC (conservation, repeats)</td><td>Full</td><td>Full</td><td>Partial</td><td>sacCer3</td><td>—</td></tr>
  <tr><td>ENCODE (chromatin)</td><td>Full</td><td>Full</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>GTEx (expression)</td><td>Full</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>ClinVar / GWAS / OpenTargets</td><td>Full</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>JASPAR (TFBS)</td><td>Full</td><td>Full</td><td>Partial</td><td>Fungi</td><td>Plants</td></tr>
  <tr><td>RepeatMasker</td><td>Full</td><td>Full</td><td>Full</td><td>Full</td><td>Full</td></tr>
</table>
</details>

</body>
</html>"""


if __name__ == "__main__":
    html = build_html()
    output = "cost_report.html"
    with open(output, "w") as f:
        f.write(html)
    print(f"Saved to {output}")
