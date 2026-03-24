#!/usr/bin/env python3
"""
Generate a minimal monochrome HTML visualization of an annotated DNA sequence.

Usage:
    python visualize.py                                      # reads combined_output_concise_described.json
    python visualize.py my_annotations_concise_described.json
"""

import json
import sys
import html


def generate_html(data: dict) -> str:
    region = data["region"]
    sequence = data["sequence"]
    description = data.get("description", "")
    reasoning = data.get("reasoning", "")
    annotations = data.get("annotations", [])

    chrom = region.get("chrom", "?")
    start = region.get("start", 0)
    end = region.get("end", 0)
    assembly = region.get("assembly", "")
    length = len(sequence)

    # Format sequence in blocks of 10, lines of 60
    seq_lines = []
    for i in range(0, length, 60):
        line_bases = sequence[i:i+60]
        blocks = [line_bases[j:j+10] for j in range(0, len(line_bases), 10)]
        pos_label = f"{i+1:>5}"
        seq_lines.append(f'<span class="pos">{pos_label}</span>  {"  ".join(blocks)}')

    seq_html = "\n".join(seq_lines)

    # Group annotations by category
    by_cat = {}
    for a in annotations:
        by_cat.setdefault(a.get("category", "other"), []).append(a)

    # Build annotation rows
    ann_sections = []
    for cat, anns in by_cat.items():
        rows = []
        for a in anns:
            s, e = a.get("start", 0), a.get("end", 0)
            atype = html.escape(a.get("type", ""))
            label = html.escape(a.get("label", ""))
            score = a.get("score")
            score_str = f'<span class="score">{score}</span>' if score is not None else ""

            # Check if it's a summary annotation
            meta = a.get("metadata", {})
            extra = ""
            if meta.get("total_count"):
                extra = f' <span class="dim">({meta["total_count"]} total)</span>'
            if meta.get("clinical_significance_counts"):
                counts = meta["clinical_significance_counts"]
                parts = [f"{k}: {v}" for k, v in list(counts.items())[:4]]
                extra += f' <span class="dim">[{", ".join(parts)}]</span>'

            rows.append(
                f'<tr>'
                f'<td class="coord">{s}–{e}</td>'
                f'<td class="type">{atype}</td>'
                f'<td>{label}{extra} {score_str}</td>'
                f'</tr>'
            )

        ann_sections.append(f"""
        <div class="cat-section">
            <h3>{html.escape(cat)}</h3>
            <table>{"".join(rows)}</table>
        </div>""")

    # Format description and reasoning as paragraphs
    def to_paragraphs(text):
        if not text:
            return "<p class='dim'>Not generated.</p>"
        paras = text.strip().split("\n\n")
        return "".join(f"<p>{html.escape(p.strip())}</p>" for p in paras if p.strip())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>chr{chrom}:{start}-{end}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap');
  body {{
    font-family: "Inter", "Helvetica Neue", sans-serif;
    font-size: 12px;
    font-weight: 300;
    line-height: 1.6;
    color: #1a1a1a;
    background: #fafafa;
    max-width: 900px;
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
  h2 {{
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #555;
    margin-top: 32px;
    margin-bottom: 8px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
  }}
  h3 {{
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #777;
    margin-top: 16px;
    margin-bottom: 4px;
  }}
  .meta {{
    font-family: "JetBrains Mono", monospace;
    color: #888;
    font-size: 10px;
    font-weight: 300;
    letter-spacing: 0.5px;
    margin-bottom: 24px;
  }}
  .seq-block {{
    font-family: "JetBrains Mono", monospace;
    background: #f4f4f4;
    border: 1px solid #ddd;
    padding: 12px 16px;
    overflow-x: auto;
    white-space: pre;
    font-size: 11px;
    font-weight: 300;
    line-height: 1.8;
    letter-spacing: 1px;
  }}
  .pos {{ color: #999; }}
  table {{
    font-family: "JetBrains Mono", monospace;
    width: 100%;
    border-collapse: collapse;
    font-size: 10px;
    font-weight: 300;
  }}
  tr {{ border-bottom: 1px solid #eee; }}
  td {{ padding: 3px 8px 3px 0; vertical-align: top; }}
  .coord {{
    white-space: nowrap;
    color: #888;
    width: 70px;
    font-variant-numeric: tabular-nums;
  }}
  .type {{
    color: #555;
    width: 180px;
    font-weight: 500;
  }}
  .score {{
    background: #eee;
    padding: 1px 5px;
    border-radius: 2px;
    font-size: 10px;
    color: #555;
  }}
  .dim {{ color: #999; font-size: 10px; }}
  .text-block {{
    font-family: "Inter", sans-serif;
    font-size: 13px;
    font-weight: 300;
    line-height: 1.75;
    color: #2a2a2a;
  }}
  .text-block p {{ margin-bottom: 12px; }}
  .cat-section {{ margin-bottom: 8px; }}
  details {{
    margin-bottom: 4px;
  }}
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
  summary::before {{
    content: "▸ ";
    color: #999;
  }}
  details[open] > summary::before {{
    content: "▾ ";
  }}
  summary::-webkit-details-marker {{ display: none; }}
  @media print {{
    body {{ font-size: 10px; max-width: 100%; margin: 20px; }}
    .seq-block {{ font-size: 9px; }}
    details {{ open: true; }}
  }}
</style>
</head>
<body>

<h1>chr{chrom}:{start:,}–{end:,}</h1>
<div class="meta">{assembly} &middot; {length} bp &middot; {len(annotations)} annotations</div>

<details open>
<summary>Sequence</summary>
<div class="seq-block">{seq_html}</div>
</details>

<details open>
<summary>Description</summary>
<div class="text-block">{to_paragraphs(description)}</div>
</details>

<details open>
<summary>Reasoning</summary>
<div class="text-block">{to_paragraphs(reasoning)}</div>
</details>

<details>
<summary>Annotations ({len(annotations)})</summary>
{"".join(ann_sections)}
</details>

</body>
</html>"""


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "combined_output_concise_described.json"

    with open(input_file) as f:
        data = json.load(f)

    # If annotations aren't in the described file, try loading from concise
    if "annotations" not in data:
        concise_file = input_file.replace("_described.json", ".json")
        try:
            with open(concise_file) as f:
                concise = json.load(f)
            data["annotations"] = concise.get("annotations", [])
        except FileNotFoundError:
            data["annotations"] = []

    output_html = generate_html(data)
    output_file = input_file.replace(".json", ".html")

    with open(output_file, "w") as f:
        f.write(output_html)

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
