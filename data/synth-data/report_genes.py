#!/usr/bin/env python3
"""
Generate an HTML report of gene statistics from a genes_*.json file.

Usage:
    python gene_report.py                    # reads genes_human.json
    python gene_report.py genes_mouse.json
"""

import json
import math
import sys
import statistics
from collections import defaultdict


def fmt_bp(v):
    if v >= 1_000_000:
        return f"{v/1e6:.1f}Mb"
    if v >= 1_000:
        return f"{v/1e3:.1f}kb"
    return f"{v:.0f}bp"


def make_histogram_svg(sizes, width=800, height=180, bins=50, label="",
                       global_log_min=None, global_log_max=None,
                       show_x_axis=True, show_x_label=True, global_max_count=None):
    """Generate an SVG histogram (monochrome, log-scale x-axis)."""
    if not sizes:
        return ""

    # Log-scale bins — use global range if provided for shared axis
    log_min = global_log_min if global_log_min is not None else math.log10(max(min(sizes), 1))
    log_max = global_log_max if global_log_max is not None else math.log10(max(sizes))
    bin_edges = [10 ** (log_min + i * (log_max - log_min) / bins) for i in range(bins + 1)]

    counts = [0] * bins
    for s in sizes:
        for i in range(bins):
            if s < bin_edges[i + 1] or i == bins - 1:
                counts[i] += 1
                break

    max_count = global_max_count if global_max_count is not None else (max(counts) if counts else 1)
    margin_left = 50
    margin_bottom = 40 if show_x_axis else 8
    margin_top = 20
    margin_right = 20
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top
    bar_w = plot_w / bins

    bars = ""
    for i, count in enumerate(counts):
        if count == 0:
            continue
        bar_h = (count / max_count) * plot_h
        x = margin_left + i * bar_w
        y = margin_top + plot_h - bar_h
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 1, 1):.1f}" height="{bar_h:.1f}" fill="#888" />\n'

    # X-axis labels (log scale)
    x_labels = ""
    tick_values = [100, 1000, 10000, 100000, 1000000]
    for tv in tick_values:
        log_pos = (math.log10(tv) - log_min) / (log_max - log_min)
        if log_pos < -0.01 or log_pos > 1.01:
            continue
        x = margin_left + log_pos * plot_w
        # Always draw tick marks as vertical grid lines
        x_labels += f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" stroke="#eee" />\n'
        if show_x_axis:
            x_labels += f'<line x1="{x:.1f}" y1="{margin_top + plot_h}" x2="{x:.1f}" y2="{margin_top + plot_h + 4}" stroke="#999" />\n'
            x_labels += f'<text x="{x:.1f}" y="{margin_top + plot_h + 16}" text-anchor="middle" fill="#888" font-size="9">{fmt_bp(tv)}</text>\n'

    # Y-axis labels
    y_labels = ""
    for frac in [0, 0.5, 1.0]:
        y = margin_top + plot_h * (1 - frac)
        val = int(max_count * frac)
        y_labels += f'<line x1="{margin_left - 4}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#999" />\n'
        y_labels += f'<text x="{margin_left - 8}" y="{y + 3:.1f}" text-anchor="end" fill="#888" font-size="9">{val:,}</text>\n'

    title_text = f'<text x="{margin_left}" y="{margin_top - 6}" fill="#555" font-size="10" font-weight="500">{label}</text>' if label else ""
    actual_height = margin_top + plot_h + margin_bottom
    axis_label = f'<text x="{margin_left + plot_w / 2}" y="{actual_height - 4}" text-anchor="middle" fill="#888" font-size="9">gene size (log scale)</text>' if (show_x_axis and show_x_label) else ""
    return f"""<svg width="{width}" height="{actual_height}" xmlns="http://www.w3.org/2000/svg" style="font-family: 'JetBrains Mono', monospace;">
  {title_text}
  <rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#ddd" />
  {x_labels}
  {y_labels}
  {bars}
  {axis_label}
</svg>"""


def generate_html(genes):
    # Group by biotype
    by_biotype = defaultdict(list)
    for g in genes:
        size = g["end"] - g["start"] + 1
        by_biotype[g["biotype"]].append(size)

    sorted_biotypes = sorted(by_biotype.items(), key=lambda x: -len(x[1]))
    all_sizes = [s for sizes in by_biotype.values() for s in sizes]

    # Per-biotype stats
    stat_rows = ""
    for biotype, sizes in sorted_biotypes:
        sizes.sort()
        n = len(sizes)
        p25 = sizes[n // 4]
        p75 = sizes[3 * n // 4]
        p95 = sizes[int(n * 0.95)]
        med = statistics.median(sizes)
        mean = statistics.mean(sizes)
        total = sum(sizes)

        stat_rows += f"""<tr>
            <td>{biotype}</td>
            <td class="num">{n:,}</td>
            <td class="num">{fmt_bp(min(sizes))}</td>
            <td class="num">{fmt_bp(p25)}</td>
            <td class="num">{fmt_bp(med)}</td>
            <td class="num">{fmt_bp(mean)}</td>
            <td class="num">{fmt_bp(p75)}</td>
            <td class="num">{fmt_bp(p95)}</td>
            <td class="num">{fmt_bp(max(sizes))}</td>
            <td class="num">{fmt_bp(total)}</td>
        </tr>"""

    # Totals
    all_sizes.sort()
    n = len(all_sizes)
    stat_rows += f"""<tr class="total">
        <td>All</td>
        <td class="num">{n:,}</td>
        <td class="num">{fmt_bp(min(all_sizes))}</td>
        <td class="num">{fmt_bp(all_sizes[n // 4])}</td>
        <td class="num">{fmt_bp(statistics.median(all_sizes))}</td>
        <td class="num">{fmt_bp(statistics.mean(all_sizes))}</td>
        <td class="num">{fmt_bp(all_sizes[3 * n // 4])}</td>
        <td class="num">{fmt_bp(all_sizes[int(n * 0.95)])}</td>
        <td class="num">{fmt_bp(max(all_sizes))}</td>
        <td class="num">{fmt_bp(sum(all_sizes))}</td>
    </tr>"""

    # Shared axis range across all histograms
    global_log_min = math.log10(max(min(all_sizes), 1))
    global_log_max = math.log10(max(all_sizes))

    # Histograms — all share the same x-axis, only bottom one shows labels
    top_biotypes = sorted_biotypes[:4]

    # Compute shared y-axis max across all histograms
    global_y_max = 2800

    all_hist = make_histogram_svg(all_sizes, width=920, height=220, label="All genes",
        global_log_min=global_log_min, global_log_max=global_log_max,
        global_max_count=global_y_max, show_x_label=True)

    # 2x2 grid for biotype histograms
    biotype_svgs = []
    for i, (biotype, sizes) in enumerate(top_biotypes):
        is_bottom = (i >= 2)
        biotype_svgs.append(make_histogram_svg(sizes, width=450, height=180,
            label=f"{biotype} ({len(sizes):,})",
            global_log_min=global_log_min, global_log_max=global_log_max,
            global_max_count=global_y_max, show_x_label=is_bottom))

    # Arrange as 2x2
    biotype_grid = '<div class="hist-grid">'
    for i, svg in enumerate(biotype_svgs):
        biotype_grid += f'<div class="hist-cell">{svg}</div>'
    biotype_grid += '</div>'

    # Chromosome distribution
    by_chrom = defaultdict(int)
    for g in genes:
        by_chrom[g["chrom"]] += 1

    chrom_order = {str(i): i for i in range(1, 23)}
    chrom_order.update({"X": 23, "Y": 24, "MT": 25})
    sorted_chroms = sorted(by_chrom.items(), key=lambda x: chrom_order.get(x[0], 99))

    max_chrom_count = max(by_chrom.values())
    chrom_bars = ""
    bar_w = 28
    chart_h = 140
    chart_margin_top = 30
    chart_margin_bottom = 16
    plot_area_h = chart_h - chart_margin_top - chart_margin_bottom
    for i, (chrom, count) in enumerate(sorted_chroms):
        bar_h = (count / max_chrom_count) * plot_area_h
        x = 40 + i * (bar_w + 4)
        y = chart_margin_top + plot_area_h - bar_h
        chrom_bars += f'<rect x="{x}" y="{y:.0f}" width="{bar_w}" height="{bar_h:.0f}" fill="#999" />\n'
        chrom_bars += f'<text x="{x + bar_w/2}" y="{chart_h - 2}" text-anchor="middle" fill="#888" font-size="8">{chrom}</text>\n'
        chrom_bars += f'<text x="{x + bar_w/2}" y="{y - 3:.0f}" text-anchor="middle" fill="#aaa" font-size="7">{count:,}</text>\n'

    chrom_svg_w = 40 + len(sorted_chroms) * (bar_w + 4) + 20
    chrom_svg = f"""<svg width="{chrom_svg_w}" height="{chart_h}" xmlns="http://www.w3.org/2000/svg" style="font-family: 'JetBrains Mono', monospace;">
  <text x="40" y="14" fill="#555" font-size="10" font-weight="500">Genes per chromosome</text>
  {chrom_bars}
</svg>"""

    # Assembly info
    assembly = genes[0].get("assembly", "unknown") if genes else "unknown"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gene Statistics — {assembly}</title>
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
    padding: 4px 6px 4px 0;
  }}
  th.num {{ text-align: right; }}
  td {{ padding: 3px 6px 3px 0; vertical-align: top; border-bottom: 1px solid #f0f0f0; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.total {{ border-top: 1px solid #999; font-weight: 500; }}
  tr.total td {{ border-bottom: none; padding-top: 6px; }}
  .chart {{ margin: 12px 0; overflow-x: auto; }}
  .hist-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 8px 0;
  }}
  .hist-cell {{ overflow-x: auto; }}
  @media print {{
    body {{ font-size: 10px; max-width: 100%; margin: 20px; }}
  }}
</style>
</head>
<body>

<h1>Gene Statistics — {assembly}</h1>
<div class="meta">{len(genes):,} genes &middot; {len(by_biotype)} biotypes &middot; {fmt_bp(sum(all_sizes))} total gene body</div>

<details open>
<summary>Size distribution by biotype</summary>
<table>
  <tr>
    <th>Biotype</th>
    <th class="num">Count</th>
    <th class="num">Min</th>
    <th class="num">P25</th>
    <th class="num">Median</th>
    <th class="num">Mean</th>
    <th class="num">P75</th>
    <th class="num">P95</th>
    <th class="num">Max</th>
    <th class="num">Total</th>
  </tr>
  {stat_rows}
</table>
</details>

<details open>
<summary>Size histograms (log scale)</summary>
<div class="chart">{all_hist}</div>
{biotype_grid}
</details>

<details open>
<summary>Chromosome distribution</summary>
<div class="chart">{chrom_svg}</div>
</details>

</body>
</html>"""


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "genes_human.json"

    with open(input_file) as f:
        genes = json.load(f)

    html = generate_html(genes)
    output = input_file.replace(".json", "_report.html")

    with open(output, "w") as f:
        f.write(html)

    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
