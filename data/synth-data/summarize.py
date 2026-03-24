#!/usr/bin/env python3
"""
Produce a concise version of the raw annotations.

Reads combined_output.json (or any AnnotatedSequence JSON) and outputs
a compact version that filters out noise and deduplicates, suitable for
feeding to an LLM for natural language description.

Rules:
  - Drop annotations that are just regex matches with no database backing
    (raw start/stop codons, GC-content windows)
  - Drop duplicate spans (e.g., same variant in both 'variants' and 'disease_clinical')
  - Collapse many-of-the-same-type into a summary count + top examples
  - Keep all annotations that carry real signal (clinical significance,
    conserved elements, named features, expression data, etc.)

Usage:
    python summarize.py                         # reads combined_output.json
    python summarize.py my_annotations.json     # reads custom file
"""

import json
import sys
from collections import Counter


# Types that are just regex/sliding-window noise — not database-backed
NOISE_TYPES = {
    "start_codon",      # every ATG in the sequence
    "stop_codon",       # every TAA/TAG/TGA
    "gc_content",       # sliding window, one per 25bp
    "simple_repeat",    # short homopolymers (CCC, TTT)
    "stem_loop",        # predicted hairpins — mostly false positives
    "Initiator",        # regex motif match
    "DPE",              # regex motif match
}

# Max annotations to keep per (category, type) before summarizing
MAX_PER_TYPE = 5


def summarize(data: dict) -> dict:
    """
    Filter and deduplicate annotations. Returns a new AnnotatedSequence dict
    with a compact annotations list.
    """
    raw = data["annotations"]

    # --- Step 1: Remove pure noise types ---
    filtered = [a for a in raw if a["type"] not in NOISE_TYPES]

    # --- Step 2: Deduplicate by (start, end, type, key identifier) ---
    # e.g., same rs_id appearing in both 'variants' and 'disease_clinical'
    seen = set()
    deduped = []
    for a in filtered:
        # Build a dedup key: use rs_id if present, else (start, end, type)
        rs_id = a.get("metadata", {}).get("rs_id")
        if rs_id:
            key = (a["start"], a["end"], rs_id)
        else:
            key = (a["start"], a["end"], a["type"], a["label"])

        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)

    # --- Step 3: Collapse types with too many entries ---
    # Group by (category, type)
    groups = {}
    for a in deduped:
        group_key = (a["category"], a["type"])
        groups.setdefault(group_key, []).append(a)

    concise = []
    summaries = []

    for (category, atype), items in groups.items():
        if len(items) <= MAX_PER_TYPE:
            concise.extend(items)
        else:
            # Keep the most interesting ones, summarize the rest
            # Sort by: clinical significance first, then by score, then by position
            def sort_key(a):
                has_clinical = bool(a.get("metadata", {}).get("clinical_significance"))
                score = a.get("score") or 0
                return (-int(has_clinical), -abs(score), a["start"])

            items.sort(key=sort_key)
            concise.extend(items[:MAX_PER_TYPE])

            # Build summary of what was collapsed
            summary_meta = {"total_count": len(items), "kept": MAX_PER_TYPE}

            # Add useful aggregate stats depending on category
            if category == "variants":
                clin_counts = Counter()
                consequence_counts = Counter()
                for a in items:
                    for cs in a.get("metadata", {}).get("clinical_significance", []):
                        clin_counts[cs] += 1
                    cons = a.get("metadata", {}).get("consequence", "")
                    if cons:
                        consequence_counts[cons] += 1
                if clin_counts:
                    summary_meta["clinical_significance_counts"] = dict(clin_counts.most_common())
                if consequence_counts:
                    summary_meta["consequence_counts"] = dict(consequence_counts.most_common(5))

            elif category == "disease_clinical":
                sig_counts = Counter()
                for a in items:
                    for cs in a.get("metadata", {}).get("clinical_significance", []):
                        sig_counts[cs] += 1
                if sig_counts:
                    summary_meta["clinical_significance_counts"] = dict(sig_counts.most_common())

            elif category == "gene_structure":
                type_counts = Counter(a["type"] for a in items)
                summary_meta["feature_counts"] = dict(type_counts)

            summaries.append({
                "start": 1,
                "end": data["region"]["end"] - data["region"]["start"] + 1,
                "type": f"{atype}_summary",
                "category": category,
                "label": f"{len(items)} {atype} annotations (showing top {MAX_PER_TYPE})",
                "strand": ".",
                "score": None,
                "metadata": summary_meta,
            })

    concise.extend(summaries)

    # --- Step 4: Sort by position ---
    concise.sort(key=lambda a: (a["start"], a["end"]))

    return {
        "sequence": data["sequence"],
        "region": data["region"],
        "annotations": concise,
        "_summary": {
            "raw_annotation_count": len(raw),
            "concise_annotation_count": len(concise),
            "reduction": f"{(1 - len(concise) / len(raw)) * 100:.0f}%",
        },
    }


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "combined_output.json"

    with open(input_file) as f:
        data = json.load(f)

    result = summarize(data)

    output_file = input_file.replace(".json", "_concise.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    s = result["_summary"]
    print(f"Raw: {s['raw_annotation_count']} -> Concise: {s['concise_annotation_count']} ({s['reduction']} reduction)")
    print(f"Saved to {output_file}")

    # Print concise summary
    by_cat = {}
    for a in result["annotations"]:
        by_cat.setdefault(a["category"], []).append(a)
    for cat, anns in by_cat.items():
        print(f"  {cat}: {len(anns)}")
        for a in anns[:3]:
            print(f"    {a['start']}-{a['end']} {a['type']}: {a['label']}")
        if len(anns) > 3:
            print(f"    ... and {len(anns) - 3} more")


if __name__ == "__main__":
    main()
