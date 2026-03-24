#!/usr/bin/env python3
"""
Generate two-fold description of an annotated DNA sequence using the Gemini API:

1. DESCRIPTION: External knowledge the model needs to understand this sequence
   (things not predictable from the raw bases alone — gene identity, clinical
   significance, disease associations, evolutionary context, expression patterns).

2. REASONING: A step-by-step reasoning trace that walks through the sequence,
   grounded in the annotations, as if a model were analyzing the DNA from scratch.

Usage:
    export GEMINI_API_KEY=your_key_here
    python describe.py                                  # uses combined_output_concise.json
    python describe.py my_annotations_concise.json      # custom file
"""

import json
import os
import sys
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


# ---------------------------------------------------------------------------
# Prompts — edit these to control the output style
# ---------------------------------------------------------------------------

DESCRIPTION_SYSTEM = """\
You are a genomics expert. You are given a DNA sequence and its structured \
annotations from multiple databases.

Your task: write only the ESSENTIAL external facts a model would need as context \
to understand this DNA snippet. Think of it as a minimal "answer key" — just the \
ground truth that cannot be predicted from the nucleotides alone.

Keep it short and factual. A few sentences at most. Only include:
- The gene name and its core function (one sentence)
- The most important clinical fact (e.g., "pathogenic variants here cause X disease")
- Any other single critical fact (e.g., tissue expression, conservation) \
  ONLY if it's essential context

Do NOT write an essay. Do NOT list all annotations. Do NOT explain biology. \
Just the bare facts a model needs as a label.\
"""

REASONING_SYSTEM = """\
You are a genomics expert. You are given a DNA sequence and its structured \
annotations from multiple databases.

Your task: write a REASONING TRACE as if you are a model that has been given \
ONLY the raw DNA sequence and must figure out what it contains. You should \
"discover" the biological content step by step by examining the sequence.

Critical rules:
- NEVER reference annotations, databases, annotation IDs, rs numbers, or \
  Ensembl/ENCODE accessions. Pretend you don't have them.
- Instead, reason FROM the sequence: "I notice an ATG at position X followed \
  by an open reading frame...", "The GT-AG dinucleotides here suggest splice \
  sites...", "This region has high GC content which could indicate..."
- Use the annotations only to know WHAT to find — but express it as if you're \
  discovering it from the sequence.
- Connect observations naturally: "Given that this appears to be a coding exon, \
  the high conservation I'd expect here makes sense..."

Write in first person as a continuous stream of thought. No section headers, \
no numbered steps, no "First pass" / "Zooming in" structure. Just a natural \
flow of reasoning that starts by noticing patterns in the sequence and \
gradually builds up to a complete understanding.

Be specific about positions and subsequences. This should read like a model \
that deeply understands DNA reasoning through what it sees.\
"""

USER_PROMPT_TEMPLATE = """\
Region: {chrom}:{start}-{end} ({assembly}), {length} bp

Sequence:
{sequence}

Annotations:
{annotations_json}
"""


def call_gemini(prompt: str, system: str, temperature: float = 0.3,
                max_tokens: int = 4096) -> str:
    """Call the Gemini API and return the text response."""
    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY environment variable")

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    resp = requests.post(
        GEMINI_URL,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        return f"No response from Gemini. Raw: {json.dumps(data)[:500]}"

    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "combined_output_concise.json"

    with open(input_file) as f:
        data = json.load(f)

    region = data["region"]
    user_prompt = USER_PROMPT_TEMPLATE.format(
        chrom=region.get("chrom", "?"),
        start=region.get("start", "?"),
        end=region.get("end", "?"),
        assembly=region.get("assembly", "?"),
        length=len(data.get("sequence", "")),
        sequence=data.get("sequence", ""),
        annotations_json=json.dumps(data["annotations"], indent=2),
    )

    print(f"Input: {len(data['annotations'])} annotations, {len(data.get('sequence', ''))} bp")
    print(f"Model: {GEMINI_MODEL}\n")

    # --- Fold 1: Description ---
    print("Generating description...", flush=True)
    description = call_gemini(user_prompt, DESCRIPTION_SYSTEM, temperature=0.3)

    # --- Fold 2: Reasoning ---
    print("Generating reasoning trace...", flush=True)
    reasoning = call_gemini(user_prompt, REASONING_SYSTEM, temperature=0.4, max_tokens=32768)

    # --- Output ---
    output = {
        "region": data["region"],
        "sequence": data["sequence"],
        "description": description,
        "reasoning": reasoning,
    }

    output_file = input_file.replace(".json", "_described.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print("DESCRIPTION")
    print("=" * 60)
    print(description)
    print(f"\n{'=' * 60}")
    print("REASONING")
    print("=" * 60)
    print(reasoning)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
