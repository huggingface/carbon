#!/usr/bin/env python3
"""
Expression & Epigenetic Annotations
====================================
Annotates: DNase hypersensitivity, histone peaks, gene expression.

Data sources:
  - ENCODE REST API — DNase-seq / ChIP-seq experiments (available for region)
  - Ensembl REST API — regulatory features
  - GTEx API — tissue-specific gene expression

Produces annotations with category="expression_epigenetic".
"""

import requests
from common import (
    GenomicRegion, Annotation, AnnotatedSequence, DEFAULT_REGION,
    ENSEMBL_REST, make_annotated_sequence, clamp_to_region, ensembl_get,
)

ENCODE_API = "https://www.encodeproject.org"
GTEX_API = "https://gtexportal.org/api/v2"


def annotate(region: GenomicRegion) -> AnnotatedSequence:
    """Fetch sequence + expression/epigenetic annotations for a genomic region."""
    result = make_annotated_sequence(region)

    # --- 1. Ensembl regulatory features (open chromatin, etc.) ---
    url = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
    params = {"feature": "regulatory", "content-type": "application/json"}
    try:
        resp = ensembl_get(url, params=params)
        resp.raise_for_status()
        for feat in resp.json():
            local_start, local_end = clamp_to_region(feat["start"], feat["end"], region)
            if local_start == 0:
                continue
            feat_type = feat.get("description", feat.get("feature_type", ""))
            result.annotations.append(Annotation(
                start=local_start, end=local_end,
                type=f"regulatory_{feat_type.replace(' ', '_').lower()}",
                category="expression_epigenetic",
                label=f"Regulatory feature: {feat_type} ({feat.get('id')})",
                metadata={"regulatory_id": feat.get("id"), "feature_type": feat_type},
            ))
    except Exception:
        pass

    # --- 2. ENCODE: available DNase-seq and ChIP-seq experiments ---
    encode_queries = [
        ("DNase-seq", "K562", None),
        ("Histone ChIP-seq", "K562", "H3K27ac"),
        ("Histone ChIP-seq", "K562", "H3K4me3"),
    ]
    for assay, biosample, target in encode_queries:
        try:
            search_params = {
                "type": "Experiment",
                "assay_title": assay,
                "biosample_ontology.term_name": biosample,
                "status": "released",
                "limit": 1,
            }
            if target:
                search_params["target.label"] = target
            headers = {"Accept": "application/json"}
            resp = requests.get(f"{ENCODE_API}/search/", params=search_params,
                                headers=headers, timeout=15)
            data = resp.json()
            if data.get("@graph"):
                exp = data["@graph"][0]
                label_parts = [assay]
                if target:
                    label_parts.append(target)
                label_parts.append(f"in {biosample}")
                result.annotations.append(Annotation(
                    start=1, end=result.length,
                    type=f"encode_{assay.replace(' ', '_').lower()}",
                    category="expression_epigenetic",
                    label=f"ENCODE {' '.join(label_parts)} ({exp.get('accession')})",
                    metadata={
                        "encode_accession": exp.get("accession"),
                        "assay": assay,
                        "target": target,
                        "biosample": biosample,
                    },
                ))
        except Exception:
            continue

    # --- 3. GTEx expression for overlapping genes ---
    try:
        gene_url = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
        gene_params = {"feature": "gene", "content-type": "application/json"}
        gene_resp = ensembl_get(gene_url, params=gene_params)
        gene_resp.raise_for_status()

        seen_genes = set()
        for gene in gene_resp.json():
            gene_name = gene.get("external_name", "")
            if not gene_name or gene_name in seen_genes:
                continue
            seen_genes.add(gene_name)

            local_start, local_end = clamp_to_region(gene["start"], gene["end"], region)
            if local_start == 0:
                continue

            # GTEx lookup
            gtex_url = f"{GTEX_API}/reference/gene"
            gtex_params = {"geneId": gene_name, "gencodeVersion": "v26",
                           "genomeBuild": "GRCh38/hg38"}
            gtex_resp = requests.get(gtex_url, params=gtex_params, timeout=15)
            if not gtex_resp.ok:
                continue
            genes = gtex_resp.json().get("data", [])
            if not genes:
                continue

            gencode_id = genes[0].get("gencodeId", "")
            expr_url = f"{GTEX_API}/expression/medianGeneExpression"
            expr_params = {"gencodeId": gencode_id, "datasetId": "gtex_v8"}
            expr_resp = requests.get(expr_url, params=expr_params, timeout=15)
            if not expr_resp.ok:
                continue

            tissues = {}
            for entry in expr_resp.json().get("data", []):
                tissue = entry.get("tissueSiteDetailId", "")
                tpm = entry.get("median", 0)
                if tpm > 0:
                    tissues[tissue] = round(tpm, 2)

            if tissues:
                sorted_tissues = dict(sorted(tissues.items(), key=lambda x: -x[1]))
                top5 = dict(list(sorted_tissues.items())[:5])
                result.annotations.append(Annotation(
                    start=local_start, end=local_end,
                    type="gene_expression", category="expression_epigenetic",
                    label=f"{gene_name} expression (top: {list(top5.keys())[0]} "
                          f"{list(top5.values())[0]} TPM)",
                    score=max(tissues.values()),
                    metadata={
                        "gene": gene_name,
                        "gencode_id": gencode_id,
                        "top_tissues": top5,
                        "num_tissues_expressed": len(tissues),
                    },
                ))
    except Exception:
        pass

    return result


def main():
    print("Fetching expression/epigenetic annotations...")
    result = annotate(DEFAULT_REGION)
    print(result.summary())
    print(f"\nSequence (first 80bp): {result.sequence[:80]}...")
    with open("07_output.json", "w") as f:
        f.write(result.to_json())
    print("Saved to 07_output.json")


if __name__ == "__main__":
    main()
