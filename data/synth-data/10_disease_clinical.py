#!/usr/bin/env python3
"""
Disease & Clinical Annotations
================================
Annotates: pathogenic variants, disease associations, GWAS hits.

Data sources:
  - Ensembl REST API — clinical variants, phenotype associations
  - Open Targets GraphQL API — disease-gene associations
  - EBI GWAS Catalog REST API — GWAS associations

Produces annotations with category="disease_clinical".
"""

import requests
from common import (
    GenomicRegion, Annotation, AnnotatedSequence, DEFAULT_REGION,
    ENSEMBL_REST, make_annotated_sequence, clamp_to_region, ensembl_get,
)

OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"


def annotate(region: GenomicRegion) -> AnnotatedSequence:
    """Fetch sequence + disease/clinical annotations for a genomic region."""
    result = make_annotated_sequence(region)

    # --- 1. Ensembl clinical variants ---
    url = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
    params = {"feature": "variation", "content-type": "application/json"}
    resp = ensembl_get(url, params=params)
    resp.raise_for_status()

    for var in resp.json():
        clin_sig = var.get("clinical_significance", [])
        if not clin_sig:
            continue

        local_start, local_end = clamp_to_region(var["start"], var["end"], region)
        if local_start == 0:
            continue

        result.annotations.append(Annotation(
            start=local_start, end=local_end,
            type="clinical_variant", category="disease_clinical",
            label=f"{var.get('id')} ({', '.join(clin_sig)})",
            metadata={
                "rs_id": var.get("id"),
                "alleles": var.get("alleles"),
                "consequence": var.get("consequence_type"),
                "clinical_significance": clin_sig,
            },
        ))

    # --- 2. Ensembl phenotypes for overlapping genes ---
    gene_url = f"{ENSEMBL_REST}/overlap/region/human/{region.ensembl_chrom}:{region.start}-{region.end}"
    gene_params = {"feature": "gene", "content-type": "application/json"}
    try:
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

            # Phenotypes from Ensembl
            try:
                pheno_url = f"{ENSEMBL_REST}/phenotype/gene/human/{gene_name}"
                pheno_params = {"content-type": "application/json"}
                pheno_resp = ensembl_get(pheno_url, params=pheno_params)
                if pheno_resp.ok:
                    phenotypes = set()
                    for entry in pheno_resp.json()[:20]:
                        desc = entry.get("description", "")
                        if desc and desc not in phenotypes:
                            phenotypes.add(desc)

                    if phenotypes:
                        result.annotations.append(Annotation(
                            start=local_start, end=local_end,
                            type="disease_association", category="disease_clinical",
                            label=f"{gene_name}: {len(phenotypes)} disease associations",
                            metadata={
                                "gene": gene_name,
                                "phenotypes": list(phenotypes)[:10],
                                "total_phenotypes": len(phenotypes),
                            },
                        ))
            except Exception:
                pass

            # Open Targets disease associations
            try:
                gene_id = gene.get("id", "")
                ot_query = """
                query ($ensemblId: String!) {
                  target(ensemblId: $ensemblId) {
                    associatedDiseases(page: {index: 0, size: 5}) {
                      count
                      rows { disease { name id } score }
                    }
                  }
                }"""
                ot_resp = requests.post(OPENTARGETS_API,
                    json={"query": ot_query, "variables": {"ensemblId": gene_id}},
                    timeout=15)
                if ot_resp.ok:
                    ot_data = ot_resp.json()
                    assoc = ot_data.get("data", {}).get("target", {}).get("associatedDiseases", {})
                    diseases = []
                    for row in assoc.get("rows", []):
                        d = row.get("disease", {})
                        diseases.append({
                            "name": d.get("name"),
                            "score": round(row.get("score", 0), 3),
                        })

                    if diseases:
                        result.annotations.append(Annotation(
                            start=local_start, end=local_end,
                            type="opentargets_association", category="disease_clinical",
                            label=f"{gene_name} Open Targets: {diseases[0]['name']} "
                                  f"(score={diseases[0]['score']})",
                            score=diseases[0]["score"],
                            metadata={
                                "gene": gene_name,
                                "total_associations": assoc.get("count", 0),
                                "top_diseases": diseases,
                            },
                        ))
            except Exception:
                pass
    except Exception:
        pass

    return result


def main():
    print("Fetching disease/clinical annotations...")
    result = annotate(DEFAULT_REGION)
    print(result.summary())
    print(f"\nSequence (first 80bp): {result.sequence[:80]}...")
    with open("10_output.json", "w") as f:
        f.write(result.to_json())
    print("Saved to 10_output.json")


if __name__ == "__main__":
    main()
