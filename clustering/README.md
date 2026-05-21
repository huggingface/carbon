# Carbon 3B — Embedding Representation Analysis

Carbon-3B learns structured internal representations of genomic sequence. Beyond likelihood and generation, we analyze whether the model organizes sequences according to biologically meaningful factors — taxonomic background, strand orientation, coding frame, and local sequence grammar — using training-free embedding probes.

---

## Embeddings

We compare two embeddings extracted from adjacent positions in the same input:

| Embedding | Token position | What it captures |
|---|---|---|
| **Separator** | `</dna>` (last token) | Genome-level and taxonomic structure |
| **Content-token** | Final DNA 6-mer (token before `</dna>`) | Local sequence state: strand orientation, codon phase, and an unresolved major axis |

Although these two hidden states differ by only one token position, they exhibit strikingly different geometries. This suggests that autoregressive DNA models internally organize biological information at multiple scales, with different token positions specializing according to their predictive role.

**Codon phase** is defined as the offset between the 6-mer tokenization boundary and the canonical codon frame: phase 0 is codon-aligned, phases +1 and +2 are shifted by one or two bases respectively.

---

## Data

| Dataset | N | Use |
|---|---|---|
| [GenerTeam/sequence-recovery](https://huggingface.co/datasets/GenerTeam/sequence-recovery) (held-out) | 30,000 | Separator embedding — taxonomy + context-length analysis |
| [GenerTeam/sequence-recovery · eukaryote/test_new.parquet](https://huggingface.co/datasets/GenerTeam/sequence-recovery/blob/main/eukaryote/test_new.parquet) (held-out) | 29,411 | Content-token embedding — strand + codon phase analysis |

Both datasets cover six taxonomic groups: fungi, plants, invertebrates, protozoa, vertebrate (other), vertebrate (mammalian).

The full eukaryotic set is constructed around centered CDS annotations, so strand orientation and codon phase are well defined at the embedding position. Sequences are 16,384 bp windows sliced as `seq[79616:96000]`.

Taxonomic structure in the separator embedding space cannot be explained by sequence homology: the sequences are randomly sampled held-out genomic windows, not aligned to homologous loci. The model must have learned distributed genome-level regularities — GC content, codon usage, repeat density, gene architecture — that recur across random genomic samples.

---

## Structure

```
carbon/clustering/
├── embedding_extraction/
│   ├── extract_content_token_embeddings.py   # content-token embedding @ 16K bp → content_embeddings.npy
│   └── extract_separator_embeddings.py       # separator embedding @ 8K/16K/48K → last_token_embeddings.npy
│
└── plots/
    ├── plot_content_token_umap_svm.py                     # 3D UMAP → 2D SVM on (U1,U2) → PDFs + round-2 caches
    ├── plot_svm_cluster_grid.py               # merged 2×3 grid — main figure
    ├── plot_cluster_stats.py               # strand / phase / species breakdown per cluster
    └── plot_context_length_ablation.py                # context-length ablation 1×3 row
```

---

## Pipelines

### Content-token embedding — SVM cluster analysis

```bash
python clustering/embedding_extraction/extract_content_token_embeddings.py
python clustering/plots/plot_content_token_umap_svm.py
python clustering/plots/plot_svm_cluster_grid.py
python clustering/plots/plot_cluster_stats.py
```

**Input:** `data/eukaryote/test_new.parquet`  
**Output:** `clustering/output/carbon_test_new_3emb_v2_16k/`

| File | Description |
|---|---|
| `content_embeddings.npy` | Hidden state at last DNA 6-mer, shape (29411, D) |
| `content_umap3d_nn100.npy` | 3D UMAP projection, cached (n_neighbors=100, cosine) |
| `{left,right}_cluster_umap2d_svm2d_u12.npy` | Round-2 UMAP per cluster, cached |
| `green_plots/umap12_clusters_overlay_svm2d.pdf` | Global (U1, U2) — clusters / strand / phase with SVM boundary |
| `green_plots/round2_left_umap12_svm2d.pdf` | Round-2 UMAP, left cluster (n=21,608) |
| `green_plots/round2_right_umap12_svm2d.pdf` | Round-2 UMAP, right cluster (n=7,803) |
| `green_plots/merged_svm_grid.pdf` | **Main figure** — 2×3 grid (strand + phase × global + left + right) |
| `green_plots/umap12_cluster_stats_svm2d.png` | Strand / phase / species breakdown per cluster |
| `green_plots/cluster_stats_{left,right}_svm2d_u12.png` | Per-cluster bar charts |

### Separator embedding — context-length ablation

```bash
python clustering/embedding_extraction/extract_separator_embeddings.py --max_length 8192  --out_dir clustering/output/carbon_genstyle_8k_full
python clustering/embedding_extraction/extract_separator_embeddings.py --max_length 16384 --out_dir clustering/output/carbon_genstyle_16k_full
python clustering/embedding_extraction/extract_separator_embeddings.py --max_length 49152 --out_dir clustering/output/carbon_genstyle_48k_full
python clustering/plots/plot_context_length_ablation.py
```

**Output:** `clustering/output/carbon_last_token_nn100_row.pdf`

---

## Results

### Separator embeddings — context-length ablation

All three metrics improve monotonically with context length. At 8k nucleotides the six taxonomic groups form partially overlapping clouds; at 48k nucleotides they resolve into more compact and distinct regions. Mammalian vertebrates form a compact isolated region while the remaining groups become more spatially distinct.

| Context length | KNN | ARI | NMI |
|---|---|---|---|
| 8 kb | 0.985 | 0.274 | 0.377 |
| 16 kb | 0.990 | 0.316 | 0.415 |
| **48 kb** | **0.995** | **0.380** | **0.474** |

![Context-length ablation](clustering/output/carbon_last_token_nn100_row.png)

---

### Content-token embeddings — SVM, cluster analysis

When projected with UMAP, content-token embeddings separate into two large clusters along the first UMAP dimension (U1). We fit a linear SVM in the (U1, U2) plane to divide the projection, then re-embed each region independently (round-2 UMAP).

**SVM method:**
1. Compute 3D UMAP (`n_neighbors=100`, cosine metric) on `content_embeddings.npy`
2. Extract U1, U2; StandardScale → KMeans(k=2) initialisation → fit LinearSVM(C=1.0)
3. Orient: lower mean U1 = **left** cluster, higher = **right** cluster
4. Re-run independent 2D UMAP on each cluster's raw embeddings (round-2)

**Split:** left 21,608 (73.5%) · right 7,803 (26.5%)

The first split is **not explained by strand orientation or codon phase** — both labels are mixed across the two major clusters. The dominant U1 axis captures an unresolved source of variation (possibly broad compositional, phylogenetic, or annotation-related factors). After separating the clusters with the SVM boundary, strand orientation and codon phase emerge as the dominant organizing axes within each cluster.

This result is notable because Carbon-3B uses non-overlapping 6-mer tokenization. The content-token geometry recovers the natural three-phase structure of coding sequences rather than a 6-periodic artifact, showing that 6-mer tokenization does not prevent the model from representing nucleotide-level reading-frame information at a resolution finer than the token unit.

#### Global U1×U2 — clusters / strand / phase
![umap12 overlay](clustering/output/carbon_test_new_3emb_v2_16k/green_plots/umap12_clusters_overlay_svm2d.png)

#### Round-2 UMAP — left cluster (n=21,608)
![round2 left](clustering/output/carbon_test_new_3emb_v2_16k/green_plots/round2_left_umap12_svm2d.png)

#### Round-2 UMAP — right cluster (n=7,803)
![round2 right](clustering/output/carbon_test_new_3emb_v2_16k/green_plots/round2_right_umap12_svm2d.png)

#### Cluster composition — strand / phase / species
![cluster stats](clustering/output/carbon_test_new_3emb_v2_16k/green_plots/umap12_cluster_stats_svm2d.png)
