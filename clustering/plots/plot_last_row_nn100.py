"""
Regenerate the n_neighbors=100 row of the context-length ablation
as a standalone 1×3 figure (8K / 16K / 48K bp).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap as umap_lib

BASE   = "/fsx/dana_aubakirova/carbon_ablations/clustering/output"
OUT    = f"{BASE}/carbon_last_token_nn100_row.pdf"
NN     = 100
SEED   = 42

CONTEXTS = [
    ("8K bp",  f"{BASE}/carbon_genstyle_8k_full"),
    ("16K bp", f"{BASE}/carbon_genstyle_16k_full"),
    ("48K bp", f"{BASE}/carbon_genstyle_48k_full"),
]

SPECIES_LIST = ["fungi", "invertebrate", "plant", "protozoa",
                "vertebrate_mammalian", "vertebrate_other"]

# barplot.py palette extended to 6 distinctive earthy colors
SP_COLORS = {
    "plant":                "#1A7A40",   # dark forest green   (Carbon 8B)
    "vertebrate_other":     "#6DBF7E",   # bright medium green (Carbon 3B)
    "fungi":                "#8C7355",   # warm brown          (Evo2 7B)
    "invertebrate":         "#C8BC99",   # pale tan            (GENERator)
    "protozoa":             "#D4874A",   # amber / burnt sienna (palette extension)
    "vertebrate_mammalian": "#F9C74F",   # yellow
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        24,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.titlesize":   26,
    "axes.titleweight": "bold",
})

fig, axes = plt.subplots(1, 3, figsize=(26, 9), facecolor="white")
kw = dict(s=8, alpha=0.6, linewidths=0, rasterized=True)

for ax, (ctx_label, ctx_dir) in zip(axes, CONTEXTS):
    print(f"  {ctx_label}: loading embeddings…")
    embs = np.load(f"{ctx_dir}/last_token_embeddings.npy")
    df   = pd.read_parquet(f"{ctx_dir}/df.parquet")

    cache = f"{ctx_dir}/last_token_umap2d_nn{NN}.npy"
    if os.path.exists(cache):
        print(f"  {ctx_label}: loading cached UMAP…")
        proj = np.load(cache)
    else:
        print(f"  {ctx_label}: running UMAP (n_neighbors={NN})…")
        proj = umap_lib.UMAP(
            n_components=2, n_neighbors=NN,
            metric="cosine", random_state=SEED,
        ).fit_transform(embs)
        np.save(cache, proj)
        print(f"  {ctx_label}: saved → {cache}")

    for sp in SPECIES_LIST:
        mask = df["type"].values == sp
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=SP_COLORS[sp], label=sp, **kw)

    ax.set_title(ctx_label, fontsize=24)
    ax.set_xlabel("UMAP 1", fontsize=22)
    ax.set_ylabel("UMAP 2", fontsize=22)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E4E2DA")
    ax.spines["bottom"].set_color("#E4E2DA")

fig.legend(
    *axes[0].get_legend_handles_labels(),
    fontsize=24, markerscale=4, frameon=False,
    loc="lower center", bbox_to_anchor=(0.5, -0.08),
    ncol=len(SPECIES_LIST),
)

fig.suptitle(
    f"Carbon 3B — `</dna>` last token  ·  context-length ablation  ·  n_neighbors = {NN}",
    fontsize=26, fontweight="bold", y=1.02, color="#1A1F18",
)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.close()
print(f"\n✓ Saved {OUT}")
