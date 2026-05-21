"""
Generate Carbon-3B 3D UMAP Hero HTML using the existing Hero template.
Replaces the embedded data with separator embeddings from carbon_genstyle_48k_full.
Subsamples to 500 per species (3,000 total) to match the original Hero.
"""
import re
import json
import numpy as np
import pandas as pd

UMAP_CSV  = "/fsx/dana_aubakirova/carbon_ablations/clustering/output/carbon_genstyle_48k_full/last_token_umap3d.csv"
HERO_IN   = "/fsx/dana_aubakirova/carbon_ablations/UMAP 3D Hero.html"
HERO_OUT  = "/fsx/dana_aubakirova/carbon_pr/clustering/figures/carbon_3b_embedding_hero.html"
N_PER_SP  = 500
SEED      = 42

# ── Load & subsample ──────────────────────────────────────────────────────────
df = pd.read_csv(UMAP_CSV)
types_order = ["fungi", "invertebrate", "plant", "protozoa",
               "vertebrate_mammalian", "vertebrate_other"]
type2idx = {t: i for i, t in enumerate(types_order)}

rng = np.random.default_rng(SEED)
parts = []
for sp in types_order:
    sub = df[df["type"] == sp]
    idx = rng.choice(len(sub), size=min(N_PER_SP, len(sub)), replace=False)
    parts.append(sub.iloc[idx])
df_sub = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Build points list: [type_idx, x, y, z] rounded to 4dp
points = [
    [type2idx[row.type],
     round(float(row.x), 4),
     round(float(row.y), 4),
     round(float(row.z), 4)]
    for _, row in df_sub.iterrows()
]
data_json = json.dumps({"types": types_order, "points": points}, separators=(",", ":"))
print(f"Points: {len(points)}  |  JSON size: {len(data_json)/1024:.1f} KB")

# ── Patch Hero HTML ───────────────────────────────────────────────────────────
with open(HERO_IN) as f:
    html = f.read()

# Replace data blob
html = re.sub(
    r"const data = await Promise\.resolve\(\{.*?\}\);",
    f"const data = await Promise.resolve({data_json});",
    html, flags=re.DOTALL,
)

# Update title and subtitle
html = html.replace(
    "<title>Last-token embedding · UMAP 3D</title>",
    "<title>Carbon-3B · Separator embedding · UMAP 3D</title>",
)
html = html.replace(
    "Last-token <em>embedding</em><br/>space",
    "Carbon-3B <em>separator</em><br/>embeddings",
)
html = html.replace(
    "UMAP projection of 3,000 species into 3 dimensions, colored by NCBI taxonomy. Drag to rotate · click a clade in the phylogeny to focus.",
    "UMAP projection of 3,000 sequences at 48K bp context, colored by taxonomic group. "
    "Separator embeddings (<code>&lt;/dna&gt;</code> token) from Carbon-3B. "
    "Drag to rotate · click a clade to focus.",
)

with open(HERO_OUT, "w") as f:
    f.write(html)

print(f"✓ Saved → {HERO_OUT}")
