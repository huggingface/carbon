"""
Generate standalone interactive HTML demo for Carbon-3B embedding analysis.

Three-panel layout (mirrors merged_svm_grid):
  (a) Global UMAP U1×U2 with SVM boundary
  (b) Left cluster round-2 UMAP
  (c) Right cluster round-2 UMAP

Buttons switch coloring across all panels: Strand | Codon phase | Species
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

DATA_PATH   = "/fsx/dana_aubakirova/carbon_ablations/data/eukaryote/test_new.parquet"
OUT_DIR     = "/fsx/dana_aubakirova/carbon_ablations/clustering/output/carbon_test_new_3emb_v2_16k"
OUT_HTML    = "/fsx/dana_aubakirova/carbon_pr/clustering/figures/demo.html"
SLICE_START = 96000 - 16384
NN = 100; SEED = 42

SPECIES_MAP = {
    "<fng>": "fungi", "<pln>": "plant", "<inv>": "invertebrate",
    "<prt>": "protozoa", "<vrt>": "vertebrate_other", "<mam>": "vertebrate_mammalian",
}

C_DARK_GREEN  = "#0A6B2A"
C_LIGHT_GREEN = "#74D9A0"
C_BROWN       = "#6B3F18"
C_INK         = "#1A1F18"
C_FAINT       = "#E4E2DA"

STRAND_CLR = {"forward (+)": C_DARK_GREEN, "reverse (-)": C_LIGHT_GREEN}
PHASE_CLR  = {0: C_DARK_GREEN, 1: C_LIGHT_GREEN, 2: C_BROWN}
PHASE_LABELS = {0: "phase 0 (codon-aligned)", 1: "phase +1", 2: "phase +2"}
SP_COLORS = {
    "plant":                "#1A7A40",
    "vertebrate_other":     "#6DBF7E",
    "fungi":                "#8C7355",
    "invertebrate":         "#C8BC99",
    "protozoa":             "#D4874A",
    "vertebrate_mammalian": "#F9C74F",
}

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data…")
df = pd.read_parquet(DATA_PATH)
df["species_name"] = df["species_type"].map(SPECIES_MAP)
df["codon_phase"]  = (df["start"] - SLICE_START) % 3
df["strand_label"] = df["strand"].map({"<+>": "forward (+)", "<->": "reverse (-)"})
df = df[df["species_name"].notna()].reset_index(drop=True)

proj3d = np.load(os.path.join(OUT_DIR, f"content_umap3d_nn{NN}.npy"))
u1 = proj3d[:, 0]; u2 = proj3d[:, 1]

proj_left  = np.load(os.path.join(OUT_DIR, "left_cluster_umap2d_svm2d_u12.npy"))
proj_right = np.load(os.path.join(OUT_DIR, "right_cluster_umap2d_svm2d_u12.npy"))

# ── SVM split ─────────────────────────────────────────────────────────────────
print("Computing SVM split…")
X = np.column_stack([u1, u2])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km = KMeans(n_clusters=2, n_init=30, random_state=SEED)
svm = SVC(kernel="linear", C=1.0, random_state=SEED)
svm.fit(X_scaled, km.fit_predict(X_scaled))
w = svm.coef_[0]; b = svm.intercept_[0]
svm_labels = svm.predict(X_scaled)
if u1[svm_labels == 0].mean() > u1[svm_labels == 1].mean():
    svm_labels = 1 - svm_labels
df["cluster"] = np.where(svm_labels == 0, "left", "right")
left  = df[df["cluster"] == "left"]
right = df[df["cluster"] == "right"]
n_L, n_R = len(left), len(right)
print(f"  Left: {n_L:,}  Right: {n_R:,}")

# SVM boundary
u1_range = np.linspace(u1.min() - 0.5, u1.max() + 0.5, 400)
u1_s     = (u1_range - scaler.mean_[0]) / scaler.scale_[0]
u2_s     = -(w[0] * u1_s + b) / w[1]
u2_bnd   = u2_s * scaler.scale_[1] + scaler.mean_[1]

# ── Build figure ──────────────────────────────────────────────────────────────
print("Building figure…")

kw = dict(mode="markers", marker=dict(size=2.5, opacity=0.65), hovertemplate="%{text}<extra></extra>")

def hover_text(sub, view_xa, view_ya):
    texts = []
    for i in range(len(sub)):
        row = sub.iloc[i]
        texts.append(
            f"<b>{row['species_name']}</b><br>"
            f"Strand: {row['strand_label']}<br>"
            f"Phase: {int(row['codon_phase'])}"
        )
    return texts

# Collect all traces grouped by color_by
# Each group: (strand | phase | species) → 3 panels × N labels = 3N traces
# We'll build all traces and toggle visibility with buttons

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        f"(a)  Global UMAP 1 × UMAP 2",
        f"(b)  Left cluster  (n = {n_L:,})",
        f"(c)  Right cluster  (n = {n_R:,})",
    ],
    horizontal_spacing=0.06,
)

traces_by_group = {"strand": [], "phase": [], "species": []}

# ── Strand traces ─────────────────────────────────────────────────────────────
for col_idx, (xa, ya, sub) in enumerate([
    (u1, u2, df),
    (proj_left[:, 0],  proj_left[:, 1],  left),
    (proj_right[:, 0], proj_right[:, 1], right),
], start=1):
    first_col = (col_idx == 1)
    for strand, color in STRAND_CLR.items():
        mask = sub["strand_label"].values == strand
        pct  = mask.mean() * 100
        lbl  = f"{strand}  ({pct:.1f}%)" if not first_col else strand
        t = go.Scatter(
            x=xa[mask], y=ya[mask],
            name=lbl, legendgroup=strand, showlegend=first_col,
            marker=dict(size=2.5, color=color, opacity=0.65),
            mode="markers", visible=True,
            text=[f"<b>{sub['species_name'].iloc[i]}</b><br>Strand: {strand}<br>Phase: {int(sub['codon_phase'].iloc[i])}"
                  for i in np.where(mask)[0]],
            hovertemplate="%{text}<extra></extra>",
        )
        fig.add_trace(t, row=1, col=col_idx)
        traces_by_group["strand"].append(len(fig.data) - 1)

# ── Phase traces ──────────────────────────────────────────────────────────────
for col_idx, (xa, ya, sub) in enumerate([
    (u1, u2, df),
    (proj_left[:, 0],  proj_left[:, 1],  left),
    (proj_right[:, 0], proj_right[:, 1], right),
], start=1):
    first_col = (col_idx == 1)
    for ph, color in PHASE_CLR.items():
        mask = sub["codon_phase"].values == ph
        pct  = mask.mean() * 100
        lbl  = f"{PHASE_LABELS[ph]}  ({pct:.1f}%)" if not first_col else PHASE_LABELS[ph]
        t = go.Scatter(
            x=xa[mask], y=ya[mask],
            name=lbl, legendgroup=PHASE_LABELS[ph], showlegend=first_col,
            marker=dict(size=2.5, color=color, opacity=0.65),
            mode="markers", visible=False,
            text=[f"<b>{sub['species_name'].iloc[i]}</b><br>Strand: {sub['strand_label'].iloc[i]}<br>Phase: {ph}"
                  for i in np.where(mask)[0]],
            hovertemplate="%{text}<extra></extra>",
        )
        fig.add_trace(t, row=1, col=col_idx)
        traces_by_group["phase"].append(len(fig.data) - 1)

# ── Species traces ────────────────────────────────────────────────────────────
for col_idx, (xa, ya, sub) in enumerate([
    (u1, u2, df),
    (proj_left[:, 0],  proj_left[:, 1],  left),
    (proj_right[:, 0], proj_right[:, 1], right),
], start=1):
    first_col = (col_idx == 1)
    for sp, color in SP_COLORS.items():
        mask = sub["species_name"].values == sp
        pct  = mask.mean() * 100
        lbl  = f"{sp}  ({pct:.1f}%)" if not first_col else sp
        t = go.Scatter(
            x=xa[mask], y=ya[mask],
            name=lbl, legendgroup=sp, showlegend=first_col,
            marker=dict(size=2.5, color=color, opacity=0.65),
            mode="markers", visible=False,
            text=[f"<b>{sp}</b><br>Strand: {sub['strand_label'].iloc[i]}<br>Phase: {int(sub['codon_phase'].iloc[i])}"
                  for i in np.where(mask)[0]],
            hovertemplate="%{text}<extra></extra>",
        )
        fig.add_trace(t, row=1, col=col_idx)
        traces_by_group["species"].append(len(fig.data) - 1)

# ── SVM boundary ──────────────────────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=u1_range, y=u2_bnd,
    mode="lines",
    line=dict(color=C_INK, width=1.5, dash="dash"),
    name="SVM boundary", showlegend=True,
    visible=True, hoverinfo="skip",
), row=1, col=1)
boundary_idx = len(fig.data) - 1

n_total = len(fig.data)

def vis_array(group):
    arr = [False] * n_total
    for i in traces_by_group[group]:
        arr[i] = True
    arr[boundary_idx] = True  # always show boundary
    return arr

# ── Layout + buttons ──────────────────────────────────────────────────────────
fig.update_layout(
    height=560, width=1300,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="DejaVu Sans, Arial, sans-serif", size=14, color=C_INK),
    title=dict(
        text="Carbon-3B · Content-token embeddings · 2D linear SVM split",
        font=dict(size=20), x=0.5, xanchor="center",
    ),
    legend=dict(
        font=dict(size=12), itemsizing="constant",
        bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0)",
        tracegroupgap=4,
    ),
    updatemenus=[dict(
        type="buttons",
        direction="right",
        buttons=[
            dict(label="Strand orientation", method="update",
                 args=[{"visible": vis_array("strand")}]),
            dict(label="Codon phase", method="update",
                 args=[{"visible": vis_array("phase")}]),
            dict(label="Species", method="update",
                 args=[{"visible": vis_array("species")}]),
        ],
        x=0.5, xanchor="center", y=1.13,
        bgcolor="#F5F5F5", bordercolor="#CCCCCC",
        font=dict(size=14),
        showactive=True,
        active=0,
    )],
    margin=dict(t=110, l=50, r=50, b=50),
    annotations=[
        dict(text="Color by:", x=0.27, y=1.115, xref="paper", yref="paper",
             showarrow=False, font=dict(size=14)),
    ],
)

for col in [1, 2, 3]:
    fig.update_xaxes(title_text="UMAP 1", showticklabels=False,
                     showgrid=False, zeroline=False,
                     linecolor=C_FAINT, mirror=True, row=1, col=col)
    fig.update_yaxes(title_text="UMAP 2", showticklabels=False,
                     showgrid=False, zeroline=False,
                     linecolor=C_FAINT, mirror=True, row=1, col=col)

fig.write_html(OUT_HTML, include_plotlyjs=True, full_html=True)
print(f"✓ Saved → {OUT_HTML}")
