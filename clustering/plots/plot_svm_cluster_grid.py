"""
Merged 2×3 grid PDF:
  rows    → strand orientation (top) | codon phase (bottom)
  columns → (a) global U1×U2  |  (b) left cluster  |  (c) right cluster
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

DATA_PATH   = "/fsx/dana_aubakirova/carbon_ablations/data/eukaryote/test_new.parquet"
OUT_DIR     = "/fsx/dana_aubakirova/carbon_ablations/clustering/output/carbon_test_new_3emb_v2_16k"
GREEN_DIR   = os.path.join(OUT_DIR, "green_plots")
SLICE_START = 96000 - 16384
NN = 100; SEED = 42

SPECIES_MAP = {
    "<fng>": "fungi", "<pln>": "plant", "<inv>": "invertebrate",
    "<prt>": "protozoa", "<vrt>": "vertebrate_other", "<mam>": "vertebrate_mammalian",
}

C_DARK_GREEN = "#0A6B2A"
C_BROWN      = "#6B3F18"
C_YELLOW     = "#74D9A0"   # lighter vivid green
C_INK        = "#1A1F18"
C_MUTED      = "#6C7066"
C_FAINT      = "#E4E2DA"
BG           = "#FFFFFF"

STRAND_CLR = {"forward (+)": C_DARK_GREEN, "reverse (-)": C_YELLOW}
PHASE_CLR  = {0: C_DARK_GREEN, 1: C_YELLOW, 2: C_BROWN}

COL_HEADERS = [
    "(a)  UMAP 1 × UMAP 2 — global",
    "(b)  Left cluster  (n = 21,608)",
    "(c)  Right cluster  (n = 7,803)",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 24,
    "axes.titlesize": 26, "axes.labelsize": 24, "legend.fontsize": 22,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "xtick.color": C_INK, "ytick.color": C_INK,
})

def style_ax(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_FAINT); ax.spines["bottom"].set_color(C_FAINT)
    ax.set_xticks([]); ax.set_yticks([])
    ax.yaxis.label.set_visible(True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading…")
df = pd.read_parquet(DATA_PATH)
df["species_name"] = df["species_type"].map(SPECIES_MAP)
df["codon_phase"]  = (df["start"] - SLICE_START) % 3
df["strand_label"] = df["strand"].map({"<+>": "forward (+)", "<->": "reverse (-)"})
df = df[df["species_name"].notna()].reset_index(drop=True)

proj3d = np.load(os.path.join(OUT_DIR, f"content_umap3d_nn{NN}.npy"))
u1 = proj3d[:, 0]; u2 = proj3d[:, 1]
X  = np.column_stack([u1, u2])

# ── SVM split ─────────────────────────────────────────────────────────────────
print("SVM split…")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km = KMeans(n_clusters=2, n_init=30, random_state=SEED)
km_labels = km.fit_predict(X_scaled)
svm = SVC(kernel="linear", C=1.0, random_state=SEED)
svm.fit(X_scaled, km_labels)
w = svm.coef_[0]; b = svm.intercept_[0]
svm_labels = svm.predict(X_scaled)
if u1[svm_labels == 0].mean() > u1[svm_labels == 1].mean():
    svm_labels = 1 - svm_labels
df["cluster"] = np.where(svm_labels == 0, "left", "right")
left  = df[df["cluster"] == "left"]
right = df[df["cluster"] == "right"]

# SVM boundary for global plot
u1_range = np.linspace(u1.min() - 0.5, u1.max() + 0.5, 400)
u1_s = (u1_range - scaler.mean_[0]) / scaler.scale_[0]
def boundary_u2(u1_s_vals, offset=0):
    u2_s = -(w[0] * u1_s_vals + b + offset) / w[1]
    return u2_s * scaler.scale_[1] + scaler.mean_[1]

# ── Load cached round-2 UMAPs ─────────────────────────────────────────────────
proj_left  = np.load(os.path.join(OUT_DIR, "left_cluster_umap2d_svm2d_u12.npy"))
proj_right = np.load(os.path.join(OUT_DIR, "right_cluster_umap2d_svm2d_u12.npy"))

kw = dict(s=5, alpha=0.7, linewidths=0, rasterized=True)
leg_kw      = dict(fontsize=22, markerscale=3, frameon=True,
                   facecolor=BG, edgecolor="none", framealpha=0.85,
                   loc="upper right", borderpad=0.3, handletextpad=0.4)
leg_kw_row2 = dict(fontsize=22, markerscale=3, frameon=True,
                   facecolor=BG, edgecolor="none", framealpha=0.85,
                   bbox_to_anchor=(0.99, 1.16), loc="upper right",
                   borderpad=0.3, handletextpad=0.4)

# ── Build 2×3 figure ──────────────────────────────────────────────────────────
print("Building figure…")
fig = plt.figure(figsize=(36, 20), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.08)

# Column data: (x, y, subset_df)
cols = [
    (u1,                  u2,                  df),
    (proj_left[:, 0],     proj_left[:, 1],     left),
    (proj_right[:, 0],    proj_right[:, 1],    right),
]

for col_idx, (xa, ya, sub) in enumerate(cols):
    sl = sub["strand_label"].values
    cp = sub["codon_phase"].values

    # ── Row 0: strand ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, col_idx])
    for strand, c in STRAND_CLR.items():
        m = sl == strand
        lbl = f"{strand}  ({m.mean():.1%})" if col_idx > 0 else strand
        ax.scatter(xa[m], ya[m], c=c, label=lbl, **kw)
    if col_idx == 0:
        ax.plot(u1_range, boundary_u2(u1_s), color=C_INK, lw=1.5, ls="--", alpha=0.6)
        ax.set_xlim(u1.min()-0.3, u1.max()+0.3)
        ax.set_ylim(u2.min()-0.3, u2.max()+0.3)
    ax.set_title(COL_HEADERS[col_idx], fontsize=26, fontweight="bold", pad=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(**leg_kw)
    style_ax(ax)
    if col_idx == 0:
        ax.set_ylabel("UMAP 2", fontsize=22, labelpad=10)

    # ── Row 1: codon phase ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, col_idx])
    for ph, lbl_base in [(0, "phase 0  (codon-aligned)"), (1, "phase +1"), (2, "phase +2")]:
        m = cp == ph
        lbl = f"{lbl_base}  ({m.mean():.1%})" if col_idx > 0 else lbl_base
        ax.scatter(xa[m], ya[m], c=PHASE_CLR[ph], label=lbl, **kw)
    if col_idx == 0:
        ax.plot(u1_range, boundary_u2(u1_s), color=C_INK, lw=1.5, ls="--", alpha=0.6)
        ax.set_xlim(u1.min()-0.3, u1.max()+0.3)
        ax.set_ylim(u2.min()-0.3, u2.max()+0.3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(**leg_kw_row2)
    style_ax(ax)
    if col_idx == 0:
        ax.set_ylabel("UMAP 2", fontsize=22, labelpad=10)

out = os.path.join(GREEN_DIR, "merged_svm_grid.pdf")
fig.tight_layout()
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"✓ {out}")
