"""Cluster stats for 2D SVM (U1×U2) split — grouped bars + per-cluster bars."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
SPECIES_LIST = list(SPECIES_MAP.values())

C_DARK_GREEN = "#1A7A40"; C_BROWN = "#8C7355"; C_YELLOW = "#F9C74F"
C_INK = "#1A1F18"; C_FAINT = "#E4E2DA"; BG = "#FFFFFF"

CLR_LEFT  = C_DARK_GREEN
CLR_RIGHT = C_YELLOW
STRAND_CLR = {"forward (+)": C_DARK_GREEN, "reverse (-)": C_YELLOW}
PHASE_CLR  = {0: C_DARK_GREEN, 1: C_YELLOW, 2: C_BROWN}

SP_COLORS = {
    "plant":                "#1A7A40",
    "vertebrate_other":     "#6DBF7E",
    "fungi":                "#8C7355",
    "invertebrate":         "#C8BC99",
    "protozoa":             "#D4874A",
    "vertebrate_mammalian": "#95D5B2",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "xtick.color": C_INK, "ytick.color": C_INK,
})

def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_FAINT); ax.spines["bottom"].set_color(C_FAINT)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=C_FAINT, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

# ── Load & split ──────────────────────────────────────────────────────────────
print("Loading…")
df = pd.read_parquet(DATA_PATH)
df["species_name"] = df["species_type"].map(SPECIES_MAP)
df["codon_phase"]  = (df["start"] - SLICE_START) % 3
df["strand_label"] = df["strand"].map({"<+>": "forward (+)", "<->": "reverse (-)"})
df = df[df["species_name"].notna()].reset_index(drop=True)

proj3d = np.load(os.path.join(OUT_DIR, f"content_umap3d_nn{NN}.npy"))
u1 = proj3d[:, 0]; u2 = proj3d[:, 1]
X  = np.column_stack([u1, u2])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km = KMeans(n_clusters=2, n_init=30, random_state=SEED)
km_labels = km.fit_predict(X_scaled)
svm = SVC(kernel="linear", C=1.0, random_state=SEED)
svm.fit(X_scaled, km_labels)
svm_labels = svm.predict(X_scaled)
if u1[svm_labels == 0].mean() > u1[svm_labels == 1].mean():
    svm_labels = 1 - svm_labels

df["cluster"] = np.where(svm_labels == 0, "left", "right")
left  = df[df["cluster"] == "left"]
right = df[df["cluster"] == "right"]
n_L, n_R = len(left), len(right)
print(f"  Left: {n_L:,} ({n_L/len(df)*100:.1f}%)   Right: {n_R:,} ({n_R/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Left vs Right grouped bars (strand / phase / species)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] umap12_cluster_stats_svm2d.png")
panels = [
    ("Strand",      "strand_label", ["forward (+)", "reverse (-)"], lambda k: k),
    ("Codon phase", "codon_phase",  [0, 1, 2],                      lambda k: f"phase {k}"),
    ("Species",     "species_name", SPECIES_LIST,                    lambda k: k),
]
fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=BG)
bar_w = 0.38
for ax, (title, col, keys, lbl_fn) in zip(axes, panels):
    x = np.arange(len(keys))
    vals_l = [(left[col]  == k).mean() * 100 for k in keys]
    vals_r = [(right[col] == k).mean() * 100 for k in keys]
    ax.bar(x - bar_w/2, vals_l, bar_w, color=CLR_LEFT,  alpha=0.9, label=f"left  (n={n_L:,})",  zorder=3)
    ax.bar(x + bar_w/2, vals_r, bar_w, color=CLR_RIGHT, alpha=0.9, label=f"right  (n={n_R:,})", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl_fn(k) for k in keys], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("% of cluster"); ax.set_ylim(0, 100)
    ax.set_title(title, fontweight="bold", pad=10)
    style_ax(ax); ax.legend(fontsize=10, frameon=False)
fig.suptitle(
    f"Left vs Right  ·  2D SVM (U1×U2)  ·  16K bp  ·  n_neighbors = {NN}",
    fontsize=13, fontweight="bold", y=1.02, color=C_INK)
plt.tight_layout()
plt.savefig(os.path.join(GREEN_DIR, "umap12_cluster_stats_svm2d.png"), dpi=180, bbox_inches="tight")
plt.close()
print("  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 & 3 — Per-cluster bar charts
# ══════════════════════════════════════════════════════════════════════════════
for side_name, sub, c_main in [("left", left, CLR_LEFT), ("right", right, CLR_RIGHT)]:
    print(f"\n[{'2' if side_name=='left' else '3'}] cluster_stats_{side_name}_svm2d_u12.png")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG)

    # Strand
    ax = axes[0]
    strands = ["forward (+)", "reverse (-)"]
    vals = [(sub["strand_label"] == s).mean() * 100 for s in strands]
    bars = ax.bar(strands, vals, color=[STRAND_CLR[s] for s in strands], alpha=0.9, width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10, color=C_INK)
    ax.set_ylim(0, 100); ax.set_ylabel("% of cluster")
    ax.set_title("DNA strand", fontweight="bold"); style_ax(ax)

    # Phase
    ax = axes[1]
    phase_labels = ["phase 0\n(codon-aligned)", "phase +1", "phase +2"]
    vals = [(sub["codon_phase"] == ph).mean() * 100 for ph in [0, 1, 2]]
    bars = ax.bar(phase_labels, vals,
                  color=[PHASE_CLR[0], PHASE_CLR[1], PHASE_CLR[2]],
                  alpha=0.9, width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10, color=C_INK)
    ax.set_ylim(0, 100); ax.set_title("Codon phase", fontweight="bold"); style_ax(ax)

    # Species
    ax = axes[2]
    sp_labels = [s.replace("_", "\n") for s in SPECIES_LIST]
    vals = [(sub["species_name"] == sp).mean() * 100 for sp in SPECIES_LIST]
    bars = ax.bar(sp_labels, vals, color=[SP_COLORS[sp] for sp in SPECIES_LIST],
                  alpha=0.9, width=0.6, zorder=3)
    for bar, v in zip(bars, vals):
        if v > 2:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9, color=C_INK)
    ax.set_ylim(0, 100); ax.set_ylabel("% of cluster")
    ax.set_title("Species", fontweight="bold")
    ax.tick_params(axis="x", labelsize=9); style_ax(ax)

    fig.suptitle(
        f"{side_name.capitalize()} cluster  (n={len(sub):,}, {len(sub)/len(df)*100:.0f}%)  ·  "
        f"2D SVM (U1×U2)  ·  16K bp",
        fontsize=13, fontweight="bold", y=1.02, color=C_INK)
    plt.tight_layout()
    plt.savefig(os.path.join(GREEN_DIR, f"cluster_stats_{side_name}_svm2d_u12.png"), dpi=180, bbox_inches="tight")
    plt.close()
    print("  ✓")

print(f"\nDone → {GREEN_DIR}/")
