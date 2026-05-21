"""
Content embedding → 3D UMAP (cached) → 2D linear SVM on U1×U2.
Saves to green_plots/:
  umap12_clusters_overlay_svm2d.pdf
  round2_left_umap12_svm2d.pdf
  round2_right_umap12_svm2d.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import umap as umap_lib
import os

DATA_PATH   = "/fsx/dana_aubakirova/carbon_ablations/data/eukaryote/test_new.parquet"
OUT_DIR     = "/fsx/dana_aubakirova/carbon_ablations/clustering/output/carbon_test_new_3emb_v2_16k"
GREEN_DIR   = os.path.join(OUT_DIR, "green_plots")
SLICE_START = 96000 - 16384
NN          = 100
SEED        = 42

SPECIES_MAP = {
    "<fng>": "fungi", "<pln>": "plant", "<inv>": "invertebrate",
    "<prt>": "protozoa", "<vrt>": "vertebrate_other", "<mam>": "vertebrate_mammalian",
}

C_DARK_GREEN = "#0A6B2A"   # deeper forest green
C_BROWN      = "#6B3F18"   # rich dark brown
C_YELLOW     = "#F5A800"   # saturated amber-yellow
C_INK        = "#1A1F18"
C_MUTED      = "#6C7066"
C_FAINT      = "#E4E2DA"
BG           = "#FFFFFF"

CLR_LEFT   = C_DARK_GREEN
CLR_RIGHT  = C_YELLOW
STRAND_CLR = {"forward (+)": C_DARK_GREEN, "reverse (-)": C_YELLOW}
PHASE_CLR  = {0: C_DARK_GREEN, 1: C_YELLOW, 2: C_BROWN}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 24,
    "axes.titlesize": 26, "axes.labelsize": 24, "legend.fontsize": 22,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "xtick.color": C_INK, "ytick.color": C_INK,
})


def style_ax(ax, no_ticks=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_FAINT)
    ax.spines["bottom"].set_color(C_FAINT)
    if no_ticks:
        ax.set_xticks([]); ax.set_yticks([])


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading parquet…")
df = pd.read_parquet(DATA_PATH)
df["species_name"] = df["species_type"].map(SPECIES_MAP)
df["codon_phase"]  = (df["start"] - SLICE_START) % 3
df["strand_label"] = df["strand"].map({"<+>": "forward (+)", "<->": "reverse (-)"})
df = df[df["species_name"].notna()].reset_index(drop=True)

# ── 3D UMAP (cached) ──────────────────────────────────────────────────────────
umap3d_path = os.path.join(OUT_DIR, f"content_umap3d_nn{NN}.npy")
if os.path.exists(umap3d_path):
    print(f"Loading cached 3D UMAP…")
    proj3d = np.load(umap3d_path)
else:
    print(f"Computing 3D UMAP (n_neighbors={NN})…")
    content_emb = np.load(os.path.join(OUT_DIR, "content_embeddings.npy"))
    proj3d = umap_lib.UMAP(
        n_components=3, n_neighbors=NN,
        metric="cosine", random_state=SEED,
    ).fit_transform(content_emb)
    np.save(umap3d_path, proj3d)
    print(f"  Saved → {umap3d_path}")

u1 = proj3d[:, 0]
u2 = proj3d[:, 1]
X  = np.column_stack([u1, u2])

# ── 2D linear SVM on U1×U2 ───────────────────────────────────────────────────
print("Computing 2D SVM on U1×U2…")
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=2, n_init=30, random_state=SEED)
km_labels = km.fit_predict(X_scaled)

svm = SVC(kernel="linear", C=1.0, random_state=SEED)
svm.fit(X_scaled, km_labels)

w = svm.coef_[0]
b = svm.intercept_[0]

svm_labels = svm.predict(X_scaled)
if u1[svm_labels == 0].mean() > u1[svm_labels == 1].mean():
    svm_labels = 1 - svm_labels

df["cluster"] = np.where(svm_labels == 0, "left", "right")
left  = df[df["cluster"] == "left"]
right = df[df["cluster"] == "right"]
n_L, n_R = len(left), len(right)
print(f"  Left: {n_L:,}  ({n_L/len(df)*100:.1f}%)   Right: {n_R:,}  ({n_R/len(df)*100:.1f}%)")

# ── Boundary line in U1×U2 space ─────────────────────────────────────────────
u1_range = np.linspace(u1.min() - 0.5, u1.max() + 0.5, 400)
u1_s     = (u1_range - scaler.mean_[0]) / scaler.scale_[0]

def boundary_u2(u1_s_vals, offset=0):
    u2_s = -(w[0] * u1_s_vals + b + offset) / w[1]
    return u2_s * scaler.scale_[1] + scaler.mean_[1]

u2_mid = boundary_u2(u1_s, offset=0)
u2_lo  = boundary_u2(u1_s, offset=+1)
u2_hi  = boundary_u2(u1_s, offset=-1)


def draw_boundary(ax):
    ax.plot(u1_range, u2_mid, color=C_INK, lw=1.8, ls="-", alpha=0.85, zorder=5,
            label="SVM boundary")
    ax.fill_between(u1_range, u2_lo, u2_hi, alpha=0.07, color=C_MUTED, zorder=4)
    ax.set_xlim(u1.min() - 0.3, u1.max() + 0.3)
    ax.set_ylim(u2.min() - 0.3, u2.max() + 0.3)


kw = dict(s=4, alpha=0.7, linewidths=0, rasterized=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1) umap12_clusters_overlay_svm2d.png
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] umap12_clusters_overlay_svm2d.png")
fig, axes = plt.subplots(1, 3, figsize=(32, 10), facecolor=BG)

leg_kw = dict(fontsize=20, markerscale=3, frameon=False, loc="upper right")

ax = axes[0]
for lbl, c, nice in [("left", CLR_LEFT, f"left  (n={n_L:,})"),
                      ("right", CLR_RIGHT, f"right  (n={n_R:,})")]:
    m = df["cluster"].values == lbl
    ax.scatter(u1[m], u2[m], c=c, label=nice, **kw)
draw_boundary(ax)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("(A)  Taxonomic clusters", fontweight="bold")
ax.legend(**leg_kw)
style_ax(ax)

ax = axes[1]
for strand, c in STRAND_CLR.items():
    m = df["strand_label"].values == strand
    ax.scatter(u1[m], u2[m], c=c, label=strand, **kw)
draw_boundary(ax)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("(B)  DNA strand orientation", fontweight="bold")
ax.legend(**leg_kw)
style_ax(ax)

ax = axes[2]
for ph, lbl in [(0, "phase 0  (codon-aligned)"), (1, "phase +1"), (2, "phase +2")]:
    m = df["codon_phase"].values == ph
    ax.scatter(u1[m], u2[m], c=PHASE_CLR[ph], label=lbl, **kw)
draw_boundary(ax)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("(C)  Codon phase", fontweight="bold")
ax.legend(**leg_kw)
style_ax(ax)

fig.suptitle(
    "Carbon 3B · Content token  ·  UMAP 1 vs UMAP 2  ·  16K bp  ·  2D linear SVM (U1×U2)",
    fontsize=27, fontweight="bold", y=1.02, color=C_INK)
plt.tight_layout()
plt.savefig(os.path.join(GREEN_DIR, "umap12_clusters_overlay_svm2d.pdf"), bbox_inches="tight")
plt.close()
print("  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# 2 & 3) Round-2 UMAPs — plot as UMAP1 vs UMAP2
# ══════════════════════════════════════════════════════════════════════════════
content_emb = np.load(os.path.join(OUT_DIR, "content_embeddings.npy"))

for side_name, sub, c_main in [("left", left, CLR_LEFT), ("right", right, CLR_RIGHT)]:
    tag = "[2]" if side_name == "left" else "[3]"
    print(f"\n{tag} round2_{side_name}_umap12_svm2d.png  (n={len(sub):,})")
    cache = os.path.join(OUT_DIR, f"{side_name}_cluster_umap2d_svm2d_u12.npy")
    if os.path.exists(cache):
        print("  Loading cached…")
        proj2d = np.load(cache)
    else:
        idx    = sub.index.values
        emb_sub = content_emb[idx]
        print(f"  Running UMAP (n_neighbors={NN})…")
        proj2d = umap_lib.UMAP(
            n_components=2, n_neighbors=NN,
            metric="cosine", random_state=SEED,
        ).fit_transform(emb_sub)
        np.save(cache, proj2d)
        print(f"  Saved → {cache}")

    ua = proj2d[:, 0]; ub = proj2d[:, 1]
    fig, axes = plt.subplots(1, 3, figsize=(32, 10), facecolor=BG)

    leg_kw = dict(fontsize=20, markerscale=3, frameon=False, loc="upper right")

    ax = axes[0]
    ax.scatter(ua, ub, c=c_main, **kw)
    ax.set_title(f"(A)  {side_name.capitalize()} cluster\nn = {len(sub):,}",
                 fontweight="bold", color=c_main)

    ax = axes[1]
    sl = sub["strand_label"].values
    for strand, sc in STRAND_CLR.items():
        m = sl == strand
        ax.scatter(ua[m], ub[m], c=sc, label=f"{strand}  ({m.mean():.1%})", **kw)
    ax.set_title("(B)  DNA strand orientation", fontweight="bold")
    ax.legend(**leg_kw)

    ax = axes[2]
    cp = sub["codon_phase"].values
    for ph, lbl in [(0, "phase 0  (codon-aligned)"), (1, "phase +1"), (2, "phase +2")]:
        m = cp == ph
        ax.scatter(ua[m], ub[m], c=PHASE_CLR[ph], label=f"{lbl}  ({m.mean():.1%})", **kw)
    ax.set_title("(C)  Codon phase", fontweight="bold")
    ax.legend(**leg_kw)

    for ax in axes:
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        style_ax(ax)

    fig.suptitle(
        f"Round-2 UMAP · {side_name.capitalize()} cluster (2D SVM U1×U2)  ·  16K bp  ·  n_neighbors = {NN}",
        fontsize=27, fontweight="bold", y=1.02, color=C_INK)
    plt.tight_layout()
    plt.savefig(os.path.join(GREEN_DIR, f"round2_{side_name}_umap12_svm2d.pdf"), bbox_inches="tight")
    plt.close()
    print("  ✓")

print(f"\nDone → {GREEN_DIR}/")
