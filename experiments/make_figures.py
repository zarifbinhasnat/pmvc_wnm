"""
make_figures.py — renders every chart used in the PDF report from the
results/*.csv files. Run after exp1-exp7 have all completed.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
OUT = "report/figures"
import os
os.makedirs(OUT, exist_ok=True)

CLASSES = ["Negative", "Neutral", "Positive"]


# --- Fig 1: Q1 classifier comparison bar chart ------------------------------
df1 = pd.read_csv("results/exp1_classifier_matrix.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = {"A": "#4C72B0", "B": "#DD8452"}
labels = [f"{r.view}: {r.classifier}" for r in df1.itertuples()]
bars = ax.barh(labels, df1["f1_mean"], xerr=df1["f1_std"],
               color=[colors[v] for v in df1["view"]], capsize=3)
ax.set_xlabel("Macro-F1 (mean over 3 seeds, error = std)")
ax.set_title("Q1: View A / View B classifier comparison")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_classifier_comparison.png")
plt.close()

# --- Fig 2: Q3b purity trajectory -------------------------------------------
# hand-entered from exp3's printed per-iteration trace (seed 42)
iters = list(range(12))
standard_purity = [0.916, 0.910, 0.900, 0.902, 0.901, 0.899, 0.899, 0.899, 0.897, 0.897, 0.892, 0.891]
agreement_purity = [0.929, 0.880, 0.859, 0.863, 0.863, 0.864, 0.864, 0.866, 0.859, 0.862, 0.854, 0.854]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(iters, standard_purity, "o-", label="View-A-confidence only (standard)", color="#4C72B0")
ax.plot(iters, agreement_purity, "s-", label="Agreement + confident (PMVC-WNM)", color="#C44E52")
ax.axvspan(0, 1, alpha=0.1, color="gray")
ax.annotate("crossover", xy=(1.5, 0.88), fontsize=9, style="italic", color="gray")
ax.set_xlabel("Co-training iteration")
ax.set_ylabel("Pseudo-label purity (vs. hidden true label)")
ax.set_title("Q3: Purity trajectory -- the gate's advantage inverts over iterations")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_purity_trajectory.png")
plt.close()

# --- Fig 3: Q3c threshold sweep ---------------------------------------------
df3 = pd.read_csv("results/exp3_threshold_sweep.csv")
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(df3["threshold"], df3["f1_best_mean"], "o-", color="#4C72B0", label="Macro-F1")
ax1.set_xlabel("Confidence threshold")
ax1.set_ylabel("Macro-F1", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax2 = ax1.twinx()
ax2.plot(df3["threshold"], df3["mean_purity"], "s--", color="#C44E52", label="Purity")
ax2.set_ylabel("Pseudo-label purity", color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")
ax1.axvline(0.55, color="gray", linestyle=":", alpha=0.7)
ax1.annotate("deployed (0.55)", xy=(0.55, ax1.get_ylim()[0]), fontsize=9, color="gray")
ax1.set_title("Q3: Threshold sweep -- F1 vs. purity trade-off")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_threshold_sweep.png")
plt.close()

# --- Fig 4: Q4 improvement ladder -------------------------------------------
df4 = pd.read_csv("results/exp4_improvement_ladder.csv")
df4_plot = df4[~df4["rung"].str.contains("View B|FULL-SUPERVISION")]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(df4_plot["rung"], df4_plot["f1_mean"], xerr=df4_plot["f1_std"],
       color="#55A868", capsize=3)
ax.set_xlabel("Macro-F1")
ax.set_title("Q4: Cumulative single-view improvement ladder")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_improvement_ladder.png")
plt.close()

# --- Fig 5: exp6 exhaustive model zoo (both views) --------------------------
df6 = pd.read_csv("results/exp6_exhaustive_model_zoo.csv")
for view in ["A", "B"]:
    sub = df6[df6.view == view].sort_values("f1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    family_colors = {"linear": "#4C72B0", "probabilistic": "#DD8452", "instance_based": "#55A868",
                     "tree": "#C44E52", "tree_ensemble": "#8172B2", "boosting": "#937860",
                     "neural": "#DA8BC3"}
    colors = [family_colors.get(f, "gray") for f in sub["family"]]
    ax.barh(sub["model"], sub["f1_mean"], xerr=sub["f1_std"], color=colors, capsize=2)
    ax.set_xlabel("Held-out macro-F1 (mean over 3 seeds)")
    ax.set_title(f"Exhaustive model comparison -- View {view}")
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f) for f, c in family_colors.items()
              if f in sub["family"].values]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="ML family")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_exhaustive_zoo_view{view}.png")
    plt.close()

# --- Fig 6: CV vs holdout scatter (generalization gap) ----------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
for view, marker in [("A", "o"), ("B", "^")]:
    sub = df6[df6.view == view]
    ax.scatter(sub["cv_f1_mean"], sub["f1_mean"], marker=marker, s=60, alpha=0.7, label=f"View {view}")
lims = [0.35, 0.72]
ax.plot(lims, lims, "k--", alpha=0.4, label="CV = holdout")
ax.set_xlabel("5-fold CV macro-F1 (on the 600-row seed)")
ax.set_ylabel("Held-out test macro-F1")
ax.set_title("CV estimate vs. true generalization")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_cv_vs_holdout.png")
plt.close()

# --- Fig 7: label-budget sensitivity ----------------------------------------
df7a = pd.read_csv("results/exp7_budget_sensitivity.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
styles = {"team_A": ("o-", "#4C72B0"), "best_A": ("o--", "#4C72B0"),
         "team_B": ("s-", "#DD8452"), "best_B": ("s--", "#DD8452")}
for model, (style, color) in styles.items():
    sub = df7a[df7a.model == model].sort_values("n_labeled")
    ax.plot(sub["n_labeled"], sub["mean"], style, color=color, label=model)
ax.set_xlabel("Labeled seed size")
ax.set_ylabel("Macro-F1")
ax.set_title("Label-budget sensitivity: team's model vs. best-found model")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_budget_sensitivity.png")
plt.close()

# --- Fig 8: noise robustness -------------------------------------------------
df7b = pd.read_csv("results/exp7_noise_robustness.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
width = 0.2
noise_rates = sorted(df7b.noise_rate.unique())
models = ["team_A", "best_A", "team_B", "best_B"]
x = np.arange(len(noise_rates))
for i, model in enumerate(models):
    sub = df7b[df7b.model == model].sort_values("noise_rate")
    ax.bar(x + i * width, sub["mean"], width, label=model)
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([f"{r:.0%}" for r in noise_rates])
ax.set_xlabel("Injected spelling-noise rate")
ax.set_ylabel("Macro-F1")
ax.set_title("Spelling-noise robustness")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_noise_robustness.png")
plt.close()

# --- Fig 9/10: confusion matrices -------------------------------------------
with open("results/exp7_best_models.txt") as f:
    lines = dict(l.strip().split("=") for l in f if "=" in l)
for view, fname in [("A", "results/exp7_confusion_matrix_A.npy"),
                    ("B", "results/exp7_confusion_matrix_B.npy")]:
    cm = np.load(fname)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    kind = lines.get(f"BEST_{view}_KIND", "?")
    ax.set_title(f"Confusion matrix -- View {view} best model ({kind})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig9_confusion_view{view}.png")
    plt.close()

print("All figures written to report/figures/")
import glob
for f in sorted(glob.glob(f"{OUT}/*.png")):
    print(" ", f)
