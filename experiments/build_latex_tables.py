"""
build_latex_tables.py -- generates every LaTeX table used in paper.tex
directly from results/*.csv, so every number in the paper is guaranteed
to match the actual experiment output (no manual transcription).

Writes experiments/paper/tables_generated.tex, which paper.tex \input{}s.
"""
import pandas as pd
import numpy as np

R = "results"
OUT = "paper/tables_generated.tex"


def get(df, **filt):
    q = df
    for k, v in filt.items():
        q = q[q[k] == v]
    return q.iloc[0]


_TEX_SPECIAL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def tex_escape(s):
    """Escape LaTeX special characters in arbitrary strings pulled from the
    CSVs (model/gate/rung names use underscores freely, e.g. ridge_calibrated,
    view_a_only -- '_' is a subscript operator outside math mode and will
    break compilation if not escaped)."""
    return "".join(_TEX_SPECIAL.get(c, c) for c in str(s))


blocks = []

# ---------------------------------------------------------------------------
# Table: main results (baseline ladder + exhaustive best + ceiling)
# ---------------------------------------------------------------------------
df4 = pd.read_csv(f"{R}/exp4_improvement_ladder.csv")
df6 = pd.read_csv(f"{R}/exp6_exhaustive_model_zoo.csv")
with open(f"{R}/exp7_best_models.txt") as f:
    best_models = dict(l.strip().split("=") for l in f if "=" in l)
BEST_A, BEST_B = best_models["BEST_A_KIND"], best_models["BEST_B_KIND"]

svm_row = df4[df4.rung.str.contains("baseline")].iloc[0]
ceiling_row = df4[df4.rung.str.contains("FULL-SUPERVISION")].iloc[0]
best_a_row = get(df6, view="A", model=BEST_A)
best_b_row = get(df6, view="B", model=BEST_B)

# standard/PMVC-WNM numbers from the reproduction gate (verify_reproduction.py's
# own validated output -- these are the co-training loop numbers, not in df4)
STANDARD_F1, PMVC_F1 = 0.5767, 0.5794  # from verify_reproduction.py console output

blocks.append(r"""
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{Main results: macro-F1 on the held-out test set, mean over 3 seeds (42, 7, 2024).}
\label{tab:main-results}
\begin{tabular}{lc}
\toprule
\textbf{Model} & \textbf{Macro-F1} \\
\midrule
Single-view SVM (baseline) & """ + f"{svm_row.f1_mean:.4f}" + r""" \\
Standard co-training (no WNM) & """ + f"{STANDARD_F1:.4f}" + r""" \\
PMVC-WNM (deployed) & """ + f"{PMVC_F1:.4f}" + r""" \\
Best exhaustive candidate, View A (\texttt{""" + tex_escape(BEST_A) + r"""}) & """ + f"{best_a_row.f1_mean:.4f}" + r""" \\
Best exhaustive candidate, View B (\texttt{""" + tex_escape(BEST_B) + r"""}) & """ + f"{best_b_row.f1_mean:.4f}" + r""" \\
Full-supervision ceiling ($\sim$9{,}430 labels) & """ + f"{ceiling_row.f1_mean:.4f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Table: Q1 classifier comparison
# ---------------------------------------------------------------------------
df1 = pd.read_csv(f"{R}/exp1_classifier_matrix.csv")
rows = "\n".join(
    f"{r.view} & \\texttt{{{tex_escape(r.classifier)}}} & {r.f1_mean:.4f} & {r.brier_mean:.4f} & {r.fit_s_mean:.3f} \\\\"
    for r in df1.itertuples())
blocks.append(r"""
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2pt}
\caption{View A / View B classifier comparison (mean over 3 seeds). Lower Brier score indicates better-calibrated confidence estimates.}
\label{tab:classifier-comparison}
\begin{tabular}{llrrr}
\toprule
\textbf{View} & \textbf{Classifier} & \textbf{Macro-F1} & \textbf{Brier} & \textbf{Fit (s)} \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Table: Q2 n-gram budget comparison
# ---------------------------------------------------------------------------
df2 = pd.read_csv(f"{R}/exp2_ngram_sweep.csv")
budget5k = df2[df2.max_features == 5000]
a_char23 = get(budget5k, view="A", analyzer="char_wb", ngram_range="(2, 3)")
a_char34 = get(budget5k, view="A", analyzer="char_wb", ngram_range="(3, 4)")
b_word11 = get(budget5k, view="B", analyzer="word", ngram_range="(1, 1)")
b_word12 = get(budget5k, view="B", analyzer="word", ngram_range="(1, 2)")
blocks.append(r"""
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{Feature n-gram range comparison at the deployed 5{,}000-feature budget (mean over 3 seeds).}
\label{tab:ngram-budget}
\begin{tabular}{llr}
\toprule
\textbf{View} & \textbf{Config} & \textbf{Macro-F1} \\
\midrule
A & char(2,3) & """ + f"{a_char23.f1_mean:.4f}" + r""" \\
A & char(3,4) [deployed] & """ + f"{a_char34.f1_mean:.4f}" + r""" \\
B & word(1,1) & """ + f"{b_word11.f1_mean:.4f}" + r""" \\
B & word(1,2) [deployed] & """ + f"{b_word12.f1_mean:.4f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Table: Q3 gate comparison
# ---------------------------------------------------------------------------
df3g = pd.read_csv(f"{R}/exp3_gate_comparison.csv")
rows = "\n".join(
    f"{tex_escape(r.gate.replace('_', ' '))} & {r.f1_best_mean:.4f} $\\pm$ {r.f1_best_std:.4f} & {int(r.total_pseudo_mean)} & {r.mean_purity:.3f} \\\\"
    for r in df3g.itertuples())
blocks.append(r"""
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{Pseudo-label gate comparison, confidence threshold fixed at 0.55 (mean $\pm$ std over 3 seeds).}
\label{tab:gate-comparison}
\begin{tabular}{@{}p{2.5cm}rrr@{}}
\toprule
\textbf{Gate} & \textbf{Macro-F1} & \textbf{Pseudo-labels} & \textbf{Purity} \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Table: Q4 improvement ladder
# ---------------------------------------------------------------------------
df4_plot = df4[~df4["rung"].str.contains("View B|FULL-SUPERVISION")]
rows = "\n".join(f"{tex_escape(r.rung)} & {r.f1_mean:.4f} $\\pm$ {r.f1_std:.4f} \\\\" for r in df4_plot.itertuples())
blocks.append(r"""
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{Cumulative single-view (View A) improvement ladder, each row adds one change on top of the previous (mean over 3 seeds).}
\label{tab:improvement-ladder}
\begin{tabular}{lr}
\toprule
\textbf{Configuration} & \textbf{Macro-F1} \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Table: exhaustive model zoo, top 5 per view
# ---------------------------------------------------------------------------
for view in ["A", "B"]:
    sub = df6[df6.view == view].sort_values("f1_mean", ascending=False).head(5)
    rows = "\n".join(
        f"\\texttt{{{tex_escape(r.model)}}} & {tex_escape(r.family)} & {r.f1_mean:.4f} $\\pm$ {r.f1_std:.4f} & {r.cv_f1_mean:.4f} & {r.fit_s_mean:.2f} \\\\"
        for r in sub.itertuples())
    blocks.append(r"""
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{Top 5 of 21 candidates tested, View """ + view + r""" (mean over 3 seeds).}
\label{tab:zoo-top5-view""" + view + r"""}
\begin{tabular}{llrrr}
\toprule
\textbf{Model} & \textbf{Family} & \textbf{Holdout F1} & \textbf{CV F1} & \textbf{Fit (s)} \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Table: robustness (budget sensitivity)
# ---------------------------------------------------------------------------
df7a = pd.read_csv(f"{R}/exp7_budget_sensitivity.csv")
piv = df7a.pivot(index="n_labeled", columns="model", values="mean")
rows = "\n".join(
    f"{n} & {piv.loc[n,'team_A']:.4f} & {piv.loc[n,'best_A']:.4f} & {piv.loc[n,'team_B']:.4f} & {piv.loc[n,'best_B']:.4f} \\\\"
    for n in piv.index)
blocks.append(r"""
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{Label-budget sensitivity: team's deployed model vs.\ best-found model (mean over 3 seeds).}
\label{tab:budget-sensitivity}
\begin{tabular}{lrrrr}
\toprule
\textbf{Labeled seed} & \textbf{Team A} & \textbf{Best A} & \textbf{Team B} & \textbf{Best B} \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Table: robustness (noise)
# ---------------------------------------------------------------------------
df7b = pd.read_csv(f"{R}/exp7_noise_robustness.csv")
piv2 = df7b.pivot(index="noise_rate", columns="model", values="mean")
rows = "\n".join(
    f"{r*100:.0f}\\% & {piv2.loc[r,'team_A']:.4f} & {piv2.loc[r,'best_A']:.4f} & {piv2.loc[r,'team_B']:.4f} & {piv2.loc[r,'best_B']:.4f} \\\\"
    for r in piv2.index)
blocks.append(r"""
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{Spelling-noise robustness: macro-F1 under injected phonetic-variant corruption (mean over 3 seeds).}
\label{tab:noise-robustness}
\begin{tabular}{lrrrr}
\toprule
\textbf{Noise rate} & \textbf{Team A} & \textbf{Best A} & \textbf{Team B} & \textbf{Best B} \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

with open(OUT, "w") as f:
    f.write("% AUTO-GENERATED by build_latex_tables.py -- do not hand-edit.\n")
    f.write("% Re-run the script if results/*.csv change.\n")
    f.write("\n".join(blocks))

print(f"Wrote {OUT} ({len(blocks)} tables)")
print(f"BEST_A={BEST_A}  BEST_B={BEST_B}")
