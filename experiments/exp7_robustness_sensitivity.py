"""
exp7_robustness_sensitivity.py — test cases beyond a single accuracy number:

  (a) Label-budget sensitivity: does the best model's advantage over the
      team's deployed model hold at different seed sizes (300/600/900/1200),
      or is it specific to N_LABELED=600?
  (b) Spelling-noise robustness: corrupt the TEST set with the same
      phonetic-variant swaps used elsewhere in this repo, at two severities,
      and measure how much each model's macro-F1 degrades.
  (c) Confusion matrix for the best model on each view (rubric requirement:
      "confusion matrices for the best model").

Reads results/exp6_exhaustive_model_zoo.csv to pick the best model per view
by held-out test macro-F1, so this script must be run after exp6.
"""
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from common import (
    build_working_set, build_view_a, build_view_b_soundex, make_split,
    make_view_a_classifier, make_view_b_classifier, PHONETIC_VARIANTS, SEEDS, CLASSES,
)
from exp6_exhaustive_model_zoo import make_model

df = build_working_set()
texts = df["clean_text"].tolist()
X_A, vec_A = build_view_a(texts)
X_B, vec_B, _ = build_view_b_soundex(texts)

zoo = pd.read_csv("results/exp6_exhaustive_model_zoo.csv")
best_a = zoo[zoo.view == "A"].sort_values("f1_mean", ascending=False).iloc[0]
best_b = zoo[zoo.view == "B"].sort_values("f1_mean", ascending=False).iloc[0]
print(f"Best View A model (from exp6): {best_a['model']} (f1={best_a['f1_mean']:.4f})")
print(f"Best View B model (from exp6): {best_b['model']} (f1={best_b['f1_mean']:.4f})")

TEAM_A_KIND, TEAM_B_KIND = "linsvc_softmax", "logreg"
BEST_A_KIND, BEST_B_DENSE = best_a["model"], bool(best_a.get("dense", False))
BEST_B_KIND = best_b["model"]
DENSE_MODELS = {"gradient_boosting_capped", "hist_gb_capped"}


def fit_predict(kind, X_L, y_L, X_T, seed):
    dense = kind in DENSE_MODELS
    clf = make_model(kind, seed)
    if dense:
        clf.fit(X_L.toarray(), y_L)
        return clf.predict(X_T.toarray())
    clf.fit(X_L, y_L)
    return clf.predict(X_T)


# --- (a) Label-budget sensitivity ------------------------------------------
print("\n" + "=" * 90)
print("(a) LABEL-BUDGET SENSITIVITY: team's model vs. best-found model")
print("=" * 90)
budget_rows = []
for n_labeled in [300, 600, 900, 1200]:
    for seed in SEEDS:
        d = make_split(df, X_A, X_B, seed, n_labeled=n_labeled)
        pred_team_a = fit_predict(TEAM_A_KIND, d["XA_L"], d["y_L"], d["XA_T"], seed)
        pred_best_a = fit_predict(BEST_A_KIND, d["XA_L"], d["y_L"], d["XA_T"], seed)
        pred_team_b = fit_predict(TEAM_B_KIND, d["XB_L"], d["y_L"], d["XB_T"], seed)
        pred_best_b = fit_predict(BEST_B_KIND, d["XB_L"], d["y_L"], d["XB_T"], seed)
        for label, pred in [("team_A", pred_team_a), ("best_A", pred_best_a),
                            ("team_B", pred_team_b), ("best_B", pred_best_b)]:
            budget_rows.append(dict(n_labeled=n_labeled, seed=seed, model=label,
                                    f1=f1_score(d["y_T"], pred, average="macro")))
    print(f"  n_labeled={n_labeled} done")

budget_df = pd.DataFrame(budget_rows)
budget_summary = budget_df.groupby(["n_labeled", "model"])["f1"].agg(["mean", "std"]).round(4)
print("\n" + budget_summary.to_string())
budget_summary.reset_index().to_csv("results/exp7_budget_sensitivity.csv", index=False)


# --- (b) Spelling-noise robustness -----------------------------------------
print("\n" + "=" * 90)
print("(b) SPELLING-NOISE ROBUSTNESS: corrupt TEST text, measure degradation")
print("=" * 90)


def corrupt_text(text, rng, prob):
    out = []
    for w in text.split():
        if rng.random() < prob:
            variants = PHONETIC_VARIANTS[:]
            rng.shuffle(variants)
            for a, b in variants:
                if a in w:
                    w = w.replace(a, b, 1); break
        out.append(w)
    return " ".join(out)


noise_rows = []
for noise_rate in [0.0, 0.20, 0.35]:
    for seed in SEEDS:
        d = make_split(df, X_A, X_B, seed)
        rng = random.Random(seed)
        test_texts_raw = list(df["clean_text"].values[d["idx_test"]])
        corrupted = [corrupt_text(t, rng, noise_rate) for t in test_texts_raw]

        XA_T_noisy, _ = build_view_a(corrupted, fit=False, vectorizer=vec_A)
        XB_T_noisy, _, _ = build_view_b_soundex(corrupted, fit=False, vectorizer=vec_B)

        pred_team_a = fit_predict(TEAM_A_KIND, d["XA_L"], d["y_L"], XA_T_noisy, seed)
        pred_best_a = fit_predict(BEST_A_KIND, d["XA_L"], d["y_L"], XA_T_noisy, seed)
        pred_team_b = fit_predict(TEAM_B_KIND, d["XB_L"], d["y_L"], XB_T_noisy, seed)
        pred_best_b = fit_predict(BEST_B_KIND, d["XB_L"], d["y_L"], XB_T_noisy, seed)
        for label, pred in [("team_A", pred_team_a), ("best_A", pred_best_a),
                            ("team_B", pred_team_b), ("best_B", pred_best_b)]:
            noise_rows.append(dict(noise_rate=noise_rate, seed=seed, model=label,
                                   f1=f1_score(d["y_T"], pred, average="macro")))
    print(f"  noise_rate={noise_rate} done")

noise_df = pd.DataFrame(noise_rows)
noise_summary = noise_df.groupby(["noise_rate", "model"])["f1"].agg(["mean", "std"]).round(4)
print("\n" + noise_summary.to_string())
noise_summary.reset_index().to_csv("results/exp7_noise_robustness.csv", index=False)

print("\n--- Relative degradation from clean (0.0) to heavy noise (0.35) ---")
piv = noise_summary["mean"].unstack("model")
for col in piv.columns:
    clean, heavy = piv.loc[0.0, col], piv.loc[0.35, col]
    print(f"  {col:8s}: clean={clean:.4f} -> noisy={heavy:.4f}  "
          f"(drop {clean-heavy:+.4f}, {100*(clean-heavy)/clean:.1f}%)")


# --- (c) Confusion matrices for the best models -----------------------------
print("\n" + "=" * 90)
print("(c) CONFUSION MATRICES (best model per view, seed=42)")
print("=" * 90)
d = make_split(df, X_A, X_B, 42)
pred_best_a = fit_predict(BEST_A_KIND, d["XA_L"], d["y_L"], d["XA_T"], 42)
pred_best_b = fit_predict(BEST_B_KIND, d["XB_L"], d["y_L"], d["XB_T"], 42)

cm_a = confusion_matrix(d["y_T"], pred_best_a, labels=CLASSES)
cm_b = confusion_matrix(d["y_T"], pred_best_b, labels=CLASSES)
print(f"\nView A ({BEST_A_KIND}):")
print(pd.DataFrame(cm_a, index=CLASSES, columns=CLASSES).to_string())
print(f"\nView B ({BEST_B_KIND}):")
print(pd.DataFrame(cm_b, index=CLASSES, columns=CLASSES).to_string())

np.save("results/exp7_confusion_matrix_A.npy", cm_a)
np.save("results/exp7_confusion_matrix_B.npy", cm_b)
with open("results/exp7_best_models.txt", "w") as f:
    f.write(f"BEST_A_KIND={BEST_A_KIND}\nBEST_B_KIND={BEST_B_KIND}\n")

print("\nSaved confusion matrices and best-model selection to results/")
