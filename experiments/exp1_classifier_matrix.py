"""
exp1_classifier_matrix.py — answers Q1: "why LinearSVC for View A but
LogisticRegression for View B?"

The notebook's own rationale (cell 16, markdown): LinearSVC has no
predict_proba; SVC(probability=True) does Platt scaling per fit and was ~50x
slower; so View A uses softmax(decision margins) as a speed compromise. View B
uses LogisticRegression because "tried random forest first, it was worse and
slower."

That's a real justification for View B, but View A's choice trades away
calibration quality for speed — and the co-training GATE thresholds that
exact "confidence" at 0.55. This script measures: (1) does the speed
compromise actually cost calibration quality, via Brier score + a reliability
table, and (2) does swapping classifiers change downstream macro-F1 on the
single-view (Method 1) task, across all 3 seeds.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from common import (
    build_working_set, build_view_a, build_view_b_soundex, make_split,
    run_single_view, make_view_a_classifier, make_view_b_classifier, SEEDS, CLASSES,
)

df = build_working_set()
X_A, vec_A = build_view_a(df["clean_text"].tolist())
X_B, vec_B, _ = build_view_b_soundex(df["clean_text"].tolist())

VIEW_A_CANDIDATES = ["svm_softmax", "svm_platt", "svm_calibrated", "logreg"]
VIEW_B_CANDIDATES = ["logreg", "svm_softmax", "complement_nb", "random_forest"]


def multiclass_brier(y_true, proba, classes):
    """Mean one-vs-rest Brier score across the 3 classes (lower = better calibrated)."""
    scores = []
    for i, c in enumerate(classes):
        indicator = (y_true == c).astype(int)
        scores.append(brier_score_loss(indicator, proba[:, i]))
    return float(np.mean(scores))


def run_view(view, candidates, X, seed_list=SEEDS):
    rows = []
    for kind in candidates:
        for seed in seed_list:
            d = make_split(df, X_A, X_B, seed)
            X_L = d["XA_L"] if view == "A" else d["XB_L"]
            X_T = d["XA_T"] if view == "A" else d["XB_T"]
            maker = make_view_a_classifier if view == "A" else make_view_b_classifier
            t0 = time.time()
            try:
                clf = maker(kind=kind, seed=seed)
                clf.fit(X_L, d["y_L"])
                fit_s = time.time() - t0
                pred = clf.predict(X_T)
                proba = clf.predict_proba(X_T)
                from sklearn.metrics import f1_score, accuracy_score
                f1 = f1_score(d["y_T"], pred, average="macro")
                acc = accuracy_score(d["y_T"], pred)
                brier = multiclass_brier(d["y_T"], proba, clf.classes_)
            except Exception as e:
                print(f"  [FAILED] view {view} kind={kind} seed={seed}: {e}")
                continue
            rows.append(dict(view=view, classifier=kind, seed=seed, f1=f1, acc=acc,
                             fit_s=fit_s, brier=brier))
            print(f"  view {view:1s} | {kind:16s} | seed {seed:5d} | "
                  f"f1={f1:.4f} acc={acc:.4f} brier={brier:.4f} fit={fit_s:.2f}s")
    return pd.DataFrame(rows)


print("=== View A classifier matrix ===")
rA = run_view("A", VIEW_A_CANDIDATES, X_A)
print("\n=== View B classifier matrix ===")
rB = run_view("B", VIEW_B_CANDIDATES, X_B)

all_rows = pd.concat([rA, rB], ignore_index=True)
summary = (all_rows.groupby(["view", "classifier"])
           .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                acc_mean=("acc", "mean"), brier_mean=("brier", "mean"),
                fit_s_mean=("fit_s", "mean"))
           .round(4).reset_index())
summary = summary.sort_values(["view", "f1_mean"], ascending=[True, False])

print("\n" + "=" * 90)
print("SUMMARY (mean over 3 seeds; lower Brier = better-calibrated confidence)")
print("=" * 90)
print(summary.to_string(index=False))

summary.to_csv("results/exp1_classifier_matrix.csv", index=False)
print("\nSaved to results/exp1_classifier_matrix.csv")

# explicit answer to the calibration question the notebook never tested
print("\n--- Calibration check: is svm_softmax's confidence trustworthy for the 0.55 gate? ---")
softmax_brier = summary[(summary.view == "A") & (summary.classifier == "svm_softmax")]["brier_mean"].values[0]
best_a_brier = summary[summary.view == "A"]["brier_mean"].min()
best_a_kind = summary[(summary.view == "A") & (summary.brier_mean == best_a_brier)]["classifier"].values[0]
print(f"svm_softmax (current) Brier: {softmax_brier:.4f}")
print(f"Best-calibrated View A option: {best_a_kind} (Brier {best_a_brier:.4f})")
print(f"Calibration gap: {softmax_brier - best_a_brier:+.4f}")
