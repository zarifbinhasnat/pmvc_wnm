"""
exp4_single_view_improve.py — answers Q4: "how can the initial single-view
accuracy be improved?" (baseline: Single-view SVM, f1=0.5533)

The notebook's own honest diagnosis (cell 30): full supervision on all
~9,430 training rows only reaches ~0.68 macro-F1, so there is limited
headroom above the 600-row seed's ~0.55; and the View B Soundex encoder may
be a weaker link than the classifier choice.

This builds a cumulative-lift ladder on View A (each rung adds one change),
tests swapping in the pmvc_wnm repo's BNPC phonetic encoder for View B (which
scored ~0.59 vs ~0.39 for a similar Soundex-style encoder in earlier
benchmarking this session, on different data -- this is the direct test on
THIS dataset), and reproduces the full-supervision ceiling check.
"""
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from common import (
    build_working_set, build_view_a, build_view_b_soundex, build_view_b_bnpc,
    make_split, run_single_view, make_view_a_classifier, SEEDS,
)
from sklearn.metrics import f1_score, accuracy_score

df = build_working_set()
texts = df["clean_text"].tolist()
X_A_base, vec_A_base = build_view_a(texts)
X_B_base, vec_B_base, _ = build_view_b_soundex(texts)

rows = []


def eval_rung(name, X_A, X_B, clf_kind="svm_softmax", clf_kw=None, view="A"):
    clf_kw = clf_kw or {}
    for seed in SEEDS:
        d = make_split(df, X_A, X_B, seed)
        _, s, _ = run_single_view(d, view=view, clf_kind=clf_kind, seed=seed, **clf_kw)
        rows.append(dict(rung=name, seed=seed, f1=s["f1"], acc=s["acc"]))
        print(f"  {name:45s} | seed {seed:5d} | f1={s['f1']:.4f} acc={s['acc']:.4f}")


print("=== Rung 0: baseline (team's exact config) ===")
eval_rung("0. baseline (svm_softmax, char(3,4), 5k feat)", X_A_base, X_B_base)

print("\n=== Rung 1: max_features 5k -> 50k ===")
X_A_50k, _ = build_view_a(texts, max_features=50000)
eval_rung("1. max_features=50000", X_A_50k, X_B_base)

print("\n=== Rung 2: + min_df=2 (drop hapax noise) ===")
X_A_mindf, _ = build_view_a(texts, max_features=50000, min_df=2)
eval_rung("2. + min_df=2", X_A_mindf, X_B_base)

print("\n=== Rung 3: + class_weight='balanced' ===")
eval_rung("3. + class_weight=balanced", X_A_mindf, X_B_base,
          clf_kind="svm_softmax", clf_kw=dict(class_weight="balanced"))

print("\n=== Rung 4: + C tuned (grid) ===")
best_C, best_f1 = 1.0, -1
for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
    fs = []
    for seed in SEEDS:
        d = make_split(df, X_A_mindf, X_B_base, seed)
        _, s, _ = run_single_view(d, view="A", clf_kind="svm_softmax", seed=seed,
                                  class_weight="balanced", C=C)
        fs.append(s["f1"])
    m = np.mean(fs)
    print(f"    C={C:5.2f} -> f1={m:.4f}")
    if m > best_f1:
        best_f1, best_C = m, C
print(f"  best C = {best_C}")
eval_rung(f"4. + C tuned (C={best_C})", X_A_mindf, X_B_base,
          clf_kind="svm_softmax", clf_kw=dict(class_weight="balanced", C=best_C))

print("\n=== Rung 5: + word(1,2) char(3,5) feature union ===")
X_word, _ = build_view_a(texts, analyzer="word", ngram_range=(1, 2), max_features=25000, min_df=2)
X_char5, _ = build_view_a(texts, analyzer="char_wb", ngram_range=(3, 5), max_features=25000, min_df=2)
X_union = hstack([X_word, X_char5]).tocsr()
eval_rung("5. + word(1,2) UNION char(3,5) [50k total]", X_union, X_B_base,
          clf_kind="svm_softmax", clf_kw=dict(class_weight="balanced", C=best_C))

print("\n=== Rung 6: + probability calibration (CalibratedClassifierCV) ===")
eval_rung("6. + CalibratedClassifierCV(LinearSVC)", X_union, X_B_base,
          clf_kind="svm_calibrated", clf_kw=dict(class_weight="balanced", C=best_C))

print("\n=== Rung 7: View B swap -- BNPC (pmvc_wnm) instead of Soundex ===")
t0 = time.time()
X_B_bnpc, vec_B_bnpc, _ = build_view_b_bnpc(texts)
print(f"  BNPC build time: {time.time()-t0:.1f}s  shape={X_B_bnpc.shape}")
for seed in SEEDS:
    d = make_split(df, X_A_base, X_B_bnpc, seed)
    _, s, _ = run_single_view(d, view="B", clf_kind="logreg", seed=seed)
    rows.append(dict(rung="7. View B: BNPC (not Soundex)", seed=seed, f1=s["f1"], acc=s["acc"]))
    print(f"  View B (BNPC) | seed {seed:5d} | f1={s['f1']:.4f} acc={s['acc']:.4f}")
print("  [for comparison] View B (team's Soundex), same protocol:")
for seed in SEEDS:
    d = make_split(df, X_A_base, X_B_base, seed)
    _, s, _ = run_single_view(d, view="B", clf_kind="logreg", seed=seed)
    rows.append(dict(rung="7b. View B: Soundex (team baseline)", seed=seed, f1=s["f1"], acc=s["acc"]))
    print(f"  View B (Soundex) | seed {seed:5d} | f1={s['f1']:.4f} acc={s['acc']:.4f}")

print("\n=== Rung 8: full-supervision ceiling (all ~9,430 train rows, true labels) ===")
for seed in SEEDS:
    d = make_split(df, X_A_base, X_B_base, seed)
    from common import make_view_a_classifier
    clf = make_view_a_classifier(kind="svm_softmax", seed=seed)
    clf.fit(d["XA_train"], d["y_train"])
    pred = clf.predict(d["XA_T"])
    f1 = f1_score(d["y_T"], pred, average="macro")
    acc = accuracy_score(d["y_T"], pred)
    rows.append(dict(rung="8. FULL-SUPERVISION CEILING (~9.4k labels)", seed=seed, f1=f1, acc=acc))
    print(f"  full supervision | seed {seed:5d} | f1={f1:.4f} acc={acc:.4f}")

result_df = pd.DataFrame(rows)
summary = (result_df.groupby("rung", sort=False)
          .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"), acc_mean=("acc", "mean"))
          .round(4))
# preserve insertion order (rungs are cumulative, order matters for the ladder narrative)
order = list(dict.fromkeys(rows_i["rung"] for rows_i in rows))
summary = summary.reindex(order)

print("\n" + "=" * 95)
print("CUMULATIVE-LIFT LADDER (mean over 3 seeds)")
print("=" * 95)
print(summary.to_string())
summary.to_csv("results/exp4_improvement_ladder.csv")
print("\nSaved to results/exp4_improvement_ladder.csv")

baseline_f1 = summary.iloc[0]["f1_mean"]
print(f"\nTotal lift from baseline ({baseline_f1:.4f}):")
for rung, r in summary.iterrows():
    print(f"  {rung:50s} {r['f1_mean']:.4f}  ({r['f1_mean']-baseline_f1:+.4f})")
