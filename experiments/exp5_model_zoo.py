"""
exp5_model_zoo.py — "what other models could you have used, and what would
it cost?" (the efficiency-frontier extension the faculty implicitly asked
about alongside Q1/Q4).

Single-view (Method 1 protocol: fit on the 600-row seed only) across a wider
model zoo, reporting macro-F1 vs fit time vs a rough learned-parameter count,
to give a defensible "here is the accuracy/compute frontier we chose from"
answer without abandoning the project's no-GPU, classical-ML premise.
"""
import time
import numpy as np
import pandas as pd

from common import build_working_set, build_view_a, build_view_b_soundex, make_split, SEEDS
from sklearn.metrics import f1_score, accuracy_score

df = build_working_set()
texts = df["clean_text"].tolist()
X_A, vec_A = build_view_a(texts)
X_B, vec_B, _ = build_view_b_soundex(texts)


def param_count(clf, n_features, n_classes=3):
    name = type(clf).__name__
    if name in ("LinearSVC", "LogisticRegression", "RidgeClassifier"):
        return n_features * n_classes
    if name == "SVMWithProba":
        return n_features * n_classes
    if name == "ComplementNB":
        return n_features * n_classes * 2
    if name == "SGDClassifier":
        return n_features * n_classes
    if name in ("RandomForestClassifier",):
        return sum(t.tree_.node_count for t in clf.estimators_)
    if name in ("HistGradientBoostingClassifier",):
        return sum(len(p.nodes) for pred in clf._predictors for p in pred)
    return np.nan


MODELS = [
    ("A", "svm_softmax", {}),
    ("A", "logreg", {}),
    ("A", "svm_calibrated", {}),
    ("A", "random_forest", {}),
    ("A", "hist_gb", {}),
    ("B", "logreg", {}),
    ("B", "complement_nb", {}),
    ("B", "random_forest", {}),
    ("B", "hist_gb", {}),
]

rows = []
for view, kind, kw in MODELS:
    X = X_A if view == "A" else X_B
    for seed in SEEDS:
        d = make_split(df, X_A, X_B, seed)
        X_L = d["XA_L"] if view == "A" else d["XB_L"]
        X_T = d["XA_T"] if view == "A" else d["XB_T"]
        from common import make_view_a_classifier, make_view_b_classifier
        maker = make_view_a_classifier if view == "A" else make_view_b_classifier
        t0 = time.time()
        try:
            clf = maker(kind=kind, seed=seed, **kw)
            if kind == "hist_gb":
                clf.fit(X_L.toarray(), d["y_L"])
            else:
                clf.fit(X_L, d["y_L"])
            fit_s = time.time() - t0
            pred = clf.predict(X_T.toarray() if kind == "hist_gb" else X_T)
            f1 = f1_score(d["y_T"], pred, average="macro")
            acc = accuracy_score(d["y_T"], pred)
            params = param_count(clf.model if hasattr(clf, "model") else clf, X_L.shape[1])
        except Exception as e:
            print(f"  [FAILED] view={view} kind={kind} seed={seed}: {e}")
            continue
        rows.append(dict(view=view, model=kind, seed=seed, f1=f1, acc=acc, fit_s=fit_s,
                         params=params))
        print(f"  view {view} | {kind:16s} | seed {seed:5d} | f1={f1:.4f} "
              f"fit={fit_s:.3f}s params={params}")

result_df = pd.DataFrame(rows)
summary = (result_df.groupby(["view", "model"])
          .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"), fit_s_mean=("fit_s", "mean"),
               params=("params", "first"))
          .round(4).sort_values(["view", "f1_mean"], ascending=[True, False]).reset_index())

print("\n" + "=" * 90)
print("MODEL ZOO / EFFICIENCY FRONIER (mean over 3 seeds)")
print("=" * 90)
print(summary.to_string(index=False))
summary.to_csv("results/exp5_model_zoo.csv", index=False)
print("\nSaved to results/exp5_model_zoo.csv")
