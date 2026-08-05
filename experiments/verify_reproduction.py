"""
verify_reproduction.py — the non-negotiable first gate (see plan).

Runs the harness at the team's exact original configuration and checks the
result against the ground-truth numbers embedded in the teammate's own
notebook run (PMVC_WNM_Banglish_Classification_final.ipynb, cells 19/23):

    Single-view SVM       f1_best=0.5533  std=0.0127
    Standard Co-training  f1_best=0.5789  std=0.0100
    PMVC-WNM              f1_best=0.5802  std=0.0084

If this does not land close to those numbers, every other experiment in this
folder is untrustworthy and must not be used — this script says so explicitly
rather than letting a silent mismatch propagate.
"""
import time
import numpy as np
from common import (
    build_working_set, build_view_a, build_view_b_soundex, make_split,
    run_single_view, co_training, summarize_over_seeds, SEEDS,
)

GROUND_TRUTH = {
    "Single-view SVM":      dict(f1_best=0.5533, std=0.0127),
    "Standard Co-training": dict(f1_best=0.5789, std=0.0100),
    "PMVC-WNM":             dict(f1_best=0.5802, std=0.0084),
}
TOLERANCE = 0.02  # macro-F1 absolute tolerance (sklearn/version drift budget)

df = build_working_set()
print(f"working set: {len(df)} rows, label counts:\n{df['label'].value_counts().to_string()}\n")
assert len(df) == 11788, f"expected 11788 rows, got {len(df)} -- data loading has diverged"

X_A, vec_A = build_view_a(df["clean_text"].tolist())
X_B, vec_B, _ = build_view_b_soundex(df["clean_text"].tolist())
print(f"View A: {X_A.shape}   View B: {X_B.shape}\n")

results = {"Single-view SVM": [], "Standard Co-training": [], "PMVC-WNM": []}
t0 = time.time()
for seed in SEEDS:
    d = make_split(df, X_A, X_B, seed)
    print(f"seed {seed}: L={d['XA_L'].shape[0]} U={d['XA_U'].shape[0]} test={d['XA_T'].shape[0]}")

    _, s1, _ = run_single_view(d, view="A", seed=seed)
    results["Single-view SVM"].append([dict(iteration=0, f1=s1["f1"], acc=s1["acc"])])

    _, _, h2, _ = co_training(d, vec_A, mode="standard", seed=seed)
    results["Standard Co-training"].append(h2)

    _, _, h3, _ = co_training(d, vec_A, mode="pmvc_wnm", seed=seed)
    results["PMVC-WNM"].append(h3)

    print(f"  svm={s1['f1']:.4f}  cotrain_best={max(r['f1'] for r in h2):.4f}"
          f"  pmvc_best={max(r['f1'] for r in h3):.4f}")

print(f"\ndone in {time.time()-t0:.1f}s\n")

print(f"{'method':24s} {'f1_best':>9s} {'std':>8s}   {'ground_truth':>13s}  verdict")
all_pass = True
for name, hists in results.items():
    summ, _ = summarize_over_seeds(hists)
    gt = GROUND_TRUTH[name]
    diff = abs(summ["f1_best_mean"] - gt["f1_best"])
    ok = diff <= TOLERANCE
    all_pass &= ok
    print(f"{name:24s} {summ['f1_best_mean']:9.4f} {summ['f1_best_std']:8.4f}   "
          f"{gt['f1_best']:13.4f}  {'PASS' if ok else 'FAIL'} (diff={diff:.4f})")

print()
if all_pass:
    print(">>> REPRODUCTION GATE: PASSED. Proceeding with exp1-exp5 is justified.")
else:
    print(">>> REPRODUCTION GATE: FAILED. Do not trust downstream experiments "
          "until this is resolved.")
    raise SystemExit(1)
