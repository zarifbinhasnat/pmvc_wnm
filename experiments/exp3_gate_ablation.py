"""
exp3_gate_ablation.py — answers Q3: "why is the agreement gate so
conservative, and what does it cost/buy?"

The teammate's own write-up (notebook cell 30) already claims: "when both
views agree AND both are confident, pseudo-label purity sits around
0.86-0.92, versus roughly 0.70 if you just take view A's confident
predictions without the gate" and diagnoses the bottleneck as *quantity*, not
purity: "a model sitting at ~0.57 accuracy simply does not have thousands of
confidently-correct predictions lying around."

This script (1) verifies that purity claim directly instead of trusting it,
(2) extends it into a full quantity-vs-purity trade-off across 5 gate
variants, and (3) sweeps the confidence threshold on the team's gate to show
whether 0.55 is well-chosen.
"""
import numpy as np
import pandas as pd

from common import (
    build_working_set, build_view_a, build_view_b_soundex, make_split, co_training,
    summarize_over_seeds, GATES, SEEDS,
)

df = build_working_set()
texts = df["clean_text"].tolist()
X_A, vec_A = build_view_a(texts)
X_B, vec_B, _ = build_view_b_soundex(texts)

# --- Part 1: gate variants, team's threshold (0.55) held fixed -------------
print("=== Part 1: gate variant comparison (threshold=0.55 fixed) ===\n")
gate_rows = []
for gate_name, gate_fn in GATES.items():
    hists = []
    for seed in SEEDS:
        d = make_split(df, X_A, X_B, seed)
        use_wnm = "pmvc_wnm" in gate_name or "agreement" in gate_name
        _, _, h, _ = co_training(d, vec_A, mode="custom", gate_fn=gate_fn,
                                 use_wnm=use_wnm, seed=seed)
        hists.append(h)
        # aggregate purity/quantity across iterations for this seed
        chosen_counts = [r.get("n_chosen_this_iter", 0) for r in h]
        purities = [r.get("purity_this_iter", np.nan) for r in h if not np.isnan(r.get("purity_this_iter", np.nan))]
        f1_best = max(r["f1"] for r in h)
        total_pseudo = sum(chosen_counts)
        mean_purity = np.mean(purities) if purities else np.nan
        gate_rows.append(dict(gate=gate_name, seed=seed, f1_best=f1_best,
                              total_pseudo_labels=total_pseudo, mean_purity=mean_purity,
                              n_iterations_completed=len(h)))
        print(f"  {gate_name:38s} | seed {seed:5d} | f1_best={f1_best:.4f} | "
              f"total_pseudo={total_pseudo:5d} | mean_purity={mean_purity:.4f} | "
              f"iters={len(h)}")

gate_df = pd.DataFrame(gate_rows)
gate_summary = (gate_df.groupby("gate")
               .agg(f1_best_mean=("f1_best", "mean"), f1_best_std=("f1_best", "std"),
                    total_pseudo_mean=("total_pseudo_labels", "mean"),
                    mean_purity=("mean_purity", "mean"),
                    iters_mean=("n_iterations_completed", "mean"))
               .round(4).sort_values("f1_best_mean", ascending=False).reset_index())

print("\n" + "=" * 100)
print("GATE COMPARISON SUMMARY")
print("=" * 100)
print(gate_summary.to_string(index=False))
gate_summary.to_csv("results/exp3_gate_comparison.csv", index=False)

# --- Part 2: threshold sweep on the team's actual gate ----------------------
print("\n\n=== Part 2: confidence threshold sweep (team's agreement gate) ===\n")
from common import gate_agreement_and_confident

thr_rows = []
for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    for seed in SEEDS:
        d = make_split(df, X_A, X_B, seed)
        _, _, h, _ = co_training(d, vec_A, mode="pmvc_wnm", threshold=threshold, seed=seed)
        chosen_counts = [r.get("n_chosen_this_iter", 0) for r in h]
        purities = [r.get("purity_this_iter", np.nan) for r in h if not np.isnan(r.get("purity_this_iter", np.nan))]
        f1_best = max(r["f1"] for r in h)
        thr_rows.append(dict(threshold=threshold, seed=seed, f1_best=f1_best,
                             total_pseudo=sum(chosen_counts),
                             mean_purity=np.mean(purities) if purities else np.nan))
        print(f"  threshold={threshold:.2f} | seed {seed:5d} | f1_best={f1_best:.4f} | "
              f"total_pseudo={sum(chosen_counts):5d} | purity={np.mean(purities) if purities else float('nan'):.4f}")

thr_df = pd.DataFrame(thr_rows)
thr_summary = (thr_df.groupby("threshold")
              .agg(f1_best_mean=("f1_best", "mean"), f1_best_std=("f1_best", "std"),
                   total_pseudo_mean=("total_pseudo", "mean"), mean_purity=("mean_purity", "mean"))
              .round(4).reset_index())

print("\n" + "=" * 90)
print("THRESHOLD SWEEP SUMMARY")
print("=" * 90)
print(thr_summary.to_string(index=False))
thr_summary.to_csv("results/exp3_threshold_sweep.csv", index=False)

print("\n--- Verifying the notebook's purity claim (0.86-0.92 gated vs ~0.70 ungated) ---")
gated_purity = gate_summary[gate_summary.gate.str.contains("pmvc_wnm|agreement_and_confident")]["mean_purity"].values
ungated_purity = gate_summary[gate_summary.gate.str.contains("view_a_only")]["mean_purity"].values
print(f"Agreement-gated purity: {gated_purity}")
print(f"View-A-only purity: {ungated_purity}")

best_thr_row = thr_summary.loc[thr_summary.f1_best_mean.idxmax()]
print(f"\nBest threshold found: {best_thr_row['threshold']} "
      f"(f1={best_thr_row['f1_best_mean']:.4f}, vs team's 0.55)")
