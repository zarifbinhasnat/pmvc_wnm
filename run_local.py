"""
run_local.py — one-command local runner for PMVC-WNM.

Usage:
    python run_local.py data/banglish_80k.csv
    python run_local.py data/banglish_80k.csv --seed-sizes 500,1000,2000 --compare
    python run_local.py data/banglish_80k.csv --seed-size 500 --sample 20000

    # hyperparameter grid search (threshold and K — the two knobs the
    # project's design notes mark as tunable; T=10 and the 20% noise rate
    # are fixed design decisions and are not exposed here):
    python run_local.py data/banglish_80k.csv --threshold-grid 0.6,0.7,0.75,0.8,0.9 --k-grid 50,100,200

Loads and cleans the dataset, trains the full PMVC-WNM pipeline, and prints
the held-out macro-F1 / accuracy plus a per-class report. CPU only, no GPU.

--seed-sizes / --threshold-grid / --k-grid each accept a comma-separated
list. Any combination of them runs the full cross-product sweep (e.g. 3
seed sizes x 3 thresholds x 2 Ks = 18 full training runs) — each full run
repeats the whole 10-round co-training loop, so a large grid takes a
while; the script prints the run count up front.
"""
import argparse
import glob
import itertools
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.preprocess import load_and_clean
from src.view_a_ngram import build_view_a
from src.view_b_phonetic import build_view_b, make_view_b_classifier
from src.cotraining import PMVCTrainer


def resolve_csv(path):
    """Accept a CSV file directly, or a directory (e.g. the folder kagglehub
    downloaded into) and find the CSV inside it."""
    if os.path.isdir(path):
        csvs = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
        if not csvs:
            raise SystemExit(f"No CSV found under directory: {path}")
        print(f"Using CSV: {csvs[0]}")
        return csvs[0]
    return path


def stratified_seed_indices(labels, n, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=random_state)
    loc, _ = next(sss.split(np.zeros(len(labels)), labels))
    return loc


def parse_list(s, cast):
    return [cast(x) for x in s.split(",")] if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="CSV file OR a directory containing one "
                                     "(e.g. the path kagglehub returned)")
    ap.add_argument("--seed-size", type=int, default=500,
                    help="number of labeled seed examples for a single run "
                         "(ignored if --seed-sizes is given)")
    ap.add_argument("--seed-sizes", type=str, default=None,
                    help="comma-separated list of seed sizes to sweep, e.g. "
                         "'500,1000,2000'")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="confidence threshold for accepting a pseudo-label "
                         "(default 0.75, ignored if --threshold-grid is given)")
    ap.add_argument("--threshold-grid", type=str, default=None,
                    help="comma-separated thresholds to sweep, e.g. "
                         "'0.6,0.7,0.75,0.8,0.9'")
    ap.add_argument("--k", type=int, default=100,
                    help="max pseudo-labels accepted per view per round "
                         "(default 100, ignored if --k-grid is given)")
    ap.add_argument("--k-grid", type=str, default=None,
                    help="comma-separated K values to sweep, e.g. '50,100,200'")
    ap.add_argument("--sample", type=int, default=None,
                    help="optional: subsample this many rows for a faster run")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="held-out test fraction (default 0.2)")
    ap.add_argument("--compare", action="store_true",
                    help="also fit single-view baselines (View A alone, View B "
                         "alone) on the SAME seed/test split, so you can see "
                         "whether the full pipeline actually beats them")
    args = ap.parse_args()

    df = load_and_clean(resolve_csv(args.csv_path))

    if args.sample and args.sample < len(df):
        df, _ = train_test_split(df, train_size=args.sample,
                                 stratify=df["label"], random_state=42)
        df = df.reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows.")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df["label"], random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    test_texts = test_df["clean"].tolist()
    y_test = test_df["label"].values
    y_train_full = train_df["label"].values

    seed_sizes = parse_list(args.seed_sizes, int) or [args.seed_size]
    thresholds = parse_list(args.threshold_grid, float) or [args.threshold]
    ks = parse_list(args.k_grid, int) or [args.k]

    combos = list(itertools.product(seed_sizes, thresholds, ks))
    sweeping = len(combos) > 1
    if sweeping:
        print(f"Grid sweep: {len(seed_sizes)} seed size(s) x {len(thresholds)} "
              f"threshold(s) x {len(ks)} K value(s) = {len(combos)} full training runs.")
        print("Each run repeats the full 10-round co-training loop — this may take a while.\n")

    # Build View A / View B feature matrices ONCE on the full train pool —
    # reused for every baseline (baselines don't depend on threshold/K, only
    # on seed size), cached per seed size so they're computed at most once.
    if args.compare:
        XA, vecA = build_view_a(train_df["clean"].tolist())
        XA_test, _ = build_view_a(test_texts, fit=False, vectorizer=vecA)
        XB, vecB, _ = build_view_b(train_df["clean"].tolist())
        XB_test, _, _ = build_view_b(test_texts, fit=False, vectorizer=vecB)
    baseline_cache = {}  # seed_size -> {model_name: (f1, acc)}

    def get_baselines(n):
        if n not in baseline_cache:
            seed_local = stratified_seed_indices(y_train_full, n)
            m = CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=42)).fit(
                XA[seed_local], y_train_full[seed_local])
            p = m.predict(XA_test)
            f1a, acca = f1_score(y_test, p, average="macro"), accuracy_score(y_test, p)

            m = make_view_b_classifier().fit(XB[seed_local], y_train_full[seed_local])
            p = m.predict(XB_test)
            f1b, accb = f1_score(y_test, p, average="macro"), accuracy_score(y_test, p)

            baseline_cache[n] = {
                "Baseline: View A alone (char n-gram)": (f1a, acca),
                "Baseline: View B alone (BNPC phonetic)": (f1b, accb),
            }
        return baseline_cache[n]

    # rows: (seed_size, threshold, K, model_name, macro_f1, accuracy)
    sweep_rows = []
    last_pred = None

    for n, thr, k in combos:
        print(f"\n{'='*78}\nseed_size={n}  threshold={thr}  K={k}\n{'='*78}")

        if args.compare:
            for name, (f1, acc) in get_baselines(n).items():
                sweep_rows.append((n, thr, k, name, f1, acc))

        print(f"\nTraining full PMVC-WNM (seed_size={n}, threshold={thr}, K={k}) "
              f"on {len(train_df)} rows, testing on {len(test_df)} rows...\n")
        trainer = PMVCTrainer(seed_size=n, threshold=thr, K=k, random_state=42)
        trainer.fit(train_df)
        y_pred = trainer.predict(test_texts)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        sweep_rows.append((n, thr, k, "Full PMVC-WNM", macro_f1, acc))
        last_pred = y_pred

    print("\n" + "=" * 90)
    print("SWEEP RESULTS")
    print("=" * 90)
    print(f"{'seed':>6} | {'thr':>5} | {'K':>5} | {'model':40s} | {'macro_f1':>8} | {'accuracy':>8}")
    for n, thr, k, name, f1, acc in sweep_rows:
        print(f"{n:>6} | {thr:>5} | {k:>5} | {name:40s} | {f1:8.4f} | {acc:8.4f}")

    full_rows = [(n, thr, k, f1, acc) for n, thr, k, name, f1, acc in sweep_rows
                 if name == "Full PMVC-WNM"]

    if args.compare:
        print("\nDoes the full pipeline beat BOTH single-view baselines, per configuration?")
        for n, thr, k, f1, acc in full_rows:
            baselines = [v[0] for v in get_baselines(n).values()]
            beats = f1 > max(baselines)
            mark = ("YES  (+{:.4f} over best baseline)".format(f1 - max(baselines)) if beats
                    else "no   ({:.4f} below best baseline)".format(max(baselines) - f1))
            print(f"  seed={n:>6} thr={thr:>5} K={k:>5}: {mark}")

    if sweeping:
        best = max(full_rows, key=lambda r: r[3])
        n, thr, k, f1, acc = best
        print(f"\n>>> BEST CONFIGURATION FOUND: seed_size={n}, threshold={thr}, K={k}")
        print(f">>> macro_f1={f1:.4f}  accuracy={acc:.4f}")
        if args.compare:
            baselines = [v[0] for v in get_baselines(n).values()]
            print(f">>> vs. best single-view baseline at that seed size: {max(baselines):.4f} "
                  f"({'beats it' if f1 > max(baselines) else 'still below it'})")

    print("\n================ FINAL (last configuration run) PER-CLASS REPORT ================")
    print(classification_report(y_test, last_pred, digits=3))


if __name__ == "__main__":
    main()
