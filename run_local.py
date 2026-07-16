"""
run_local.py — one-command local runner for PMVC-WNM.

Usage:
    python run_local.py data/banglish_80k.csv
    python run_local.py data/banglish_80k.csv --seed-sizes 500,1000,2000 --compare
    python run_local.py data/banglish_80k.csv --seed-size 500 --sample 20000

Loads and cleans the dataset, trains the full PMVC-WNM pipeline, and prints
the held-out macro-F1 / accuracy plus a per-class report. CPU only, no GPU.

--seed-sizes runs a LABEL-EFFICIENCY SWEEP: it repeats the whole comparison
at each seed size in the list, so you can see at what label budget the full
pipeline starts to beat the single-view baselines (and by how much).
"""
import argparse
import glob
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="CSV file OR a directory containing one "
                                     "(e.g. the path kagglehub returned)")
    ap.add_argument("--seed-size", type=int, default=500,
                    help="number of labeled seed examples for a single run "
                         "(ignored if --seed-sizes is given)")
    ap.add_argument("--seed-sizes", type=str, default=None,
                    help="comma-separated list of seed sizes to sweep, e.g. "
                         "'500,1000,2000' — repeats the run at each size so "
                         "you can see where the full pipeline starts beating "
                         "the single-view baselines")
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

    seed_sizes = ([int(s) for s in args.seed_sizes.split(",")]
                  if args.seed_sizes else [args.seed_size])

    # Build View A / View B feature matrices ONCE on the full train pool —
    # reused for every baseline in the sweep so we don't re-vectorize per
    # seed size (only the full PMVC-WNM trainer re-vectorizes internally,
    # since it needs its own fit() call per seed size regardless).
    if args.compare:
        XA, vecA = build_view_a(train_df["clean"].tolist())
        XA_test, _ = build_view_a(test_texts, fit=False, vectorizer=vecA)
        XB, vecB, _ = build_view_b(train_df["clean"].tolist())
        XB_test, _, _ = build_view_b(test_texts, fit=False, vectorizer=vecB)

    sweep_rows = []  # (seed_size, model_name, macro_f1, accuracy)
    last_pred, last_macro_f1, last_acc = None, None, None

    for n in seed_sizes:
        print(f"\n{'='*70}\nSEED SIZE = {n}\n{'='*70}")

        if args.compare:
            seed_local = stratified_seed_indices(y_train_full, n)
            m = CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=42)).fit(
                XA[seed_local], y_train_full[seed_local])
            p = m.predict(XA_test)
            sweep_rows.append((n, "Baseline: View A alone (char n-gram)",
                               f1_score(y_test, p, average="macro"), accuracy_score(y_test, p)))

            m = make_view_b_classifier().fit(XB[seed_local], y_train_full[seed_local])
            p = m.predict(XB_test)
            sweep_rows.append((n, "Baseline: View B alone (BNPC phonetic)",
                               f1_score(y_test, p, average="macro"), accuracy_score(y_test, p)))

        print(f"\nTraining full PMVC-WNM (seed_size={n}) on {len(train_df)} rows, "
              f"testing on {len(test_df)} rows...\n")
        trainer = PMVCTrainer(seed_size=n, random_state=42)
        trainer.fit(train_df)
        y_pred = trainer.predict(test_texts)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        sweep_rows.append((n, "Full PMVC-WNM", macro_f1, acc))
        last_pred, last_macro_f1, last_acc = y_pred, macro_f1, acc

    print("\n" + "=" * 78)
    print("LABEL-EFFICIENCY SWEEP RESULTS")
    print("=" * 78)
    print(f"{'seed_size':>9} | {'model':40s} | {'macro_f1':>8} | {'accuracy':>8}")
    for n, name, f1, acc in sweep_rows:
        print(f"{n:>9} | {name:40s} | {f1:8.4f} | {acc:8.4f}")

    if args.compare:
        print("\nDoes the full pipeline beat BOTH single-view baselines, at each seed size?")
        by_size = {}
        for n, name, f1, acc in sweep_rows:
            by_size.setdefault(n, {})[name] = f1
        for n in seed_sizes:
            row = by_size[n]
            full = row.get("Full PMVC-WNM")
            baselines = [v for k, v in row.items() if k != "Full PMVC-WNM"]
            if full is not None and baselines:
                beats = full > max(baselines)
                mark = "YES  (+{:.4f} over best baseline)".format(full - max(baselines)) if beats \
                    else "no   ({:.4f} below best baseline)".format(max(baselines) - full)
                print(f"  seed_size={n:>6}: {mark}")

    print("\n================ FINAL (largest seed size) PER-CLASS REPORT ================")
    print(classification_report(y_test, last_pred, digits=3))


if __name__ == "__main__":
    main()
