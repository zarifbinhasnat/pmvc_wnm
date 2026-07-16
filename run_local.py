"""
run_local.py — one-command local runner for PMVC-WNM.

Usage:
    python run_local.py data/banglish_80k.csv
    python run_local.py data/banglish_80k.csv --seed-size 500 --sample 20000

Loads and cleans the dataset, trains the full PMVC-WNM pipeline, and prints
the held-out macro-F1 / accuracy plus a per-class report. CPU only, no GPU.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="CSV file OR a directory containing one "
                                     "(e.g. the path kagglehub returned)")
    ap.add_argument("--seed-size", type=int, default=500,
                    help="number of labeled seed examples (default 500)")
    ap.add_argument("--sample", type=int, default=None,
                    help="optional: subsample this many rows for a faster run")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="held-out test fraction (default 0.2)")
    ap.add_argument("--compare", action="store_true",
                    help="also fit single-view baselines (View A alone, View B "
                         "alone) on the SAME seed/test split, so you can see "
                         "whether the full pipeline actually beats them on "
                         "this dataset")
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

    results = []

    if args.compare:
        print(f"--compare: fitting single-view baselines on the same "
              f"{args.seed_size}-example seed for a fair comparison...\n")
        sss = StratifiedShuffleSplit(n_splits=1, train_size=args.seed_size, random_state=42)
        seed_local, _ = next(sss.split(np.zeros(len(train_df)), train_df["label"]))

        XA, vecA = build_view_a(train_df["clean"].tolist())
        XA_test, _ = build_view_a(test_texts, fit=False, vectorizer=vecA)
        m = CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=42)).fit(
            XA[seed_local], train_df["label"].values[seed_local])
        p = m.predict(XA_test)
        results.append(("Baseline: View A alone (char n-gram, seed only)",
                        f1_score(y_test, p, average="macro"), accuracy_score(y_test, p)))

        XB, vecB, _ = build_view_b(train_df["clean"].tolist())
        XB_test, _, _ = build_view_b(test_texts, fit=False, vectorizer=vecB)
        m = make_view_b_classifier().fit(XB[seed_local], train_df["label"].values[seed_local])
        p = m.predict(XB_test)
        results.append(("Baseline: View B alone (BNPC phonetic, seed only)",
                        f1_score(y_test, p, average="macro"), accuracy_score(y_test, p)))

    print(f"\nTraining PMVC-WNM (seed_size={args.seed_size}) on {len(train_df)} rows, "
          f"testing on {len(test_df)} rows...\n")

    trainer = PMVCTrainer(seed_size=args.seed_size, random_state=42)
    trainer.fit(train_df)

    y_pred = trainer.predict(test_texts)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    results.append(("Full PMVC-WNM", macro_f1, acc))

    if args.compare:
        print("\n================ COMPARISON (same seed/test split) ================")
        print(f"{'model':52s} {'macro_f1':>9s} {'accuracy':>9s}")
        for name, f1, a in results:
            print(f"{name:52s} {f1:9.4f} {a:9.4f}")

    print("\n================ FULL PMVC-WNM RESULTS ================")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
