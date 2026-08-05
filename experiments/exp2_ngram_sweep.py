"""
exp2_ngram_sweep.py — answers Q2: "why char (3,4)-grams for View A but
word (1,2)-grams for View B?"

The notebook's own rationale (cell 9/11 markdown): View A is orthography, so
character n-grams catch spelling/typos; View B is already phoneme-normalized,
so "char-level here is useless because the alphabet is ~12 symbols and
everything looks identical" -- hence word-level tokens on the encoded string.

This sweeps both views' n-gram range and vocabulary cap, holding everything
else (classifier, split, seeds) fixed at the team's config, and evaluates
Method 1 (single-view, trained on the 600-row seed only -- the same protocol
used to justify these choices in the first place).
"""
import time
import numpy as np
import pandas as pd

from common import build_working_set, build_view_a, build_view_b_soundex, make_split, run_single_view, SEEDS

df = build_working_set()
texts = df["clean_text"].tolist()

VIEW_A_GRIDS = [
    dict(analyzer="char_wb", ngram_range=(2, 3)),
    dict(analyzer="char_wb", ngram_range=(2, 4)),
    dict(analyzer="char_wb", ngram_range=(3, 4)),   # team's choice
    dict(analyzer="char_wb", ngram_range=(3, 5)),
    dict(analyzer="char_wb", ngram_range=(4, 5)),
    dict(analyzer="char_wb", ngram_range=(2, 5)),
    dict(analyzer="word", ngram_range=(1, 1)),      # word-level control, to show it fails on A
    dict(analyzer="word", ngram_range=(1, 2)),
]
VIEW_B_GRIDS = [
    dict(analyzer="word", ngram_range=(1, 1)),
    dict(analyzer="word", ngram_range=(1, 2)),      # team's choice
    dict(analyzer="word", ngram_range=(1, 3)),
    dict(analyzer="word", ngram_range=(2, 3)),
    dict(analyzer="char_wb", ngram_range=(2, 3)),   # char-level control, to test their claim
    dict(analyzer="char_wb", ngram_range=(3, 4)),
]
MAX_FEATURES = [5000, 20000]


def sweep(view, grids, X_other_builder, is_view_a):
    rows = []
    for cfg in grids:
        for max_feat in MAX_FEATURES:
            for seed in SEEDS:
                t0 = time.time()
                if is_view_a:
                    X, vec = build_view_a(texts, max_features=max_feat, **cfg)
                else:
                    X, vec, _ = build_view_b_soundex(texts, max_features=max_feat, **cfg)
                build_s = time.time() - t0

                # need both views for make_split's shape; reuse a cheap dummy for the other
                if is_view_a:
                    X_A, X_B = X, X_other_builder
                else:
                    X_A, X_B = X_other_builder, X

                d = make_split(df, X_A, X_B, seed)
                _, s, _ = run_single_view(d, view=view, seed=seed)
                vocab = len(vec.vocabulary_) if hasattr(vec, "vocabulary_") else X.shape[1]
                rows.append(dict(view=view, analyzer=cfg["analyzer"], ngram_range=str(cfg["ngram_range"]),
                                 max_features=max_feat, seed=seed, f1=s["f1"], acc=s["acc"],
                                 vocab_used=vocab, build_s=build_s))
                print(f"  view {view} | {cfg['analyzer']:8s} {str(cfg['ngram_range']):8s} "
                      f"| max_feat={max_feat:6d} | seed {seed:5d} | f1={s['f1']:.4f} "
                      f"vocab={vocab}")
    return pd.DataFrame(rows)


# a fixed cheap "other view" placeholder so make_split's dict shape works;
# not used by run_single_view when scoring the view under test.
X_B_fixed, _, _ = build_view_b_soundex(texts, max_features=5000, ngram_range=(1, 2))
X_A_fixed, _ = build_view_a(texts, max_features=5000, ngram_range=(3, 4))

print("=== View A n-gram sweep ===")
rA = sweep("A", VIEW_A_GRIDS, X_B_fixed, is_view_a=True)
print("\n=== View B n-gram sweep ===")
rB = sweep("B", VIEW_B_GRIDS, X_A_fixed, is_view_a=False)

all_rows = pd.concat([rA, rB], ignore_index=True)
summary = (all_rows.groupby(["view", "analyzer", "ngram_range", "max_features"])
           .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"), vocab_used=("vocab_used", "first"))
           .round(4).reset_index())
summary = summary.sort_values(["view", "f1_mean"], ascending=[True, False])

print("\n" + "=" * 90)
print("SUMMARY (mean over 3 seeds)")
print("=" * 90)
print(summary.to_string(index=False))
summary.to_csv("results/exp2_ngram_sweep.csv", index=False)
print("\nSaved to results/exp2_ngram_sweep.csv")

print("\n--- Does word-level actually fail on View A (spelling)? ---")
a_char = summary[(summary.view == "A") & (summary.analyzer == "char_wb")]["f1_mean"].max()
a_word = summary[(summary.view == "A") & (summary.analyzer == "word")]["f1_mean"].max()
print(f"Best char_wb on View A: {a_char:.4f}   Best word on View A: {a_word:.4f}   "
      f"gap={a_char-a_word:+.4f}")

print("\n--- Does char-level actually fail on View B (already phoneme-normalized)? ---")
b_word = summary[(summary.view == "B") & (summary.analyzer == "word")]["f1_mean"].max()
b_char = summary[(summary.view == "B") & (summary.analyzer == "char_wb")]["f1_mean"].max()
print(f"Best word on View B: {b_word:.4f}   Best char_wb on View B: {b_char:.4f}   "
      f"gap={b_word-b_char:+.4f}")
