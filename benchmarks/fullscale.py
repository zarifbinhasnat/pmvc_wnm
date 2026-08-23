"""Full-scale run: entire BnSentMix (20K rows), seed=500, 15% test."""
import sys, time, re
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
sys.path.insert(0, ".")
from src.view_a_ngram import build_view_a
from src.view_b_phonetic import build_view_b, make_view_b_classifier
from src.cotraining import PMVCTrainer

SEED = 42
np.random.seed(SEED)
EXT = sys.argv[1] if len(sys.argv) > 1 else "../ML-Banglish-co-training-prototype"
df = pd.read_csv(EXT + "/huggingface bensentMix.csv").rename(
    columns={"Sentence": "text", "Label": "label"}).dropna().reset_index(drop=True)

def clean_text(t):
    t = str(t).lower().strip()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()

df["clean"] = df["text"].apply(clean_text)
df = df[df["clean"].str.len() > 2].reset_index(drop=True)
y = df["label"].values
idx = np.arange(len(df))
pool, test = train_test_split(idx, test_size=0.15, stratify=y, random_state=SEED)
y_test = y[test]
test_texts = df["clean"].iloc[test].tolist()
print(f"FULL SCALE: total={len(df)}  train pool={len(pool)}  test={len(test)}  seed labels=500")
print(f"class balance: {dict(pd.Series(y).value_counts().sort_index())}\n")

results = []
def report(name, y_pred, secs):
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    results.append({"model": name, "macro_f1": round(f1, 4), "accuracy": round(acc, 4), "train_s": round(secs, 1)})
    print(f">>> {name:42s} F1={f1:.4f}  acc={acc:.4f}  ({secs:.1f}s)")
    return y_pred

# seed-only baselines (500 labels, stratified — same seeding as the trainer)
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, train_size=500, random_state=SEED)
seed_local, _ = next(sss.split(np.zeros(len(pool)), y[pool]))
seed_idx = pool[seed_local]

XA, _ = build_view_a(df["clean"].tolist())
XB, _, _ = build_view_b(df["clean"].tolist())

t0 = time.time()
m = CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=SEED)).fit(XA[seed_idx], y[seed_idx])
report("Baseline: LinearSVC char n-gram (500 seed)", m.predict(XA[test]), time.time() - t0)

t0 = time.time()
m = make_view_b_classifier().fit(XB[seed_idx], y[seed_idx])
report("Baseline: LR + BNPC phonetic (500 seed)", m.predict(XB[test]), time.time() - t0)

# full labeled upper bound (what if we had ALL 17K labels?)
t0 = time.time()
m = CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=SEED)).fit(XA[pool], y[pool])
report("Upper bound: LinearSVC with ALL 17K labels", m.predict(XA[test]), time.time() - t0)

train_df = df.iloc[pool].reset_index(drop=True)[["clean", "label"]]

t0 = time.time()
tr = PMVCTrainer(seed_size=500, t_start=999, random_state=SEED)
tr.fit(train_df)
report("Co-training (no noise), fixed voting", tr.predict(test_texts), time.time() - t0)

t0 = time.time()
tr_full = PMVCTrainer(seed_size=500, random_state=SEED)
tr_full.fit(train_df)
pred_full = report("PMVC-WNM full, fixed voting", tr_full.predict(test_texts), time.time() - t0)

print("\nPer-class report for full PMVC-WNM:")
print(classification_report(y_test, pred_full, digits=3))
out = pd.DataFrame(results)
print(out.to_string(index=False))
out.to_csv("benchmarks/fullscale_results.csv", index=False)
