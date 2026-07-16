"""Head-to-head benchmark: local PMVC-WNM (pmvc_wnm) vs external prototype
(Rafat-Pantho/ML-Banglish-co-training-prototype) on the SAME data.

Shared protocol:
  - BnSentMix (20K, 4 classes), stratified 3000-row sample (seed 42)
  - 15% held-out test split, stratified
  - 300 labeled seed samples, rest = unlabeled pool
"""
import sys, time, re, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import vstack

RANDOM_STATE = 42
EXT = sys.argv[1] if len(sys.argv) > 1 else "../ML-Banglish-co-training-prototype"
sys.path.insert(0, ".")

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ---------------- shared data ----------------
df = pd.read_csv(EXT + "/huggingface bensentMix.csv").rename(
    columns={"Sentence": "text", "Label": "label"}).dropna().reset_index(drop=True)
df, _ = train_test_split(df, train_size=3000, stratify=df["label"], random_state=RANDOM_STATE)
df = df.reset_index(drop=True)

def clean_text(text):  # external repo's cleaner (superset of local's)
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()

df["clean"] = df["text"].apply(clean_text)
df = df[df["clean"].str.len() > 2].reset_index(drop=True)

idx_all = np.arange(len(df))
y_full = df["label"].values
idx_pool, idx_test = train_test_split(idx_all, test_size=0.15, stratify=y_full, random_state=RANDOM_STATE)
N_LABELED = 300
idx_L, idx_U = train_test_split(idx_pool, train_size=N_LABELED, stratify=y_full[idx_pool], random_state=RANDOM_STATE)
y_L, y_test, y_U_hidden = y_full[idx_L], y_full[idx_test], y_full[idx_U]
test_texts = df["clean"].iloc[idx_test].tolist()
print(f"sample={len(df)}  L={len(idx_L)}  U={len(idx_U)}  test={len(idx_test)}")

results = []
def report(name, y_pred, secs):
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    results.append({"model": name, "macro_f1": round(f1, 4), "accuracy": round(acc, 4), "train_s": round(secs, 1)})
    print(f"{name:45s} F1={f1:.4f}  acc={acc:.4f}  ({secs:.1f}s)")

# ================= EXTERNAL PIPELINE (verbatim logic) =================
PHONETIC_RULES = [
    ("chh", "J"), ("kh", "G"), ("gh", "G"), ("ch", "J"), ("jh", "J"),
    ("th", "T"), ("dh", "T"), ("ph", "P"), ("bh", "P"), ("sh", "S"), ("ng", "M"),
    ("k", "G"), ("g", "G"), ("c", "J"), ("j", "J"), ("t", "T"), ("d", "T"),
    ("p", "P"), ("b", "P"), ("v", "P"), ("s", "S"), ("z", "S"),
    ("m", "M"), ("n", "M"), ("r", "R"), ("l", "R"),
    ("a", "V"), ("e", "V"), ("i", "V"), ("o", "V"), ("u", "V"),
]
def ext_phon_word(word):
    i, codes = 0, []
    while i < len(word):
        for pat, code in PHONETIC_RULES:
            if word.startswith(pat, i):
                codes.append(code); i += len(pat); break
        else:
            i += 1
    if not codes: return ""
    collapsed = [codes[0]]
    for c in codes[1:]:
        if c != collapsed[-1]: collapsed.append(c)
    return "".join(collapsed[:1] + [c for c in collapsed[1:] if c != "V"])

df["phon_ext"] = df["clean"].apply(lambda t: " ".join(ext_phon_word(w) for w in t.split()))

vA_ext = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=5000, sublinear_tf=True)
XA = vA_ext.fit_transform(df["clean"])
vB_ext = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=3000, sublinear_tf=True)
XB = vB_ext.fit_transform(df["phon_ext"])
XA_L, XA_U, XA_test = XA[idx_L], XA[idx_U], XA[idx_test]
XB_L, XB_U, XB_test = XB[idx_L], XB[idx_U], XB[idx_test]

t0 = time.time()
m = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE).fit(XA_L, y_L)
report("EXT baseline: SVM char n-gram (seed only)", m.predict(XA_test), time.time() - t0)

t0 = time.time()
m = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE).fit(XB_L, y_L)
report("EXT baseline: RF phonetic (seed only)", m.predict(XB_test), time.time() - t0)

def ext_co_training(use_noise_model, n_iterations=10, confidence_threshold=0.6, growth_per_iter=40):
    XA_cur, XB_cur, y_cur = XA_L, XB_L, list(y_L)
    remaining = list(range(XA_U.shape[0]))
    f_A = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    f_B = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    dis = {}
    for it in range(n_iterations):
        if not remaining: break
        f_A.fit(XA_cur, y_cur); f_B.fit(XB_cur, y_cur)
        pa = f_A.predict_proba(XA_U[remaining]); pb = f_B.predict_proba(XB_U[remaining])
        pred_A = f_A.classes_[np.argmax(pa, axis=1)]; pred_B = f_B.classes_[np.argmax(pb, axis=1)]
        conf_A, conf_B = pa.max(axis=1), pb.max(axis=1)
        for x, yb in zip(pred_A, pred_B):
            dis.setdefault(yb, [0, 0])
            dis[yb][0 if x == yb else 1] += 1
        top = set(np.argsort(-conf_A)[:growth_per_iter].tolist()) | set(np.argsort(-conf_B)[:growth_per_iter].tolist())
        chosen = [i for i in top if max(conf_A[i], conf_B[i]) >= confidence_threshold]
        if not chosen: break
        gl = [remaining[i] for i in chosen]
        pl = [pred_A[i] if conf_A[i] >= conf_B[i] else pred_B[i] for i in chosen]
        XA_cur = vstack([XA_cur, XA_U[gl]]); XB_cur = vstack([XB_cur, XB_U[gl]])
        y_cur = y_cur + pl
        remaining = [i for i in remaining if i not in set(gl)]
    if use_noise_model:
        nm = {c: a / max(a + d, 1) for c, (a, d) in dis.items()}
        w = np.ones(len(y_cur))
        for i, lab in enumerate(y_cur):
            if i >= len(y_L):
                w[i] = 0.5 + 0.5 * nm.get(lab, 1.0)
        f_A.fit(XA_cur, y_cur, sample_weight=w)
    return f_A

t0 = time.time()
report("EXT: standard co-training", ext_co_training(False).predict(XA_test), time.time() - t0)
t0 = time.time()
report("EXT: PMVC-WNM (reliability reweight)", ext_co_training(True).predict(XA_test), time.time() - t0)

# ================= LOCAL PIPELINE =================
from src.view_a_ngram import build_view_a
from src.view_b_phonetic import build_view_b, make_view_b_classifier
from src.cotraining import PMVCTrainer

XA_loc, _ = build_view_a(df["clean"].tolist())
XB_loc, _, _ = build_view_b(df["clean"].tolist())

t0 = time.time()
m = CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=RANDOM_STATE)).fit(XA_loc[idx_L], y_L)
report("LOCAL baseline: LinearSVC char n-gram (seed)", m.predict(XA_loc[idx_test]), time.time() - t0)

t0 = time.time()
m = make_view_b_classifier().fit(XB_loc[idx_L], y_L)
report("LOCAL baseline: LR + BNPC phonetic (seed)", m.predict(XB_loc[idx_test]), time.time() - t0)

# complementarity of views at seed scale
predA = CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=RANDOM_STATE)).fit(
    XA_loc[idx_L], y_L).predict(XA_loc[idx_test])
predB = make_view_b_classifier().fit(XB_loc[idx_L], y_L).predict(XB_loc[idx_test])
comp_local = float(((predB == y_test) & (predA != y_test)).mean())
predB_ext = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(XB_L, y_L).predict(XB_test)
predA_ext = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE).fit(XA_L, y_L).predict(XA_test)
comp_ext = float(((predB_ext == y_test) & (predA_ext != y_test)).mean())
print(f"\nView-B complementarity (B right where A wrong): LOCAL BNPC {comp_local:.1%} | EXT Soundex {comp_ext:.1%}")

# local co-training: pass only the train pool as its dataframe
train_df = df.iloc[np.concatenate([idx_L, idx_U])].reset_index(drop=True)[["clean", "label"]]

t0 = time.time()
tr = PMVCTrainer(seed_size=N_LABELED, t_start=999, random_state=RANDOM_STATE)
tr.fit(train_df)
report("LOCAL: co-training (no noise)", tr.predict(test_texts), time.time() - t0)

t0 = time.time()
tr_full = PMVCTrainer(seed_size=N_LABELED, random_state=RANDOM_STATE)
tr_full.fit(train_df)
report("LOCAL: PMVC-WNM full (noise injection)", tr_full.predict(test_texts), time.time() - t0)

print("\n" + pd.DataFrame(results).to_string(index=False))
pd.DataFrame(results).to_csv("benchmarks/results.csv", index=False)
