"""
common.py — shared harness that faithfully reproduces the pipeline in
Pantho's `final-project` branch (PMVC_WNM_Banglish_Classification_final.ipynb),
generalized so individual pieces (classifier, vectorizer config, gate logic)
can be swapped for the faculty-question experiments in exp1..exp5.

Ground truth to validate against (teammate's own run, 3 seeds, embedded in
the notebook's saved outputs):
    Single-view SVM       f1_best=0.5533  std=0.0127
    Standard Co-training  f1_best=0.5789  std=0.0100
    PMVC-WNM              f1_best=0.5802  std=0.0084
    working set: 11,788 rows (Negative 3889 / Neutral 4000 / Positive 3899)
    split: L=600, U=8830, test=2358 (per seed)
"""
import re
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import vstack, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, classification_report

DATA_DIR = Path(__file__).parent / "data"
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

RANDOM_STATE = 42
CLASSES = np.array(["Negative", "Neutral", "Positive"])
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

# fixed pipeline constants, matching the teammate's notebook exactly
PER_CLASS = 4000
N_LABELED = 600
N_ITER = 12
STEP = 450
CONF_THRESHOLD = 0.55
SEEDS = [42, 7, 2024]


# ---------------------------------------------------------------------------
# 1. Data loading + merging (verbatim logic from cell 4 of the notebook)
# ---------------------------------------------------------------------------

def _pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_bnsentmix():
    df = pd.read_csv(DATA_DIR / "huggingface bensentMix.csv")
    tcol = _pick_column(df, ["Sentence", "Text", "text"])
    lcol = _pick_column(df, ["Label", "label", "Sentiment"])
    out = df[[tcol, lcol]].rename(columns={tcol: "text", lcol: "raw_label"})
    out["source"] = "bnsentmix"
    return out


def load_b_and_b_80k():
    df = pd.read_csv(DATA_DIR / "Bengali_Banglish_80K_Dataset.csv")
    df = df.drop(columns=[df.columns[0]])  # native-script column, dropped
    tcol = _pick_column(df, ["Banglish", "banglish", "Text", "text"])
    lcol = _pick_column(df, ["Label", "label", "Emotion", "emotion"])
    out = df[[tcol, lcol]].rename(columns={tcol: "text", lcol: "raw_label"})
    out["source"] = "b_and_b_80k"
    return out


def load_binary_sentiment():
    df = pd.read_csv(DATA_DIR / "EnBn_CodeMixed_TwoClass_Sentiment_Balanced_100k.csv")
    tcol = _pick_column(df, ["Code-Mixed-Text", "Text", "text", "Review"])
    lcol = _pick_column(df, ["Sentiment", "sentiment", "Label", "label"])
    out = df[[tcol, lcol]].rename(columns={tcol: "text", lcol: "raw_label"})
    out["source"] = "binary_sentiment"
    return out


NUMERIC_LABEL_MAP = {0: "Positive", 1: "Negative", 2: "Neutral", 3: "Neutral", 4: "Neutral"}
TEXT_LABEL_MAP = {
    "joy": "Positive", "happy": "Positive", "happiness": "Positive", "love": "Positive",
    "positive": "Positive", "pos": "Positive",
    "sadness": "Negative", "sad": "Negative", "anger": "Negative", "angry": "Negative",
    "disgust": "Negative", "fear": "Negative", "hate": "Negative",
    "negative": "Negative", "neg": "Negative",
    "surprise": "Neutral", "surprised": "Neutral", "neutral": "Neutral",
    "mixed": "Neutral", "none": "Neutral",
}


def to_three_class(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().lower()
    if s in ("", "nan"):
        return np.nan
    if s in TEXT_LABEL_MAP:
        return TEXT_LABEL_MAP[s]
    try:
        return NUMERIC_LABEL_MAP.get(int(float(s)), np.nan)
    except (ValueError, TypeError):
        return np.nan


def clean_text(text):
    t = str(text).lower().strip()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


def balanced_sample(df, per_class=PER_CLASS, seed=RANDOM_STATE):
    parts = []
    for label, grp in df.groupby("label"):
        sources = grp["source"].unique()
        per_source = per_class // len(sources)
        picked = pd.concat([
            grp[grp["source"] == s].sample(min(per_source, (grp["source"] == s).sum()),
                                           random_state=seed)
            for s in sources
        ])
        if len(picked) < per_class:
            rest = grp.drop(picked.index)
            picked = pd.concat([picked, rest.sample(min(per_class - len(picked), len(rest)),
                                                    random_state=seed)])
        parts.append(picked)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


_WORKING_SET_CACHE = {}


def build_working_set(per_class=PER_CLASS, seed=RANDOM_STATE):
    """Load, merge, balance, clean — identical to the teammate's cells 4-8.
    Cached: this is the expensive I/O step and every experiment reuses it.
    """
    key = (per_class, seed)
    if key in _WORKING_SET_CACHE:
        return _WORKING_SET_CACHE[key].copy()

    frames = [load_bnsentmix(), load_b_and_b_80k(), load_binary_sentiment()]
    df_all = pd.concat(frames, ignore_index=True)
    df_all["label"] = df_all["raw_label"].map(to_three_class)
    df_all = df_all.dropna(subset=["text", "label"])
    df_all = df_all[df_all["text"].astype(str).str.strip().str.len() > 0]
    df_all = df_all.drop_duplicates(subset=["text"]).reset_index(drop=True)

    df = balanced_sample(df_all, per_class=per_class, seed=seed)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 2].reset_index(drop=True)

    _WORKING_SET_CACHE[key] = df
    return df.copy()


# ---------------------------------------------------------------------------
# 2. View B — the team's Soundex-style phonetic encoder (cells 12-13)
# ---------------------------------------------------------------------------

PHONETIC_RULES = [
    ("chh", "C"), ("kh", "K"), ("gh", "G"), ("ch", "C"), ("jh", "J"), ("th", "T"),
    ("dh", "D"), ("ph", "F"), ("bh", "B"), ("sh", "S"), ("ng", "N"),
    ("k", "K"), ("g", "G"), ("c", "C"), ("j", "J"), ("t", "T"), ("d", "D"),
    ("p", "F"), ("b", "B"), ("v", "B"), ("s", "S"), ("z", "S"), ("m", "M"), ("n", "N"),
    ("r", "R"), ("l", "R"), ("y", "V"), ("w", "B"), ("h", "H"),
    ("a", "V"), ("e", "V"), ("i", "V"), ("o", "V"), ("u", "V"),
]


def phonetic_encode_word_soundex(word):
    i, codes = 0, []
    while i < len(word):
        for pattern, symbol in PHONETIC_RULES:
            if word.startswith(pattern, i):
                codes.append(symbol); i += len(pattern); break
        else:
            i += 1
    if not codes:
        return ""
    collapsed = [codes[0]]
    for s in codes[1:]:
        if s != collapsed[-1]:
            collapsed.append(s)
    return "".join(collapsed)


def phonetic_encode_sentence_soundex(text):
    return " ".join(phonetic_encode_word_soundex(w) for w in text.split())


# ---------------------------------------------------------------------------
# 3. View builders (parameterized — this is what exp2/exp4 sweep)
# ---------------------------------------------------------------------------

def build_view_a(texts, analyzer="char_wb", ngram_range=(3, 4), max_features=5000,
                  sublinear_tf=True, min_df=1, fit=True, vectorizer=None):
    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                     max_features=max_features, sublinear_tf=sublinear_tf,
                                     min_df=min_df)
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)
    return X, vectorizer


def build_view_b_soundex(texts, ngram_range=(1, 2), max_features=5000,
                          analyzer="word", sublinear_tf=True, fit=True, vectorizer=None):
    phon_texts = [phonetic_encode_sentence_soundex(t) for t in texts]
    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                     max_features=max_features, sublinear_tf=sublinear_tf,
                                     token_pattern=r"\S+" if analyzer == "word" else None)
        X = vectorizer.fit_transform(phon_texts)
    else:
        X = vectorizer.transform(phon_texts)
    return X, vectorizer, phon_texts


def build_view_b_bnpc(texts, fit=True, vectorizer=None):
    """The pmvc_wnm repo's BNPC encoder (src/view_b_phonetic.py) — full phoneme
    sequence preserved (not Soundex-collapsed), used as the Q4 candidate fix."""
    from src.view_b_phonetic import build_view_b
    X, vec, encoded = build_view_b(texts, fit=fit, vectorizer=vectorizer)
    return X, vec, encoded


# ---------------------------------------------------------------------------
# 4. Classifiers
# ---------------------------------------------------------------------------

def softmax(z, temperature=2.0):
    z = z / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SVMWithProba:
    """LinearSVC + softmax(margins/T). The team's View A classifier — a speed
    compromise (no Platt scaling) whose confidence calibration is untested
    in the original notebook. exp1 measures that calibration directly."""

    def __init__(self, C=1.0, temperature=2.0, random_state=RANDOM_STATE, class_weight=None):
        self.model = LinearSVC(C=C, random_state=random_state, class_weight=class_weight)
        self.temperature = temperature

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return softmax(self.model.decision_function(X), self.temperature)

    def decision_function(self, X):
        return self.model.decision_function(X)


def make_view_a_classifier(kind="svm_softmax", seed=RANDOM_STATE, **kw):
    """kind in {svm_softmax (team's default), svm_platt, svm_calibrated,
    logreg, complement_nb, sgd_huber, ridge, random_forest, hist_gb}"""
    if kind == "svm_softmax":
        return SVMWithProba(random_state=seed, **kw)
    if kind == "svm_platt":
        from sklearn.svm import SVC
        return SVC(kernel="linear", probability=True, random_state=seed, **kw)
    if kind == "svm_calibrated":
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(LinearSVC(random_state=seed, **kw))
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=kw.pop("C", 5), random_state=seed, **kw)
    if kind == "complement_nb":
        from sklearn.naive_bayes import ComplementNB
        return ComplementNB(**kw)
    if kind == "sgd_huber":
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss="modified_huber", random_state=seed, **kw)
    if kind == "ridge":
        from sklearn.linear_model import RidgeClassifier
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(RidgeClassifier(random_state=seed, **kw))
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=seed, **kw)
    if kind == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed, **kw)
    raise ValueError(kind)


def make_view_b_classifier(kind="logreg", seed=RANDOM_STATE, **kw):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=kw.pop("C", 5), random_state=seed, **kw)
    if kind == "svm_softmax":
        return SVMWithProba(random_state=seed, **kw)
    if kind == "complement_nb":
        from sklearn.naive_bayes import ComplementNB
        return ComplementNB(**kw)
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=seed, **kw)
    if kind == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed, **kw)
    raise ValueError(kind)


def score(y_true, y_pred):
    return dict(f1=f1_score(y_true, y_pred, average="macro"),
                acc=accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# 5. Split protocol (cell 15, verbatim)
# ---------------------------------------------------------------------------

def make_split(df, X_A, X_B, seed, n_labeled=N_LABELED, test_size=0.2):
    y_all = df["label"].values
    texts_all = df["clean_text"].values
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, stratify=y_all,
                                           random_state=seed)
    idx_L, idx_U = train_test_split(idx_train, train_size=n_labeled,
                                    stratify=y_all[idx_train], random_state=seed)
    return dict(
        idx_L=idx_L, idx_U=idx_U, idx_test=idx_test,
        XA_L=X_A[idx_L], XB_L=X_B[idx_L], y_L=y_all[idx_L], txt_L=texts_all[idx_L],
        XA_U=X_A[idx_U], XB_U=X_B[idx_U], y_U_hidden=y_all[idx_U],
        XA_T=X_A[idx_test], XB_T=X_B[idx_test], y_T=y_all[idx_test],
        XA_train=X_A[idx_train], XB_train=X_B[idx_train], y_train=y_all[idx_train],
    )


# ---------------------------------------------------------------------------
# 6. Method 1 — single-view baseline (cell 19)
# ---------------------------------------------------------------------------

def run_single_view(d, view="A", clf_kind=None, seed=RANDOM_STATE, **clf_kw):
    X_L = d["XA_L"] if view == "A" else d["XB_L"]
    X_T = d["XA_T"] if view == "A" else d["XB_T"]
    maker = make_view_a_classifier if view == "A" else make_view_b_classifier
    kind = clf_kind or ("svm_softmax" if view == "A" else "logreg")
    clf = maker(kind=kind, seed=seed, **clf_kw)
    t0 = time.time()
    clf.fit(X_L, d["y_L"])
    fit_s = time.time() - t0
    pred = clf.predict(X_T)
    s = score(d["y_T"], pred)
    s["fit_s"] = fit_s
    s["n_features"] = X_L.shape[1]
    return clf, s, pred


# ---------------------------------------------------------------------------
# 7. Gate variants for exp3 (generalizes cell 21's "eligible/ranking" step)
# ---------------------------------------------------------------------------

def gate_view_a_only(pred_A, pred_B, conf_A, conf_B, threshold):
    """Standard co-training: view A confidence alone."""
    eligible = np.where(conf_A >= threshold)[0]
    return eligible, conf_A[eligible]


def gate_agreement_and_confident(pred_A, pred_B, conf_A, conf_B, threshold):
    """The team's PMVC-WNM gate: both views agree AND both are confident."""
    eligible = np.where((pred_A == pred_B) & (conf_A >= threshold) & (conf_B >= threshold))[0]
    return eligible, conf_A[eligible] * conf_B[eligible]


def gate_agreement_no_threshold(pred_A, pred_B, conf_A, conf_B, threshold):
    """Agreement only, no confidence floor at all."""
    eligible = np.where(pred_A == pred_B)[0]
    return eligible, conf_A[eligible] * conf_B[eligible]


def gate_or_union(pred_A, pred_B, conf_A, conf_B, threshold):
    """Either view confident qualifies (a looser variant)."""
    eligible = np.where((conf_A >= threshold) | (conf_B >= threshold))[0]
    ranking = np.maximum(conf_A[eligible], conf_B[eligible])
    return eligible, ranking

def gate_agreement_or_one_very_confident(pred_A, pred_B, conf_A, conf_B, threshold,
                                          very_confident=0.90):
    agree_ok = (pred_A == pred_B) & (conf_A >= threshold) & (conf_B >= threshold)
    solo_ok = (conf_A >= very_confident) | (conf_B >= very_confident)
    eligible = np.where(agree_ok | solo_ok)[0]
    ranking = np.maximum(conf_A[eligible], conf_B[eligible])
    return eligible, ranking


GATES = {
    "view_a_only (standard)": gate_view_a_only,
    "agreement_and_confident (pmvc_wnm)": gate_agreement_and_confident,
    "agreement_no_threshold": gate_agreement_no_threshold,
    "or_union": gate_or_union,
    "agreement_or_one_very_confident": gate_agreement_or_one_very_confident,
}


# ---------------------------------------------------------------------------
# 8. Phonetic noise injection (cell 21, verbatim)
# ---------------------------------------------------------------------------

PHONETIC_VARIANTS = [
    ("bh", "v"), ("v", "bh"), ("sh", "s"), ("s", "sh"), ("ch", "c"), ("c", "ch"),
    ("kh", "k"), ("k", "kh"), ("gh", "g"), ("th", "t"), ("ph", "f"), ("jh", "j"),
    ("o", "u"), ("i", "e"),
]


def inject_phonetic_noise(text, rng, prob=0.35):
    out = []
    for w in text.split():
        if rng.random() < prob:
            variants = PHONETIC_VARIANTS[:]
            rng.shuffle(variants)
            for a, b in variants:
                if a in w:
                    w = w.replace(a, b, 1); break
        out.append(w)
    return " ".join(out)


# ---------------------------------------------------------------------------
# 9. Generalized co-training loop (cell 20-21, with pluggable gate + view
#    classifiers so exp3/exp4 can vary one thing at a time)
# ---------------------------------------------------------------------------

def co_training(d, char_vectorizer, mode="standard", gate_fn=None, use_wnm=None,
                 n_iter=N_ITER, step=STEP, threshold=CONF_THRESHOLD, seed=RANDOM_STATE,
                 view_a_kind="svm_softmax", view_b_kind="logreg",
                 view_a_kw=None, view_b_kw=None, log_per_class=False):
    view_a_kw = view_a_kw or {}
    view_b_kw = view_b_kw or {}
    if use_wnm is None:
        use_wnm = (mode == "pmvc_wnm")
    if gate_fn is None:
        gate_fn = gate_agreement_and_confident if use_wnm else gate_view_a_only

    y_L = list(d["y_L"])
    f_A = make_view_a_classifier(kind=view_a_kind, seed=seed, **view_a_kw)
    f_B = make_view_b_classifier(kind=view_b_kind, seed=seed, **view_b_kw)
    XA_cur, XB_cur, y_cur = d["XA_L"], d["XB_L"], list(y_L)
    weights = [1.0] * len(y_L)

    transitions = np.ones((3, 3))
    rng = random.Random(seed)
    history, purity = [], np.nan

    for it in range(n_iter):
        if use_wnm and it > 0:
            noisy = [inject_phonetic_noise(t, rng) for t in d["txt_L"]]
            f_A.fit(vstack([XA_cur, char_vectorizer.transform(noisy)]), y_cur + y_L,
                    sample_weight=np.array(weights + [0.5] * len(y_L)))
        else:
            try:
                f_A.fit(XA_cur, y_cur, sample_weight=np.array(weights))
            except TypeError:
                f_A.fit(XA_cur, y_cur)
        f_B.fit(XB_cur, y_cur)

        proba_A_test = f_A.predict_proba(d["XA_T"])
        proba_B_test = f_B.predict_proba(d["XB_T"])
        ensemble = CLASSES[(proba_A_test + proba_B_test).argmax(axis=1)]
        s = score(d["y_T"], ensemble)
        row = dict(iteration=it, f1=s["f1"], acc=s["acc"], n_train=len(y_cur), purity=purity,
                   n_chosen=(0 if it == 0 else len(chosen)) if it > 0 else 0)
        history.append(row)

        proba_A = f_A.predict_proba(d["XA_U"])
        proba_B = f_B.predict_proba(d["XB_U"])
        pred_A, pred_B = CLASSES[proba_A.argmax(1)], CLASSES[proba_B.argmax(1)]
        conf_A, conf_B = proba_A.max(1), proba_B.max(1)

        for a, b in zip(pred_A, pred_B):
            transitions[CLASS_IDX[a], CLASS_IDX[b]] += 1
        trans_norm = transitions / transitions.sum(axis=0, keepdims=True)

        eligible, ranking = gate_fn(pred_A, pred_B, conf_A, conf_B, threshold)

        budget = min(step * (it + 1), len(d["y_U_hidden"]))
        labels_of_eligible = pred_A[eligible]
        chosen = []
        for c in CLASSES:
            in_class = np.where(labels_of_eligible == c)[0]
            in_class = in_class[np.argsort(-ranking[in_class])][:budget // 3]
            chosen.extend(eligible[in_class])
        chosen = np.array(chosen, dtype=int)

        if len(chosen) == 0:
            break

        pseudo = list(pred_A[chosen])
        purity = accuracy_score(d["y_U_hidden"][chosen], pseudo)
        if log_per_class:
            history[-1]["class_dist"] = {c: int((np.array(pseudo) == c).sum()) for c in CLASSES}
        history[-1]["n_chosen_this_iter"] = len(chosen)
        history[-1]["purity_this_iter"] = purity

        XA_cur = vstack([d["XA_L"], d["XA_U"][chosen]])
        XB_cur = vstack([d["XB_L"], d["XB_U"][chosen]])
        y_cur = y_L + pseudo

        if use_wnm:
            weights = [1.0] * len(y_L) + [
                float(conf_A[i] * conf_B[i] * trans_norm[CLASS_IDX[l], CLASS_IDX[l]])
                for i, l in zip(chosen, pseudo)
            ]
        else:
            weights = [1.0] * len(y_cur)

    return f_A, f_B, history, transitions


def summarize_over_seeds(histories, seeds=SEEDS):
    rows = []
    for si, h in enumerate(histories):
        for r in h:
            rows.append({**{k: v for k, v in r.items() if k != "class_dist"}, "seed": seeds[si]})
    c = pd.DataFrame(rows)
    return dict(
        f1_iter0=c[c.iteration == 0]["f1"].mean(),
        f1_best_mean=c.groupby("seed")["f1"].max().mean(),
        f1_best_std=c.groupby("seed")["f1"].max().std(),
        acc_best_mean=c.groupby("seed")["acc"].max().mean(),
    ), c
