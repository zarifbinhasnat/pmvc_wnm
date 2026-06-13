"""
view_b_phonetic.py
BNPC — Banglish Numeric Phonetic Code (View B)

Number-only phonetic encoding for Banglish text, designed for the
PMVC-WNM co-training framework. Pure classical ML stack:
rule-based deterministic encoder + TF-IDF + Logistic Regression.
No deep learning anywhere.

Design principle
----------------
A phonetic code is useful iff it satisfies two conditions:
  C1. Spelling variants of the SAME word map to the SAME code.
  C2. DIFFERENT words map to DIFFERENT codes.
The earlier whole-word -> single-letter acoustic-class encoder satisfied
C1 trivially and destroyed C2, which is why View B carried no signal
(F1 0.113). BNPC keeps the full phoneme sequence, so word identity
survives while phonologically equivalent spellings collapse.

Code space
----------
All codes are two-digit numbers (unambiguous concatenation):

  Vowel classes
    01  a, aa                      (inherent/long a)
    02  e, i, ee, ii, y            (front vowels; y treated as vowel)
    03  o, u, oo                   (back vowels; o/u free variation)

  Consonant classes
    10  b, bh, v, w                (voy / bhoy / boy all -> 10)
    11  p, ph, f
    12  t, th, tt, tth             (dental + retroflex, any aspiration)
    13  d, dh
    14  k, kh, q, ck, hard c
    15  g, gh
    16  ch, chh, soft c (before e/i)
    17  j, jh, z                   (jhamela / jamela / zamela)
    18  s, sh, ss
    19  m
    20  n, nn
    21  l
    22  r
    23  h (standalone, not part of a digraph)

  00  word boundary marker (keeps phoneme n-grams within words)
  x -> 14 18 (ks)

Granularity dials (ablation variables for the thesis)
-----------------------------------------------------
  merge_ch_s : collapse 16 -> 18.  Catches the very frequent
               progressive-tense variation (korchi/korsi, asche/asse,
               achhe/ase) at the cost of merging ch- and s- words.
  merge_a_o  : collapse 03 -> 01.  Catches inherent-vowel variation
               (kotha/katha, pagol/pagal) at the cost of merging
               a/o minimal pairs (mon/man).
Default: both OFF. Run the ablation with all four flag combinations
to demonstrate empirically that encoding granularity is the critical
design variable.

Normalization rules
-------------------
  - lowercase; strip non-alphabetic chars
  - collapse 3+ repeated letters (khubbb -> khub, valooo -> valo)
  - longest-match digraph tokenization (chh > bh,ch,sh,th,kh,gh,jh,ph,dh,ck)
  - dedupe consecutive identical codes (dukkho == dukho, naa == na)

Known out-of-scope distortions (by design)
------------------------------------------
Vowel-dropped SMS forms (vlo, hbe, tnx) and abbreviations are NOT
phonological substitutions; they are deletions. They are the noise
model's job (WNM), not the phonetic view's. This division of labor
is a defensible design statement, not a gap.
"""

import re
from collections import defaultdict

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------
# 1. Encoder
# --------------------------------------------------------------------------

VOWELS = {'a': '01', 'e': '02', 'i': '02', 'y': '02', 'o': '03', 'u': '03'}

DIGRAPHS = {
    'chh': '16',
    'bh': '10', 'ph': '11', 'th': '12', 'dh': '13', 'kh': '14',
    'gh': '15', 'ch': '16', 'jh': '17', 'sh': '18', 'ck': '14',
}

SINGLES = {
    'b': '10', 'v': '10', 'w': '10',
    'p': '11', 'f': '11',
    't': '12', 'd': '13',
    'k': '14', 'q': '14',
    'g': '15',
    'j': '17', 'z': '17',
    's': '18',
    'm': '19', 'n': '20', 'l': '21', 'r': '22', 'h': '23',
}

TOKEN_RE = re.compile(r"[a-zA-Z]+")


def encode_word(word, merge_ch_s=False, merge_a_o=False):
    """Encode one word into a list of two-digit phoneme codes."""
    w = word.lower()
    w = re.sub(r'(.)\1{2,}', r'\1', w)      # collapse 3+ repeats
    w = re.sub(r'[^a-z]', '', w)
    codes, i, n = [], 0, len(w)
    while i < n:
        seg3 = w[i:i + 3]
        if seg3 in DIGRAPHS:
            codes.append(DIGRAPHS[seg3]); i += 3; continue
        seg2 = w[i:i + 2]
        if seg2 in DIGRAPHS:
            codes.append(DIGRAPHS[seg2]); i += 2; continue
        ch = w[i]
        if ch == 'c':
            nxt = w[i + 1] if i + 1 < n else ''
            codes.append('16' if nxt in ('e', 'i') else '14')
        elif ch == 'x':
            codes.extend(['14', '18'])
        elif ch in VOWELS:
            codes.append(VOWELS[ch])
        elif ch in SINGLES:
            codes.append(SINGLES[ch])
        i += 1
    if merge_ch_s:
        codes = ['18' if c == '16' else c for c in codes]
    if merge_a_o:
        codes = ['01' if c == '03' else c for c in codes]
    out = []
    for c in codes:                          # dedupe consecutive identicals
        if not out or out[-1] != c:
            out.append(c)
    return out


def encode_text(text, merge_ch_s=False, merge_a_o=False):
    """
    Returns two parallel numeric representations of a sentence:
      phon : space-separated phoneme codes with 00 word boundaries
             -> feeds sub-word (phoneme n-gram) features
      word : space-separated concatenated word codes
             -> feeds canonical word-identity features
    """
    phon_tokens, word_tokens = [], []
    for w in TOKEN_RE.findall(text):
        codes = encode_word(w, merge_ch_s, merge_a_o)
        if codes:
            phon_tokens.extend(codes)
            phon_tokens.append('00')
            word_tokens.append(''.join(codes))
    return ' '.join(phon_tokens), ' '.join(word_tokens)


# --------------------------------------------------------------------------
# 2. View B feature extractor + classifier
# --------------------------------------------------------------------------

class PhoneticView:
    """
    Drop-in replacement for the old phonetic vectorizer.
    Two TF-IDF blocks stacked:
      block 1: phoneme-level 1-3 grams (robust to within-word distortion)
      block 2: whole-word canonical codes, 1-2 grams (word identity + order)
    """

    def __init__(self, merge_ch_s=False, merge_a_o=False):
        self.merge_ch_s = merge_ch_s
        self.merge_a_o = merge_a_o
        self.v_phon = TfidfVectorizer(token_pattern=r'\S+', ngram_range=(1, 3),
                                      min_df=2, sublinear_tf=True)
        self.v_word = TfidfVectorizer(token_pattern=r'\S+', ngram_range=(1, 2),
                                      min_df=2, sublinear_tf=True)

    def _encode(self, texts):
        phon, word = [], []
        for t in texts:
            p, w = encode_text(t, self.merge_ch_s, self.merge_a_o)
            phon.append(p)
            word.append(w)
        return phon, word

    def fit_transform(self, texts):
        p, w = self._encode(texts)
        return hstack([self.v_phon.fit_transform(p),
                       self.v_word.fit_transform(w)]).tocsr()

    def transform(self, texts):
        p, w = self._encode(texts)
        return hstack([self.v_phon.transform(p),
                       self.v_word.transform(w)]).tocsr()


def make_view_b_classifier():
    """
    Logistic Regression, NOT RandomForest.
    RF performs poorly on sparse high-dimensional TF-IDF (axis-aligned
    splits over mostly-zero features rarely sample informative dims),
    which accounted for a large share of the original 0.113.
    LR also gives calibrated predict_proba for the co-training
    confidence threshold.
    """
    return LogisticRegression(class_weight='balanced', C=2.0,
                              max_iter=2000, n_jobs=-1)


# --------------------------------------------------------------------------
# 3. Co-training framework compatibility layer
# --------------------------------------------------------------------------

def get_phonetic_code(token, merge_ch_s=False, merge_a_o=False):
    """
    Canonical phonetic code for a single token (word-identity view).
    Used by the noise model to group spelling variants of the same
    word under one key.
    """
    return ''.join(encode_word(token, merge_ch_s, merge_a_o))


def build_view_b(texts, fit=True, vectorizer=None):
    """
    View B: BNPC phonetic TF-IDF representation.
    Mirrors build_view_a's (X, vectorizer[, encoded]) signature so it
    drops into the co-training pipeline unchanged.
    """
    if fit or vectorizer is None:
        vectorizer = PhoneticView()
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)

    encoded = [encode_text(t)[1] for t in texts]
    return X, vectorizer, encoded


# --------------------------------------------------------------------------
# 4. Feasibility validation (run BEFORE any co-training re-run)
# --------------------------------------------------------------------------

VARIANT_GROUPS = [
    ['bhalo', 'valo', 'balo'],
    ['bhoy', 'voy', 'boy', 'bhoi', 'voi'],
    ['kharap', 'karap'],
    ['khushi', 'kushi', 'khusi'],
    ['dukkho', 'dukho'],
    ['kosto', 'koshto'],
    ['shanti', 'santi'],
    ['jhamela', 'jamela', 'zamela'],
    ['hashi', 'hasi'],
    ['raag', 'rag'],
    ['chinta', 'cinta'],
    ['amar', 'aamar'],
    ['tumi', 'tomi'],
    ['bishonno', 'bisonno'],
    ['biye', 'bie'],
    ['khub', 'khob', 'khuub'],
    ['lagche', 'lagchhe'],
    # flag-dependent groups (merge only with merge_ch_s=True):
    ['achhe', 'ache', 'ase'],
    ['korchi', 'korsi'],
    # flag-dependent groups (merge only with merge_a_o=True):
    ['kotha', 'katha'],
    ['pagol', 'pagal'],
]


def variant_merge_report(groups=VARIANT_GROUPS, **flags):
    """Target: >= 80% of groups fully merged under chosen flags."""
    merged = 0
    for g in groups:
        codes = {''.join(encode_word(w, **flags)) for w in g}
        ok = len(codes) == 1
        merged += ok
        print(('MERGED  ' if ok else 'SPLIT   ') + str(g) + '  ->  ' + str(sorted(codes)))
    print(f'\n{merged}/{len(groups)} groups fully merged '
          f'({100 * merged / len(groups):.0f}%)  flags={flags}')
    return merged / len(groups)


def vocab_compression(texts, **flags):
    """
    Healthy range ~1.3-2.5x.
    ~1.0  -> encoding too fine, no normalization happening.
    >10x  -> encoding too coarse, word identity being destroyed
             (the v1 single-letter scheme lives here).
    """
    raw, enc = set(), set()
    for t in texts:
        for w in TOKEN_RE.findall(t.lower()):
            raw.add(w)
            c = ''.join(encode_word(w, **flags))
            if c:
                enc.add(c)
    ratio = len(raw) / max(len(enc), 1)
    print(f'raw vocab {len(raw)}  |  encoded vocab {len(enc)}  |  '
          f'compression {ratio:.2f}x  flags={flags}')
    return ratio


def collision_report(texts, top=25, **flags):
    """
    Inspect the largest collision buckets manually. Buckets should
    contain spelling variants of one word; if unrelated words dominate,
    the encoding is over-merging. Target: < 20% spurious buckets in a
    random sample of 50.
    """
    buckets = defaultdict(set)
    for t in texts:
        for w in TOKEN_RE.findall(t.lower()):
            c = ''.join(encode_word(w, **flags))
            if c:
                buckets[c].add(w)
    collided = sorted((b for b in buckets.values() if len(b) > 1),
                      key=len, reverse=True)
    print(f'{len(collided)} codes cover >1 surface form; largest {top}:')
    for b in collided[:top]:
        print('   ' + ', '.join(sorted(b)))
    return collided


def complementarity(y_true, pred_a, pred_b):
    """
    Fraction of test samples where View B is correct and View A is wrong.
    This — not standalone F1 — is the quantity that determines whether
    View B can add anything through co-training. Target: >= 5-8%.
    """
    import numpy as np
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    frac = float(((b == y) & (a != y)).mean())
    print(f'View B correct where View A wrong: {100 * frac:.1f}% of test set')
    return frac


# --------------------------------------------------------------------------
# 5. Smoke test
# --------------------------------------------------------------------------

if __name__ == '__main__':
    print('--- default flags ---')
    variant_merge_report()
    print('\n--- merge_ch_s + merge_a_o ---')
    variant_merge_report(merge_ch_s=True, merge_a_o=True)

    demo = ['amar khub bhoy lagche', 'amar khob voy lagse',
            'tumi onek bhalo', 'tomi onek valo']
    print('\nencodings:')
    for s in demo:
        print(f'  {s!r:38s} -> {encode_text(s)[1]}')
