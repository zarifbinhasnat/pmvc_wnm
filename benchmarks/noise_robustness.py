import sys, re
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.sparse import vstack
sys.path.insert(0, ".")
from src.view_b_phonetic import build_view_b, make_view_b_classifier
from src.cotraining import PMVCTrainer
from src.evaluate import corrupt_text

seed=42
np.random.seed(seed)
EXT = sys.argv[1] if len(sys.argv) > 1 else "../ML-Banglish-co-training-prototype"
raw = pd.read_csv(EXT + "/huggingface bensentMix.csv").rename(columns={"Sentence":"text","Label":"label"}).dropna().reset_index(drop=True)
def clean_text(t):
    t = str(t).lower().strip()
    t = re.sub(r"http\S+|www\.\S+"," ",t); t = re.sub(r"[^a-z0-9\s]"," ",t)
    t = re.sub(r"(.)\1{2,}",r"\1\1",t); return re.sub(r"\s+"," ",t).strip()
df,_ = train_test_split(raw, train_size=3000, stratify=raw["label"], random_state=seed)
df = df.reset_index(drop=True); df["clean"]=df["text"].apply(clean_text)
df=df[df["clean"].str.len()>2].reset_index(drop=True)
y=df["label"].values; idx=np.arange(len(df))
pool,test = train_test_split(idx,test_size=0.15,stratify=y,random_state=seed)
L,U = train_test_split(pool,train_size=300,stratify=y[pool],random_state=seed)
yL,yT = y[L],y[test]
clean_texts = df["clean"].iloc[test].tolist()
noisy_texts = [corrupt_text(t, rate=0.30) for t in clean_texts]

PH = [("chh","J"),("kh","G"),("gh","G"),("ch","J"),("jh","J"),("th","T"),("dh","T"),("ph","P"),("bh","P"),("sh","S"),("ng","M"),("k","G"),("g","G"),("c","J"),("j","J"),("t","T"),("d","T"),("p","P"),("b","P"),("v","P"),("s","S"),("z","S"),("m","M"),("n","M"),("r","R"),("l","R"),("a","V"),("e","V"),("i","V"),("o","V"),("u","V")]
def extphon(word):
    i,codes=0,[]
    while i<len(word):
        for pat,c in PH:
            if word.startswith(pat,i): codes.append(c); i+=len(pat); break
        else: i+=1
    if not codes: return ""
    col=[codes[0]]
    for c in codes[1:]:
        if c!=col[-1]: col.append(c)
    return "".join(col[:1]+[c for c in col[1:] if c!="V"])
vA = TfidfVectorizer(analyzer="char_wb",ngram_range=(3,4),max_features=5000,sublinear_tf=True).fit(df["clean"])
XA = vA.transform(df["clean"])
f_A = SVC(kernel="linear",probability=True,random_state=seed)
XA_cur, y_cur, remaining = XA[L], list(yL), list(range(len(U)))
df["pext"]=df["clean"].apply(lambda t:" ".join(extphon(w) for w in t.split()))
vB = TfidfVectorizer(analyzer="char_wb",ngram_range=(2,3),max_features=3000,sublinear_tf=True).fit(df["pext"])
XB = vB.transform(df["pext"]); XB_cur = XB[L]
f_B = LogisticRegression(max_iter=1000,random_state=seed)
XAU, XBU = XA[U], XB[U]
for it in range(10):
    if not remaining: break
    f_A.fit(XA_cur,y_cur); f_B.fit(XB_cur,y_cur)
    pa=f_A.predict_proba(XAU[remaining]); pb=f_B.predict_proba(XBU[remaining])
    predA=f_A.classes_[pa.argmax(1)]; predB=f_B.classes_[pb.argmax(1)]
    cA,cB=pa.max(1),pb.max(1)
    top=set(np.argsort(-cA)[:40].tolist())|set(np.argsort(-cB)[:40].tolist())
    ch=[i for i in top if max(cA[i],cB[i])>=0.6]
    if not ch: break
    gl=[remaining[i] for i in ch]; pl=[predA[i] if cA[i]>=cB[i] else predB[i] for i in ch]
    XA_cur=vstack([XA_cur,XAU[gl]]); XB_cur=vstack([XB_cur,XBU[gl]]); y_cur=y_cur+pl
    remaining=[i for i in remaining if i not in set(gl)]
f_A.fit(XA_cur,y_cur)
def ext_pred(texts):
    return f_A.predict(vA.transform(texts))

XBl, vecB, _ = build_view_b(df["clean"].tolist())
bnpc = make_view_b_classifier().fit(XBl[L], yL)
def bnpc_pred(texts):
    return bnpc.predict(vecB.transform(texts))
train_df = df.iloc[np.concatenate([L,U])].reset_index(drop=True)[["clean","label"]]
tr = PMVCTrainer(seed_size=300, random_state=seed); tr.fit(train_df)

f1 = lambda p: f1_score(yT,p,average="macro")
for name, fn in [("EXT PMVC-WNM (f_A char view)", ext_pred),
                 ("LOCAL BNPC LR (seed only)", bnpc_pred),
                 ("LOCAL PMVC-WNM full", tr.predict)]:
    c, n = f1(fn(clean_texts)), f1(fn(noisy_texts))
    print(f"{name:32s} clean={c:.4f}  noisy={n:.4f}  drop={c-n:+.4f} ({(c-n)/c:.1%})")
