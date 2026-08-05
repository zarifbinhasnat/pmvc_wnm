# experiments/

Evidence pack answering the faculty's questions on the PMVC-WNM presentation.
See `../FINDINGS.md` for the write-up; this folder holds the code and raw
results that back every number in it.

## Setup

The three source CSVs are not committed (large, and one is Kaggle-gated).
Place them in `experiments/data/`:

- `huggingface bensentMix.csv` (BnSentMix)
- `Bengali_Banglish_80K_Dataset.csv` (Kaggle b-and-b-80k)
- `EnBn_CodeMixed_TwoClass_Sentiment_Balanced_100k.csv`

All three are on Pantho's repo, `final-project` branch.

## Run order

```bash
cd experiments
python3 verify_reproduction.py      # MUST pass first -- validates the harness
                                     # against the team's own logged numbers
python3 exp1_classifier_matrix.py   # Q1: SVM+softmax vs LogReg, calibration
python3 exp2_ngram_sweep.py         # Q2: char vs word n-gram ranges
python3 exp3_gate_ablation.py       # Q3: agreement gate, purity/quantity trade-off
python3 exp4_single_view_improve.py # Q4: improvement ladder + BNPC swap
python3 exp5_model_zoo.py           # supplementary: efficiency frontier
```

Each script is standalone, prints its table, and writes to `results/*.csv`.
`common.py` is the shared harness — a faithful, parameterized reproduction
of `PMVC_WNM_Banglish_Classification_final.ipynb`'s pipeline (see its
module docstring for the exact ground-truth numbers it's validated against).

`tables.tex` has booktabs tables ready to paste into the paper, matching its
existing style. Every table has a `\label{}`; add a `\ref{}` in the
surrounding text you write (required by the rubric).

## What's NOT here

No paper prose. Per the course's generative-AI policy, writing the term
paper's text is the team's job — this folder is measurement + tables only.
