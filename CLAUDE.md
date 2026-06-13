# PMVC-WNM Project — Claude Code Instructions

## Project
Phonetic Multi-View Co-training with Weakly Supervised Noise Modeling
for Banglish emotion classification.

## Stack
- Python 3.10+
- scikit-learn, pandas, numpy
- Dataset: Kaggle b-and-b-80k (place CSV in data/)

## Folder roles
- src/preprocess.py       — text cleaning
- src/view_a_ngram.py     — character n-gram TF-IDF (View A)
- src/view_b_phonetic.py  — BNPC numeric phonetic encoding (View B)
- src/cotraining.py       — co-training loop (T=10 iterations)
- src/noise_model.py      — disagreement collection + noise injection
- src/evaluate.py         — ablation table + learning curve + noise robustness
- notebooks/              — Google Colab notebook combining all stages

## Key design decisions
- f_A: LinearSVC on char 3,4-grams
- f_B: Logistic Regression on BNPC phonetic TF-IDF (View B)
- Seed size: 500 stratified samples
- Confidence threshold: 0.75
- Noise injection rate: 20%, starts at iteration t=3
- Ensemble: confidence-weighted vote f_A + f_B

## Do not change
- The BNPC phoneme code tables (VOWELS/DIGRAPHS/SINGLES) in view_b_phonetic.py
- The 20% noise injection rate
- The T=10 iteration count

## What to improve if asked
- Hyperparameter tuning of threshold and K
- BNPC granularity flags (merge_ch_s, merge_a_o) ablation
- Evaluation metrics reporting
