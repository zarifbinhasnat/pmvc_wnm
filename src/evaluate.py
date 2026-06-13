import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.view_a_ngram import build_view_a
from src.view_b_phonetic import build_view_b, make_view_b_classifier
from src.cotraining import PMVCTrainer


# ─── Noise robustness test set generator ────────────────────────────────────

SPELLING_VARIANTS = {
    'v': 'bh', 'bh': 'v', 'b': 'bh',
    'ch': 'c',  'c': 'ch',
    'kh': 'k',  'k': 'kh',
    'sh': 's',  's': 'sh',
    'o': 'u',   'u': 'o',
    'a': 'e',   'e': 'a',
    'i': 'ee',  'ee': 'i',
}

def corrupt_text(text: str, rate: float = 0.30) -> str:
    """Artificially corrupt text for noise robustness testing."""
    tokens = text.split()
    corrupted = []
    for token in tokens:
        if np.random.random() < rate:
            for src, tgt in SPELLING_VARIANTS.items():
                if src in token:
                    token = token.replace(src, tgt, 1)
                    break
        corrupted.append(token)
    return ' '.join(corrupted)


# ─── Baseline models ─────────────────────────────────────────────────────────

def run_baseline_tfidf(X_train, y_train, X_test, y_test, name="SVM+TF-IDF"):
    model = CalibratedClassifierCV(LinearSVC(max_iter=2000))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        'model': name,
        'f1': f1_score(y_test, preds, average='macro'),
        'accuracy': accuracy_score(y_test, preds)
    }


# ─── Ablation study ──────────────────────────────────────────────────────────

def run_ablation(df: pd.DataFrame, seed_size: int = 500, test_size: float = 0.2):
    """
    Run full ablation across 6 model configurations.
    Returns results dataframe.
    """
    results = []

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=42
    )

    test_texts = test_df['clean'].tolist()
    y_test = test_df['label'].values

    # shared feature matrices for baselines
    X_A_all, vec_A = build_view_a(df['clean'].tolist())
    X_B_all, vec_B, _ = build_view_b(df['clean'].tolist())

    train_idx = train_df.index.tolist()
    test_idx = test_df.index.tolist()

    seed_idx = train_idx[:seed_size]

    # M1: SVM + TF-IDF word level
    tfidf_word = TfidfVectorizer(max_features=30000, sublinear_tf=True)
    X_word = tfidf_word.fit_transform(df['clean'])
    results.append(run_baseline_tfidf(
        X_word[seed_idx], df['label'][seed_idx],
        X_word[test_idx], y_test,
        name="M1: SVM + TF-IDF (word)"
    ))

    # M2: SVM + char n-gram only
    results.append(run_baseline_tfidf(
        X_A_all[seed_idx], df['label'][seed_idx],
        X_A_all[test_idx], y_test,
        name="M2: SVM + char n-gram"
    ))

    # M3: LR + phonetic only (BNPC)
    lr = make_view_b_classifier()
    lr.fit(X_B_all[seed_idx], df['label'][seed_idx])
    preds = lr.predict(X_B_all[test_idx])
    results.append({
        'model': 'M3: LR + phonetic only (BNPC)',
        'f1': f1_score(y_test, preds, average='macro'),
        'accuracy': accuracy_score(y_test, preds)
    })

    # M4: Co-training without noise model
    trainer_no_noise = PMVCTrainer(seed_size=seed_size, t_start=999)
    trainer_no_noise.fit(train_df.reset_index(drop=True))
    preds = trainer_no_noise.predict(test_texts)
    results.append({
        'model': 'M4: Co-training (no noise)',
        'f1': f1_score(y_test, preds, average='macro'),
        'accuracy': accuracy_score(y_test, preds)
    })

    # M5: Full PMVC-WNM
    trainer_full = PMVCTrainer(seed_size=seed_size)
    trainer_full.fit(train_df.reset_index(drop=True))
    preds = trainer_full.predict(test_texts)
    results.append({
        'model': 'M5: PMVC-WNM (full)',
        'f1': f1_score(y_test, preds, average='macro'),
        'accuracy': accuracy_score(y_test, preds)
    })

    return pd.DataFrame(results), trainer_full


# ─── Learning curve ──────────────────────────────────────────────────────────

def run_learning_curve(df: pd.DataFrame, seed_sizes=[100, 250, 500, 1000, 2000]):
    """F1 vs labeled seed size for PMVC-WNM vs SVM baseline."""
    pmvc_scores = []
    svm_scores  = []

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    test_texts = test_df['clean'].tolist()
    y_test = test_df['label'].values

    X_A_all, vec_A = build_view_a(df['clean'].tolist())
    test_idx = test_df.index.tolist()

    for size in seed_sizes:
        print(f"  Seed size: {size}")
        seed_idx = train_df.index.tolist()[:size]

        # SVM baseline
        svm = CalibratedClassifierCV(LinearSVC(max_iter=2000))
        svm.fit(X_A_all[seed_idx], df['label'][seed_idx])
        preds = svm.predict(X_A_all[test_idx])
        svm_scores.append(f1_score(y_test, preds, average='macro'))

        # PMVC-WNM
        trainer = PMVCTrainer(seed_size=size)
        trainer.fit(train_df.reset_index(drop=True))
        preds = trainer.predict(test_texts)
        pmvc_scores.append(f1_score(y_test, preds, average='macro'))

    return seed_sizes, svm_scores, pmvc_scores


# ─── Noise robustness test ───────────────────────────────────────────────────

def run_noise_robustness(trainer: PMVCTrainer, df: pd.DataFrame):
    """Compare F1 on clean vs artificially corrupted test set."""
    _, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    clean_texts = test_df['clean'].tolist()
    noisy_texts = [corrupt_text(t) for t in clean_texts]
    y_test = test_df['label'].values

    clean_f1 = f1_score(y_test, trainer.predict(clean_texts), average='macro')
    noisy_f1 = f1_score(y_test, trainer.predict(noisy_texts), average='macro')

    print(f"\nNoise Robustness:")
    print(f"  Clean F1:  {clean_f1:.3f}")
    print(f"  Noisy F1:  {noisy_f1:.3f}")
    print(f"  Drop:      {clean_f1 - noisy_f1:.3f} ({(clean_f1-noisy_f1)/clean_f1:.1%})")
    return clean_f1, noisy_f1


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_ablation(results_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#cccccc', '#aaaaaa', '#888888', '#4466cc', '#1133aa']
    bars = ax.barh(results_df['model'], results_df['f1'], color=colors)
    ax.set_xlabel('Macro F1 Score')
    ax.set_title('Ablation Study — PMVC-WNM vs Baselines')
    ax.set_xlim(0.3, 0.8)
    for bar, val in zip(bars, results_df['f1']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('ablation.png', dpi=150)
    plt.show()


def plot_learning_curve(seed_sizes, svm_scores, pmvc_scores):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seed_sizes, svm_scores, 'o--', color='gray', label='SVM + char n-gram')
    ax.plot(seed_sizes, pmvc_scores, 's-', color='#1133aa', linewidth=2, label='PMVC-WNM (full)')
    ax.set_xlabel('Labeled Seed Size')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Learning Curve — Sample Efficiency')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=150)
    plt.show()


def plot_disagree_rate(disagree_rates):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(disagree_rates)+1), [r*100 for r in disagree_rates],
            'o-', color='#cc4422', linewidth=2)
    ax.set_xlabel('Co-training Iteration')
    ax.set_ylabel('Disagreement Rate (%)')
    ax.set_title('Inter-View Disagreement Convergence')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('disagree_rate.png', dpi=150)
    plt.show()
