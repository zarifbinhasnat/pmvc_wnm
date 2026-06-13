import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit

from src.view_a_ngram import build_view_a
from src.view_b_phonetic import build_view_b, make_view_b_classifier
from src.noise_model import NoiseModel


class PMVCTrainer:
    """
    PMVC-WNM: Phonetic Multi-View Co-training with
    Weakly Supervised Noise Modeling.

    f_A: LinearSVC on character n-grams         (View A)
    f_B: Logistic Regression on BNPC phonetic codes (View B)
    """

    def __init__(
        self,
        seed_size: int = 500,
        T: int = 10,
        K: int = 100,
        threshold: float = 0.75,
        t_start: int = 3,
        noise_rate: float = 0.20,
        random_state: int = 42
    ):
        self.seed_size = seed_size
        self.T = T
        self.K = K
        self.threshold = threshold
        self.t_start = t_start
        self.noise_rate = noise_rate
        self.random_state = random_state

        # classifiers
        self.f_A = CalibratedClassifierCV(
            LinearSVC(max_iter=3000, random_state=random_state)
        )
        self.f_B = make_view_b_classifier()

        # vectorizers
        self.vec_A = None
        self.vec_B = None

        # noise model
        self.noise_model = NoiseModel(injection_rate=noise_rate)

        # tracking
        self.disagree_rates = []

    def _get_seed_indices(self, labels):
        """Stratified seed sampling."""
        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.seed_size,
            random_state=self.random_state
        )
        seed_idx, _ = next(sss.split(np.zeros(len(labels)), labels))
        return list(seed_idx)

    def fit(self, df: pd.DataFrame):
        """
        Main training entry point.
        df must have columns: clean (text), label
        """
        texts = df['clean'].tolist()
        labels = df['label'].tolist()

        print("Building View A (char n-grams)...")
        X_A, self.vec_A = build_view_a(texts)

        print("Building View B (phonetic codes)...")
        X_B, self.vec_B, _ = build_view_b(texts)

        # stratified seed
        print(f"Sampling {self.seed_size} labeled seed examples...")
        seed_idx = self._get_seed_indices(labels)
        unlabeled_idx = [i for i in range(len(texts)) if i not in seed_idx]

        L_A = list(seed_idx)
        L_B = list(seed_idx)

        labels_array = np.array(labels)

        # initial training
        self.f_A.fit(X_A[L_A], labels_array[L_A])
        self.f_B.fit(X_B[L_B], labels_array[L_B])

        print(f"\nStarting co-training loop (T={self.T})...\n")

        for t in range(self.T):
            if not unlabeled_idx:
                print("Unlabeled pool exhausted.")
                break

            u_idx = np.array(unlabeled_idx)

            # predictions on unlabeled pool
            prob_A = self.f_A.predict_proba(X_A[u_idx])
            prob_B = self.f_B.predict_proba(X_B[u_idx])

            pred_A = self.f_A.predict(X_A[u_idx])
            pred_B = self.f_B.predict(X_B[u_idx])

            conf_A = np.max(prob_A, axis=1)
            conf_B = np.max(prob_B, axis=1)

            # collect disagreements
            disagree_mask = pred_A != pred_B
            disagree_count = disagree_mask.sum()
            disagree_rate = disagree_count / len(u_idx)
            self.disagree_rates.append(disagree_rate)

            disagree_texts = [texts[u_idx[i]] for i in range(len(u_idx)) if disagree_mask[i]]
            self.noise_model.update(disagree_texts)

            print(f"Iter {t+1:2d}/{self.T} | "
                  f"Unlabeled: {len(unlabeled_idx):5d} | "
                  f"Disagree: {disagree_rate:.1%} | "
                  f"L_A: {len(L_A)} | L_B: {len(L_B)}")

            # fit noise model after t_start
            if t >= self.t_start:
                self.noise_model.fit()

            # cross-teaching: f_A teaches f_B
            high_conf_A = np.where(conf_A >= self.threshold)[0]
            if len(high_conf_A) > 0:
                top_k_A = high_conf_A[np.argsort(conf_A[high_conf_A])[-self.K:]]
                new_for_B = [u_idx[i] for i in top_k_A]
                L_B.extend(new_for_B)

            # cross-teaching: f_B teaches f_A
            high_conf_B = np.where(conf_B >= self.threshold)[0]
            if len(high_conf_B) > 0:
                top_k_B = high_conf_B[np.argsort(conf_B[high_conf_B])[-self.K:]]
                new_for_A = [u_idx[i] for i in top_k_B]
                L_A.extend(new_for_A)

            # noise injection into f_A training (after t_start)
            if t >= self.t_start and self.noise_model.fitted:
                noisy_texts = self.noise_model.inject_batch(
                    [texts[i] for i in L_A]
                )
                X_A_noisy, _ = build_view_a(
                    noisy_texts, fit=False, vectorizer=self.vec_A
                )
                self.f_A.fit(X_A_noisy, labels_array[L_A])
            else:
                self.f_A.fit(X_A[L_A], labels_array[L_A])

            self.f_B.fit(X_B[L_B], labels_array[L_B])

            # remove pseudo-labeled from unlabeled pool
            pseudo_labeled = set(
                new_for_A if len(high_conf_A) > 0 else []
            ) | set(
                new_for_B if len(high_conf_B) > 0 else []
            )
            unlabeled_idx = [i for i in unlabeled_idx if i not in pseudo_labeled]

        print("\nCo-training complete.")
        return self

    def predict(self, texts: list) -> np.ndarray:
        """
        Ensemble prediction: confidence-weighted vote.
        f_B wins on high confidence, f_A as fallback.
        """
        X_a, _ = build_view_a(texts, fit=False, vectorizer=self.vec_A)
        X_b, _, _ = build_view_b(texts, fit=False, vectorizer=self.vec_B)

        prob_A = self.f_A.predict_proba(X_a)
        prob_B = self.f_B.predict_proba(X_b)

        pred_A = self.f_A.predict(X_a)
        pred_B = self.f_B.predict(X_b)

        conf_B = np.max(prob_B, axis=1)

        final = np.where(conf_B >= self.threshold, pred_B, pred_A)
        return final

    def predict_proba(self, texts: list) -> np.ndarray:
        """Return averaged class probabilities from both views."""
        X_a, _ = build_view_a(texts, fit=False, vectorizer=self.vec_A)
        X_b, _, _ = build_view_b(texts, fit=False, vectorizer=self.vec_B)

        prob_A = self.f_A.predict_proba(X_a)
        prob_B = self.f_B.predict_proba(X_b)

        return (prob_A + prob_B) / 2
