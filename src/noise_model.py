import numpy as np
from collections import defaultdict
from src.view_b_phonetic import get_phonetic_code


class NoiseModel:
    """
    Weakly supervised noise model.
    Learns spelling-to-pronunciation distortions from
    inter-view disagreement cases, then injects them
    back into training to improve orthographic robustness.
    """

    def __init__(self, injection_rate: float = 0.20):
        self.injection_rate = injection_rate
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = {}
        self.fitted = False

    def update(self, texts: list):
        """
        Add disagreement examples to the transition count matrix.
        Called each iteration when f_A and f_B disagree.
        """
        for text in texts:
            tokens = text.split()
            for token in tokens:
                ph_code = get_phonetic_code(token)
                self.transition_counts[ph_code][token] += 1

    def fit(self):
        """
        Compute conditional probability matrix from counts:
        P(C_observed | P_true) = Count(C_observed ∩ P_true)
                                 / Σ Count(Ck ∩ P_true)
        """
        self.transition_probs = {}
        for ph_code, spellings in self.transition_counts.items():
            total = sum(spellings.values())
            if total > 0:
                self.transition_probs[ph_code] = {
                    sp: cnt / total
                    for sp, cnt in spellings.items()
                }
        self.fitted = True
        print(f"Noise model fitted: {len(self.transition_probs)} phonetic codes tracked")

    def inject(self, text: str) -> str:
        """
        Corrupt 20% of tokens in a text using learned
        spelling variant probabilities.
        Only applied to View A (orthographic) training data.
        """
        if not self.fitted:
            return text

        tokens = text.split()
        noisy_tokens = []

        for token in tokens:
            if np.random.random() < self.injection_rate:
                ph_code = get_phonetic_code(token)
                if ph_code in self.transition_probs:
                    variants = list(self.transition_probs[ph_code].keys())
                    weights = list(self.transition_probs[ph_code].values())
                    replacement = np.random.choice(variants, p=weights)
                    noisy_tokens.append(replacement)
                else:
                    noisy_tokens.append(token)
            else:
                noisy_tokens.append(token)

        return ' '.join(noisy_tokens)

    def inject_batch(self, texts: list) -> list:
        """Apply noise injection to a list of texts."""
        return [self.inject(t) for t in texts]

    def get_top_distortions(self, top_n: int = 10) -> dict:
        """
        Return most common spelling distortions per phonetic code.
        Useful for the Spelling Error Matrix visualization.
        """
        result = {}
        for ph_code, spellings in self.transition_probs.items():
            sorted_spellings = sorted(
                spellings.items(), key=lambda x: x[1], reverse=True
            )
            result[ph_code] = sorted_spellings[:top_n]
        return result
