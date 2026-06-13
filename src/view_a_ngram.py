from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse


def build_view_a(texts, fit=True, vectorizer=None):
    """
    View A: Character 3-gram and 4-gram TF-IDF representation.
    Captures orthographic patterns and spelling variations.
    """
    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 4),
            max_features=50000,
            sublinear_tf=True,
            min_df=2
        )
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)

    return X, vectorizer
