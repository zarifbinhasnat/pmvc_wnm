"""
model_factory.py — the make_model() classifier factory, extracted into its
own side-effect-free module so exp7 (and anything else) can import it
without triggering exp6's full multi-minute experiment loop, which lives at
module level in exp6_exhaustive_model_zoo.py.
"""


def make_model(kind, seed, dense=False):
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import (LogisticRegression, RidgeClassifier, SGDClassifier,
                                      Perceptron, PassiveAggressiveClassifier)
    from sklearn.naive_bayes import ComplementNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  AdaBoostClassifier, GradientBoostingClassifier,
                                  HistGradientBoostingClassifier, BaggingClassifier)
    from sklearn.neural_network import MLPClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from common import SVMWithProba

    if kind == "linsvc_softmax":
        return SVMWithProba(random_state=seed)
    if kind == "linsvc_calibrated":
        return CalibratedClassifierCV(LinearSVC(random_state=seed))
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=5, random_state=seed)
    if kind == "ridge_calibrated":
        return CalibratedClassifierCV(RidgeClassifier(random_state=seed))
    if kind == "sgd_hinge_calibrated":
        return CalibratedClassifierCV(SGDClassifier(loss="hinge", random_state=seed))
    if kind == "sgd_modified_huber":
        return SGDClassifier(loss="modified_huber", random_state=seed)
    if kind == "perceptron_calibrated":
        return CalibratedClassifierCV(Perceptron(random_state=seed))
    if kind == "passive_aggressive_cal":
        return CalibratedClassifierCV(PassiveAggressiveClassifier(random_state=seed))
    if kind == "complement_nb":
        return ComplementNB()
    if kind == "multinomial_nb":
        return MultinomialNB()
    if kind == "knn_k5":
        return KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
    if kind == "knn_k15":
        return KNeighborsClassifier(n_neighbors=15, metric="cosine", n_jobs=-1)
    if kind == "nearest_centroid":
        return NearestCentroid()
    if kind == "decision_tree":
        return DecisionTreeClassifier(max_depth=20, random_state=seed)
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=50, max_depth=15, random_state=seed, n_jobs=-1)
    if kind == "extra_trees":
        return ExtraTreesClassifier(n_estimators=50, max_depth=15, random_state=seed, n_jobs=-1)
    if kind == "bagging_linsvc":
        return BaggingClassifier(LinearSVC(random_state=seed), n_estimators=20, random_state=seed,
                                 n_jobs=-1)
    if kind == "adaboost":
        return AdaBoostClassifier(n_estimators=50, random_state=seed)
    if kind == "gradient_boosting_capped":
        return GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=seed)
    if kind == "hist_gb_capped":
        return HistGradientBoostingClassifier(max_iter=40, random_state=seed)
    if kind == "mlp_small":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed,
                             early_stopping=True)
    raise ValueError(kind)
