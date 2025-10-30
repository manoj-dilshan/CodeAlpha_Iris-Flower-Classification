from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_model(name: str, random_state: int = 42, **kwargs):
    """
    Factory for supported models.
    """
    key = name.lower().strip()
    if key in ("logreg", "logistic", "logistic_regression"):
        return LogisticRegression(
            max_iter=1000, random_state=random_state, n_jobs=None, **kwargs
        )
    elif key in ("svm", "svc"):
        # probability=True enables predict_proba (useful for outputs)
        return SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state, **kwargs
        )
    elif key in ("rf", "random_forest", "randomforest"):
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1, **kwargs
        )
    elif key in ("knn", "k-nearest", "knearest"):
        return KNeighborsClassifier(n_neighbors=7, weights="distance", **kwargs)
    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: logistic_regression | svm | random_forest | knn"
        )