from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .data_utils import IRIS_FEATURES


def build_preprocessor(scale: bool = True):
    """
    Build a ColumnTransformer that scales numeric features if requested.
    """
    if not scale:
        return "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), IRIS_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model, scale: bool = True):
    """
    Create a pipeline that applies preprocessing then the model.
    """
    pre = build_preprocessor(scale)
    if pre == "passthrough":
        return Pipeline(steps=[("model", model)])
    else:
        return Pipeline(steps=[("preprocess", pre), ("model", model)])