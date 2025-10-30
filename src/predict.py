from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from data_utils import standardize_iris_columns, IRIS_FEATURES


def parse_args():
    p = argparse.ArgumentParser(description="Predict Iris species for new samples.")
    p.add_argument("--model-path", type=str, required=True, help="Path to saved model .joblib")
    p.add_argument("--input-path", type=str, required=True, help="CSV with sepal_length,sepal_width,petal_length,petal_width")
    p.add_argument("--output-path", type=str, default="outputs/predictions/predictions.csv", help="Where to write predictions CSV")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load model pipeline
    pipe = joblib.load(args.model_path)

    # 2) Load and standardize input
    df_in = pd.read_csv(args.input_path)
    df_in = standardize_iris_columns(df_in)

    missing = [c for c in IRIS_FEATURES if c not in df_in.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}. Found: {list(df_in.columns)}")

    X_new = df_in[IRIS_FEATURES].copy()

    # 3) Predict
    preds = pipe.predict(X_new)

    # 4) Predict probabilities (if supported)
    out = df_in.copy()
    out["predicted_species"] = preds

    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_new)
            classes = pipe.named_steps["model"].classes_ if "model" in pipe.named_steps else pipe.classes_
            for i, cls in enumerate(classes):
                out[f"proba_{cls}"] = proba[:, i]
        except Exception:
            # Some models may not be probability-enabled
            pass

    # 5) Save
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path.resolve()}")


if __name__ == "__main__":
    main()