from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_utils import load_iris_csv, IRIS_FEATURES, TARGET_COL, train_test_split_stratified
from features import make_pipeline
from models import get_model
from evaluate import compute_metrics, save_metrics, plot_and_save_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Train an Iris classifier.")
    p.add_argument("--data-path", type=str, default="data/Iris.csv", help="Path to Iris CSV or folder containing it.")
    p.add_argument("--model", type=str, default="svm", help="Model: logistic_regression | svm | random_forest | knn")
    p.add_argument("--scale", type=str, default="true", help="Whether to scale features: true/false")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size fraction.")
    p.add_argument("--cv", type=int, default=5, help="CV folds for training set performance estimate.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--out-dir", type=str, default="outputs", help="Output directory.")
    return p.parse_args()


def str2bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y", "t")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    models_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset
    X, y, df = load_iris_csv(args.data_path)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # 3) Build model pipeline
    model = get_model(args.model, random_state=args.random_state)
    pipe = make_pipeline(model, scale=str2bool(args.scale))

    # 4) Cross-validation on train split
    if args.cv and args.cv > 1:
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"[CV] Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    else:
        cv_scores = None

    # 5) Fit on train, evaluate on test
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, average="macro")
    print(f"[Test] Accuracy: {metrics['accuracy']:.3f} | F1_macro: {metrics['f1_macro']:.3f}")

    # 6) Save artifacts
    model_path = models_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    # Class labels in the trained model
    classes = list(pipe.named_steps["model"].classes_) if "model" in pipe.named_steps else list(pipe.classes_)
    # Fallback: Some pipelines name steps differently; but ours is consistent.

    # Save metrics (plus CV stats)
    if cv_scores is not None:
        metrics["cv"] = {"mean_accuracy": float(cv_scores.mean()), "std": float(cv_scores.std()), "k": args.cv}

    save_metrics(metrics, metrics_dir / "metrics.json")

    # Confusion matrix
    plot_and_save_confusion_matrix(
        y_test, y_pred, labels=classes, path=figures_dir / "confusion_matrix.png", title=f"Confusion Matrix — {args.model.upper()}"
    )

    # Save metadata (features and classes)
    metadata = {
        "features": IRIS_FEATURES,
        "target": TARGET_COL,
        "classes": classes,
        "model": args.model,
        "scaled": str2bool(args.scale),
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_dir / 'metrics.json'}")
    print(f"Saved confusion matrix: {figures_dir / 'confusion_matrix.png'}")
    print(f"Saved metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()