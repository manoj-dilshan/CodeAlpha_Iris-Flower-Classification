from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, average: str = "macro") -> dict:
    """
    Compute scalar metrics + full classification report dict.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        f"precision_{average}": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        f"recall_{average}": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        f"f1_{average}": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def save_metrics(metrics: dict, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_and_save_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    path: str | Path,
    title: str = "Confusion Matrix",
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)