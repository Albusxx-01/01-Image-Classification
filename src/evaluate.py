"""Evaluation metrics and plots."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import CLASS_NAMES


def evaluate_model(
    model: Model, x_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, float]:
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, test_accuracy


def get_predictions(model: Model, x: np.ndarray) -> np.ndarray:
    return np.argmax(model.predict(x, verbose=0), axis=-1)


def _save_figure(fig, save_path: Optional[str]) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    model: Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str] = None,
) -> np.ndarray:
    y_true = np.argmax(y_test, axis=-1)
    y_pred = get_predictions(model, x_test)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    threshold = cm.max() / 2.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = cm[row, col]
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    _save_figure(fig, save_path)
    return cm


def print_classification_report(
    model: Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str] = None,
) -> str:
    y_true = np.argmax(y_test, axis=-1)
    y_pred = get_predictions(model, x_test)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}\n")
    print(report)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(
            f"Test Accuracy: {accuracy:.4f}\n\n{report}", encoding="utf-8"
        )

    return report


def show_misclassified(
    model: Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_samples: int = 10,
    save_path: Optional[str] = None,
) -> None:
    y_true = np.argmax(y_test, axis=-1)
    y_pred = get_predictions(model, x_test)
    misclassified = np.where(y_true != y_pred)[0]

    if len(misclassified) == 0:
        print("No misclassified samples!")
        return

    indices = misclassified[:num_samples]
    cols = 5
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    for slot, sample_idx in enumerate(indices):
        axes[slot].imshow(x_test[sample_idx])
        axes[slot].set_title(
            f"T:{CLASS_NAMES[y_true[sample_idx]]}\nP:{CLASS_NAMES[y_pred[sample_idx]]}",
            fontsize=8,
            color="red",
        )
        axes[slot].axis("off")

    for ax in axes[len(indices) :]:
        ax.axis("off")

    fig.suptitle("Misclassified Samples", y=1.02)
    fig.tight_layout()
    _save_figure(fig, save_path)


def show_sample_grid(
    x: np.ndarray,
    y: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    num_samples: int = 15,
) -> None:
    cols = 5
    rows = int(np.ceil(num_samples / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    labels = np.argmax(y, axis=-1) if y.ndim == 2 else y
    for i in range(num_samples):
        axes[i].imshow(x[i])
        axes[i].set_title(class_names[labels[i]], fontsize=8)
        axes[i].axis("off")

    for ax in axes[num_samples:]:
        ax.axis("off")

    fig.suptitle("Sample Training Images", y=1.02)
    fig.tight_layout()
    plt.show()
