"""Evaluation: test metrics, confusion matrix, report, and error visualization."""

import os

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import CLASS_NAMES, RESULTS_DIR


def evaluate_model(model: Model, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """Return (test_loss, test_accuracy) from model.evaluate."""
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, test_accuracy


def get_predictions(model: Model, x: np.ndarray) -> np.ndarray:
    """Return integer class predictions for input batch."""
    return np.argmax(model.predict(x, verbose=0), axis=-1)


def plot_confusion_matrix(
    model: Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = None,
) -> np.ndarray:
    """Compute, plot, and return the confusion matrix."""
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

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                "%d" % cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    return cm


def print_classification_report(
    model: Model, x_test: np.ndarray, y_test: np.ndarray, save_path: str = None
) -> str:
    """Print and return the per-class classification report."""
    y_true = np.argmax(y_test, axis=-1)
    y_pred = get_predictions(model, x_test)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=3
    )
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}\n")
    print(report)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
            f.write(report)
    return report


def show_misclassified(
    model: Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_samples: int = 10,
    save_path: str = None,
) -> None:
    """Display a grid of misclassified samples with true vs predicted labels."""
    y_true = np.argmax(y_test, axis=-1)
    y_pred = get_predictions(model, x_test)
    mis_idx = np.where(y_true != y_pred)[0]

    if len(mis_idx) == 0:
        print("No misclassified samples!")
        return

    idx = mis_idx[:num_samples]
    cols = 5
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    for ax, i in enumerate(idx):
        axes[ax].imshow(x_test[i])
        axes[ax].set_title(
            f"T:{CLASS_NAMES[y_true[i]]}\nP:{CLASS_NAMES[y_pred[i]]}",
            fontsize=8,
            color="red",
        )
        axes[ax].axis("off")

    for ax in axes[len(idx):]:
        ax.axis("off")

    plt.suptitle("Misclassified Samples", y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def show_sample_grid(
    x: np.ndarray,
    y: np.ndarray,
    class_names: list = CLASS_NAMES,
    num_samples: int = 15,
) -> None:
    """Display a grid of sample images with their true labels."""
    cols = 5
    rows = int(np.ceil(num_samples / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    y_labels = np.argmax(y, axis=-1) if y.ndim == 2 else y
    for i in range(num_samples):
        axes[i].imshow(x[i])
        axes[i].set_title(class_names[y_labels[i]], fontsize=8)
        axes[i].axis("off")

    for ax in axes[num_samples:]:
        ax.axis("off")

    plt.suptitle("Sample Training Images", y=1.02)
    plt.tight_layout()
    plt.show()