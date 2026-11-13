"""Data loading, preprocessing, splitting, and augmentation."""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import NUM_CLASSES, SEED, VALIDATION_SPLIT


def load_cifar10(data_dir: str) -> tuple:
    """Download and load CIFAR-10 (cached on first call).

    Returns:
        (x_train, y_train, x_test, y_test) where x shape is (N, 32, 32, 3)
        and y shape is (N, 1) with integer class indices.
    """
    cache_dir = os.path.join(data_dir, "keras_cache")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data(
        path=os.path.join(cache_dir, "cifar-10-batches-py.tar.gz")
    )
    return x_train, y_train, x_test, y_test


def normalize(x: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0.0, 1.0] float32."""
    return x.astype("float32") / 255.0


def one_hot(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert integer labels to one-hot vectors."""
    return keras.utils.to_categorical(y, num_classes)


def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = VALIDATION_SPLIT,
    seed: int = SEED,
) -> tuple:
    """Split training data into train/validation sets (shuffled).

    Returns (x_train, y_train, x_val, y_val).
    """
    indices = np.arange(len(x))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_val = int(len(indices) * val_ratio)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def prepare_data(data_dir: str) -> dict:
    """Full pipeline: load → normalize → one-hot → split.

    Returns a dict with keys: x_train, y_train, x_val, y_val, x_test, y_test.
    """
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_cifar10(data_dir)

    x_train_raw, y_train_raw, x_val_raw, y_val_raw = split_train_val(
        x_train_raw, y_train_raw
    )

    x_train = normalize(x_train_raw)
    x_val = normalize(x_val_raw)
    x_test = normalize(x_test_raw)

    y_train = one_hot(y_train_raw)
    y_val = one_hot(y_val_raw)
    y_test = one_hot(y_test_raw)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def make_augmentation(
    rotation: float = 0.15,
    shift: float = 0.1,
    zoom: float = 0.1,
    horizontal_flip: bool = True,
) -> keras.Sequential:
    """Build a Keras data-augmentation pipeline applied on-the-fly."""
    layers = keras.layers
    augmenter = keras.Sequential(
        [
            layers.RandomRotation(rotation, fill_mode="nearest"),
            layers.RandomTranslation(shift, shift, fill_mode="nearest"),
            layers.RandomZoom(zoom, zoom, fill_mode="nearest"),
            layers.RandomFlip("horizontal") if horizontal_flip else layers.Lambda(
                lambda x: x
            ),
        ],
        name="augmentation",
    )
    return augmenter