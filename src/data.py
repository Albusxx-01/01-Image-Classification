"""CIFAR-10 loading, preprocessing, splitting, and augmentation."""

import os

import numpy as np
from tensorflow import keras

from config import NUM_CLASSES, SEED, VALIDATION_SPLIT


def load_cifar10(
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_dir = os.path.join(data_dir, "keras_cache")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data(
        path=os.path.join(cache_dir, "cifar-10-batches-py.tar.gz")
    )
    return x_train, y_train, x_test, y_test


def normalize(x: np.ndarray) -> np.ndarray:
    return x.astype("float32") / 255.0


def one_hot(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    return keras.utils.to_categorical(y, num_classes)


def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = VALIDATION_SPLIT,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(x))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_val = int(len(indices) * val_ratio)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def prepare_data(data_dir: str) -> dict[str, np.ndarray]:
    x_train, y_train, x_test, y_test = load_cifar10(data_dir)
    x_train, y_train, x_val, y_val = split_train_val(x_train, y_train)

    return {
        "x_train": normalize(x_train),
        "y_train": one_hot(y_train),
        "x_val": normalize(x_val),
        "y_val": one_hot(y_val),
        "x_test": normalize(x_test),
        "y_test": one_hot(y_test),
    }


def make_augmentation(
    rotation: float = 0.15,
    shift: float = 0.1,
    zoom: float = 0.1,
    horizontal_flip: bool = True,
) -> keras.Sequential:
    augmentations = [
        keras.layers.RandomRotation(rotation, fill_mode="nearest"),
        keras.layers.RandomTranslation(shift, shift, fill_mode="nearest"),
        keras.layers.RandomZoom(zoom, zoom, fill_mode="nearest"),
    ]
    if horizontal_flip:
        augmentations.append(keras.layers.RandomFlip("horizontal"))

    return keras.Sequential(augmentations, name="augmentation")
