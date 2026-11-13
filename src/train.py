"""Training setup, callbacks, and history plotting."""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model, callbacks, optimizers

from config import (
    BATCH_SIZE,
    CHECKPOINT_NAME,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    PATIENCE,
    SEED,
)


def compile_model(model: Model, learning_rate: float = LEARNING_RATE) -> Model:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(
    patience: int = PATIENCE,
    checkpoint_path: Optional[str] = None,
) -> list:
    if checkpoint_path is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        checkpoint_path = os.path.join(MODELS_DIR, CHECKPOINT_NAME)

    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def make_dataset(
    x,
    y,
    batch_size: int = BATCH_SIZE,
    augmenter=None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        ds = ds.shuffle(min(len(x), 10000), seed=SEED)

    if augmenter is not None:
        ds = ds.map(
            lambda images, labels: (augmenter(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_model(
    model: Model,
    data: dict,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    use_augmentation: bool = True,
    augmenter=None,
) -> callbacks.History:
    train_ds = make_dataset(
        data["x_train"],
        data["y_train"],
        batch_size,
        augmenter if use_augmentation else None,
    )
    val_ds = make_dataset(data["x_val"], data["y_val"], batch_size, shuffle=False)

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(),
        verbose=1,
    )


def plot_training_history(
    history: callbacks.History, save_path: Optional[str] = None
) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(history.history["loss"], label="train")
    ax_loss.plot(history.history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(history.history["accuracy"], label="train")
    ax_acc.plot(history.history["val_accuracy"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    plt.show()
