"""Model compilation, callbacks, and training loop."""

import os

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
    """Compile the model with Adam + categorical cross-entropy + accuracy."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(
    patience: int = PATIENCE,
    checkpoint_path: str = None,
) -> list:
    """Return early stopping, checkpointing, and LR-reduction callbacks."""
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
    """Build a batched tf.data pipeline, optionally augmenting every epoch."""
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
) -> object:
    """Train with per-epoch on-the-fly augmentation via tf.data.

    Returns the training history object.
    """
    callbacks_list = get_callbacks()
    active_augmenter = augmenter if use_augmentation else None

    train_ds = make_dataset(
        data["x_train"], data["y_train"], batch_size, active_augmenter
    )
    val_ds = make_dataset(
        data["x_val"], data["y_val"], batch_size, None, shuffle=False
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )
    return history


def plot_training_history(history, save_path: str = None) -> None:
    """Plot training and validation loss + accuracy curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="train")
    ax1.plot(history.history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["accuracy"], label="train")
    ax2.plot(history.history["val_accuracy"], label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()