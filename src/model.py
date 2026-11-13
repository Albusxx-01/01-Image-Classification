"""CIFAR-10 CNN architecture."""

from keras import Model, layers, models

from config import DENSE_UNITS, DROPOUT_RATE, IMAGE_SHAPE, NUM_CLASSES


def build_cnn(
    input_shape: tuple = IMAGE_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    dense_units: int = DENSE_UNITS,
) -> Model:
    return models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            layers.Flatten(),
            layers.Dense(dense_units, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cifar10_cnn",
    )
