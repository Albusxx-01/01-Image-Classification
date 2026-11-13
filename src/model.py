"""CNN model definition and compilation."""

from keras import Model, layers, models


def build_cnn(
    input_shape: tuple = (32, 32, 3),
    num_classes: int = 10,
    dropout_rate: float = 0.25,
    dense_units: int = 256,
) -> Model:
    """Build a sequential CNN: Conv → Pool → Dropout → Flatten → Dense → Softmax.

    Architecture:
        Conv(32) → BN → Conv(32) → MaxPool → Dropout
        Conv(64) → BN → Conv(64) → MaxPool → Dropout
        Conv(128) → BN → Conv(128) → MaxPool → Dropout
        Flatten → Dense(256) → BN → Dropout(0.5) → Dense(num_classes, softmax)
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),

            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),

            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),

            # Classifier
            layers.Flatten(),
            layers.Dense(dense_units, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cifar10_cnn",
    )
    return model