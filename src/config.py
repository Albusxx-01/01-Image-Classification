"""Project-wide configuration."""

import os

SEED = 42

DATA_DIR = os.path.join(os.getcwd(), "data")
MODELS_DIR = os.path.join(os.getcwd(), "models")
RESULTS_DIR = os.path.join(os.getcwd(), "results")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = len(CLASS_NAMES)
DROPOUT_RATE = 0.25
DENSE_UNITS = 256

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
PATIENCE = 5

CHECKPOINT_NAME = "img_class_best.keras"


def make_dirs() -> None:
    for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        os.makedirs(directory, exist_ok=True)
