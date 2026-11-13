"""Global configuration for the Image Classification project."""

import os

SEED = 42

# Dataset
DATA_DIR = os.path.join(os.getcwd(), "data")
MODELS_DIR = os.path.join(os.getcwd(), "models")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]  # CIFAR-10 class names

# Input / model
IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = len(CLASS_NAMES)
DROPOUT_RATE = 0.25
DENSE_UNITS = 256

# Training
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
PATIENCE = 5

# Checkpoints
CHECKPOINT_NAME = "img_class_best.keras"


def make_dirs() -> None:
    """Create output directories if they do not exist."""
    for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        os.makedirs(directory, exist_ok=True)