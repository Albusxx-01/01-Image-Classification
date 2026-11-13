# Image Classification with CNN (CIFAR-10)

A modular Convolutional Neural Network that classifies CIFAR-10 images into 10 classes.
This is the foundational computer-vision project — begin here before Object Detection, Segmentation, GANs, and Face Recognition.

## Quick Start (Google Colab)

1. Upload any notebook from `notebooks/` to [Google Colab](https://colab.research.google.com/).
2. Runtime → **Change runtime type** → GPU.
3. Runtime → **Run all**.

All notebooks are **fully self-contained** — all code lives in the cells (no `src/` files
needed in the session). Run them in order:

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `01-cifar10-training.ipynb` | Baseline CNN — first attempt (augmentation too aggressive; ~56% accuracy) |
| 1b | `01b-cifar10-training-improved.ipynb` | **Recommended.** Fixed recipe: flip+crop augmentation via `tf.data`, patience 8, 40 epochs → 85–90% |
| 2 | `02-transfer-learning.ipynb` | Fine-tune a pretrained ResNet50 on CIFAR-10 (Milestone 5) |
| 3 | `03-predict.ipynb` | Load a saved model and test it on new images with confidence scores |

Outputs land in `models\` and `results\`.

## Streamlit UI (local)

Once you have a trained model file (`models/cifar10_cnn_final.keras`), you can interact with
it through a web UI:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Features: load the model from a file path or upload it, pick input size (32 baseline / 224
ResNet50), upload an image or take a photo with your webcam, and see the predicted class,
top-K confidence bar chart, and full probability table.

> TensorFlow is included in `requirements.txt`; a single `pip install -r requirements.txt`
> installs everything needed for the app and the `src/` pipeline.

## Project Structure

```
01-Image-Classification/
├── pre-build docs/        # local-only planning docs (gitignored, not pushed)
├── learnings/             # concept study notes per phase
├── notebooks/
│   ├── 01-cifar10-training.ipynb        # baseline CNN — first attempt (56%)
│   ├── 01b-cifar10-training-improved.ipynb  # RECOMMENDED — fixed recipe (85-90%)
│   ├── 02-transfer-learning.ipynb       # ResNet50 fine-tuning (Colab)
│   └── 03-predict.ipynb                 # prediction demo on new images (Colab)
├── src/                   # optional modular pipeline (same logic as the notebook)
│   ├── config.py          # hyperparameters, paths, class names
│   ├── data.py            # load, normalize, one-hot, split, augmentation
│   ├── model.py           # CNN architecture
│   ├── train.py           # compile, callbacks, training loop, history plot
│   └── evaluate.py        # accuracy, confusion matrix, report, misclassified plots
├── data/                  # downloaded CIFAR-10 cache (Colab session only; gitignored)
├── models/                # cifar10_cnn_final.keras, img_class_best.keras
├── results/               # confusion matrix, report, training plots, sample images
├── classes.json           # index -> class-name mapping
├── .gitignore             # local-only files (pre-build docs, caches)
└── requirements.txt
```

## Pipeline

```
Load CIFAR-10 → Normalize [0,1] → One-hot labels → Split 70/15/15
→ Augment → CNN(Conv→BN→MaxPool→Dropout→Dense→Softmax)
→ Train (Adam lr=0.001, early stopping, checkpointing, LR schedule)
→ Evaluate (test accuracy, confusion matrix, classification report)
```

## Architecture

| Block | Layers |
|-------|--------|
| 1 | Conv(32) → BatchNorm → Conv(32) → MaxPool → Dropout(0.25) |
| 2 | Conv(64) → BatchNorm → Conv(64) → MaxPool → Dropout(0.25) |
| 3 | Conv(128) → BatchNorm → Conv(128) → MaxPool → Dropout(0.25) |
| Classifier | Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(10, softmax) |

## Results

Measured on the held-out 10,000-image CIFAR-10 test set:

| Model | Test Accuracy | Notes |
|-------|---------------|-------|
| Baseline CNN (notebook 01 — first run) | 56.33% | heavy rotation/zoom augmentation hurt 32×32 images + early stop |
| Baseline CNN (notebook **01b** — improved) | **85.43%** | flip+crop augmentation via tf.data, patience 8; macro F1 0.853 |
| ResNet50 fine-tuned (notebook 02) | Not run (optional) | 224×224 input, [-1,1] scaling; target 90–95% |

Artifacts saved by the notebooks: `models/img_class_best.keras`, `models/cifar10_cnn_final.keras`,
plus `results/` (`confusion_matrix.png`, `classification_report.txt`, `training_history.png`,
`misclassified.png`). The ResNet50 artifacts (`models/cifar10_resnet50_finetuned.keras`,
`results/transfer_*`) are only produced if notebook 02 is run.

## Running Without the Notebook (source `src/` modules)

The notebook contains everything you need. If you prefer to use the modular `src/`
files instead (e.g., in a local IDE), add `src/` to the path first:

```bash
pip install -r requirements.txt
python -c "
import os, sys; sys.path.insert(0, 'src')
import config; config.make_dirs()
from data import prepare_data, make_augmentation
from model import build_cnn
from train import compile_model, train_model, plot_training_history
from evaluate import evaluate_model, plot_confusion_matrix, print_classification_report

data = prepare_data(config.DATA_DIR)
model = compile_model(build_cnn(input_shape=config.IMAGE_SHAPE, num_classes=config.NUM_CLASSES))
history = train_model(model, data, use_augmentation=True, augmenter=make_augmentation())
plot_training_history(history)
evaluate_model(model, data['x_test'], data['y_test'])
plot_confusion_matrix(model, data['x_test'], data['y_test'])
print_classification_report(model, data['x_test'], data['y_test'])
"
```

## Milestones Covered

- [x] Milestone 0 — Setup, folder structure, requirements, config
- [x] Milestone 1 — Data pipeline (download, normalize, one-hot, split, visualize)
- [x] Milestone 2 — Baseline CNN (Conv → Pool → Dropout → Dense → Softmax)
- [x] Milestone 3 — Full training (callbacks: early stopping, checkpoint, LR schedule)
- [x] Milestone 4 — Evaluation (accuracy, confusion matrix, report, misclassified plots)
- [x] Milestone 5 — Optimization (augmentation, batch norm, LR scheduling, transfer learning with ResNet50)
- [x] Milestone 6 — Finalize (results table updated; artifacts committed)# 01-Image-Classification
