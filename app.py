"""Streamlit UI for the CIFAR-10 classifier."""

import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from tensorflow import keras
except ImportError:
    keras = None

DEFAULT_CLASS_NAMES = [
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
CLASSES_PATH = Path(__file__).with_name("classes.json")
DEFAULT_MODEL = os.path.join("models", "cifar10_cnn_final.keras")


def load_class_names() -> list[str]:
    try:
        mapping = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        return [mapping[str(i)] for i in range(len(mapping))]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return DEFAULT_CLASS_NAMES


CLASS_NAMES = load_class_names()

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="centered")
st.title("🖼️ CIFAR-10 Image Classifier")
st.caption("Upload an image and the CNN will predict its class with confidence scores.")


@st.cache_resource(show_spinner="Loading model...")
def get_model(source: str) -> "keras.Model":
    return keras.models.load_model(source)


@st.cache_resource(show_spinner="Loading model from upload...")
def get_model_from_bytes(data: bytes) -> "keras.Model":
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return keras.models.load_model(tmp_path)
    finally:
        os.unlink(tmp_path)


def preprocess(image: Image.Image, input_size: int) -> np.ndarray:
    image = image.resize((input_size, input_size), Image.LANCZOS)
    arr = np.asarray(image, dtype="float32") / 255.0
    if input_size == 224:  # ResNet50 expects [-1, 1]
        arr = arr * 2.0 - 1.0
    return arr


def main() -> None:
    if keras is None:
        st.error(
            "TensorFlow is not installed. Install dependencies with "
            "`pip install -r requirements.txt`."
        )
        return

    st.sidebar.header("Model")
    model_source = st.sidebar.radio("Model source", ["File path", "Upload .keras"])

    model = None
    if model_source == "File path":
        default_path = DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else ""
        model_path = st.sidebar.text_input(
            "Path to .keras model", value=default_path, help="Absolute or relative path."
        )
        if model_path and os.path.exists(model_path):
            model = get_model(model_path)
        elif model_path:
            st.sidebar.warning("File not found.")
    else:
        model_file = st.sidebar.file_uploader("Upload model (.keras)", type=["keras"])
        if model_file is not None:
            model = get_model_from_bytes(model_file.getvalue())

    if model is None:
        st.info(
            "Load a trained model first (see sidebar). If you don't have one, run "
            "`01b-cifar10-training-improved.ipynb` in Colab and download "
            "`cifar10_cnn_final.keras` into the `models/` folder."
        )
        return

    input_shape = getattr(model, "input_shape", None)
    detected_size = input_shape[1] if input_shape and len(input_shape) >= 3 else None
    if detected_size in (32, 224):
        input_size = int(detected_size)
        st.sidebar.info(f"Input size: {input_size}×{input_size} (auto-detected)")
    else:
        input_size = st.sidebar.selectbox(
            "Input size",
            [32, 224],
            index=0,
            help="32 for baseline CNN, 224 for ResNet50.",
        )
    top_k = st.sidebar.slider("Show top-K classes", 1, 10, 3)

    tab_upload, tab_camera = st.tabs(["Upload image", "Take a photo"])
    image_bytes = None
    with tab_upload:
        uploaded = st.file_uploader(
            "Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
    with tab_camera:
        camera = st.camera_input("Take a photo")
        if camera is not None:
            image_bytes = camera.getvalue()

    if image_bytes is None:
        st.info("Upload an image or take a photo to get started.")
        return

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    probs = model.predict(preprocess(image, input_size)[None, ...], verbose=0)[0]

    col_pic, col_chart = st.columns([1, 1.3])
    with col_pic:
        st.image(image, caption="Your image", use_container_width=True)

    top_idx = np.argsort(probs)[::-1][:top_k]
    pred_label, pred_conf = CLASS_NAMES[top_idx[0]], probs[top_idx[0]]
    st.success(f"**Prediction: {pred_label}** (confidence {pred_conf:.2%})")

    with col_chart:
        chart_data = pd.DataFrame(
            {
                "Class": [CLASS_NAMES[i] for i in top_idx],
                "Probability": [probs[i] for i in top_idx],
            }
        )
        st.bar_chart(chart_data, x="Class", y="Probability", horizontal=True)

    with st.expander("See all class probabilities"):
        all_data = pd.DataFrame(
            {"Class": CLASS_NAMES, "Probability": probs}
        ).sort_values("Probability", ascending=False)
        st.dataframe(all_data, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
