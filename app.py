"""Streamlit UI for the CIFAR-10 CNN image classifier.

Run locally with:
    streamlit run app.py

Requires: pip install streamlit tensorflow pillow numpy matplotlib
"""

import io
import json
import os
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except Exception as exc:  # pragma: no cover - only when TF is missing
    TF_AVAILABLE = False
    TF_ERROR = exc

DEFAULT_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_class_names() -> list:
    """Load the index -> name mapping from classes.json when available."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes.json")
    try:
        with open(path, encoding="utf-8") as f:
            mapping = json.load(f)
        return [mapping[str(i)] for i in range(len(mapping))]
    except (OSError, KeyError, ValueError):
        return DEFAULT_CLASS_NAMES


CLASS_NAMES = load_class_names()
DEFAULT_MODEL = os.path.join("models", "cifar10_cnn_final.keras")

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="centered")

st.title("🖼️ CIFAR-10 Image Classifier")
st.caption("Upload an image and the CNN will predict its class with confidence scores.")


@st.cache_resource(show_spinner="Loading model...")
def get_model(source: str) -> object:
    """Load the Keras model from a file path (cached)."""
    return keras.models.load_model(source)


@st.cache_resource(show_spinner="Loading model from upload...")
def get_model_from_bytes(data: bytes) -> object:
    """Load the Keras model from uploaded bytes (cached)."""
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    model = keras.models.load_model(tmp_path)
    os.unlink(tmp_path)
    return model


def preprocess(image: Image.Image, input_size: int) -> np.ndarray:
    """Resize, normalize, and scale an image to the model's expected input."""
    image = image.resize((input_size, input_size), Image.LANCZOS)
    arr = np.asarray(image, dtype="float32") / 255.0
    if input_size == 224:  # ResNet50 expects [-1, 1]
        arr = arr * 2.0 - 1.0
    return arr


def predict(model, arr: np.ndarray) -> np.ndarray:
    """Return a probability vector for a single preprocessed image."""
    probs = model.predict(arr[None, ...], verbose=0)[0]
    return probs


def main() -> None:
    if not TF_AVAILABLE:
        st.error(
            "TensorFlow is not installed. Run: "
            "`pip install tensorflow keras streamlit pillow numpy matplotlib`"
        )
        return

    # ---------- Sidebar: model selection ----------
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

    # ---------- Input size + UI params ----------
    model_shape = getattr(model, "input_shape", None)
    detected_size = model_shape[1] if model_shape and len(model_shape) >= 3 else None
    if detected_size in (32, 224):
        input_size = int(detected_size)
        st.sidebar.info(f"Input size: {input_size}×{input_size} (auto-detected)")
    else:
        input_size = st.sidebar.selectbox(
            "Input size", [32, 224], index=0, help="32 for baseline CNN, 224 for ResNet50."
        )
    top_k = st.sidebar.slider("Show top-K classes", 1, 10, 3)

    # ---------- Image input ----------
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
    arr = preprocess(image, input_size)
    probs = predict(model, arr)

    # ---------- Results ----------
    col_pic, col_chart = st.columns([1, 1.3])
    with col_pic:
        st.image(image, caption="Your image", use_container_width=True)

    top_idx = np.argsort(probs)[::-1][:top_k]
    pred_label, pred_conf = CLASS_NAMES[top_idx[0]], probs[top_idx[0]]
    st.success(f"**Prediction: {pred_label}** (confidence {pred_conf:.2%})")

    with col_chart:
        import pandas as pd

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