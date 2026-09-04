from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Banana Ripeness Assessment",
    page_icon="🍌",
    layout="wide",
)

MODEL_PATH = Path(__file__).with_name("kmeans_banana_svm.joblib")

DEFAULT_RANDOM_SEED = 42
DEFAULT_IMAGE_SIZE = (64, 64)  # (width, height)
DEFAULT_K = 3
DEFAULT_FEATURE_NAMES = [
    "dark_ratio",
    "medium_ratio",
    "bright_ratio",
    "dark_center",
    "medium_center",
    "bright_center",
    "dark_std",
    "medium_std",
    "bright_std",
]


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None

    loaded = joblib.load(MODEL_PATH)

    if isinstance(loaded, dict) and "model" in loaded:
        return loaded

    return {
        "model": loaded,
        "image_size": DEFAULT_IMAGE_SIZE,
        "k": DEFAULT_K,
        "feature_names": DEFAULT_FEATURE_NAMES,
        "random_seed": DEFAULT_RANDOM_SEED,
    }


def resize_with_padding(image, target_size, pad_value=0):
    target_width, target_height = target_size
    height, width = image.shape[:2]

    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )

    if image.ndim == 2:
        canvas = np.full(
            (target_height, target_width),
            pad_value,
            dtype=image.dtype,
        )
    else:
        canvas = np.full(
            (target_height, target_width, image.shape[2]),
            pad_value,
            dtype=image.dtype,
        )

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    return canvas


def decode_uploaded_image(uploaded_file):
    file_bytes = np.asarray(
        bytearray(uploaded_file.getvalue()),
        dtype=np.uint8,
    )
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("The uploaded file could not be decoded as an image.")

    return image_bgr


def prepare_grayscale_from_bgr(image_bgr, image_size):
    resized_bgr = resize_with_padding(
        image_bgr,
        target_size=image_size,
        pad_value=0,
    )
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    return resized_bgr, gray


def segment_kmeans_grayscale(gray, k, random_seed):
    if gray.ndim != 2:
        raise ValueError("K-Means input must be a single-channel grayscale image.")

    filtered = cv2.medianBlur(gray, 3)
    pixels = filtered.reshape(-1, 1).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2,
    )

    cv2.setRNGSeed(int(random_seed))

    _, raw_labels, raw_centres = cv2.kmeans(
        pixels,
        int(k),
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    raw_centres = raw_centres.reshape(-1)

    centre_order = np.argsort(raw_centres)
    raw_to_ordered = np.empty_like(centre_order)
    raw_to_ordered[centre_order] = np.arange(k)

    ordered_labels = raw_to_ordered[
        raw_labels.reshape(-1)
    ].reshape(filtered.shape)

    ordered_centres = raw_centres[centre_order]

    segmented = np.rint(
        ordered_centres[ordered_labels]
    ).clip(0, 255).astype(np.uint8)

    return filtered, ordered_labels, ordered_centres, segmented


def extract_kmeans_features_from_gray(gray, k, random_seed):
    filtered, ordered_labels, ordered_centres, segmented = (
        segment_kmeans_grayscale(gray, k, random_seed)
    )

    total_pixels = filtered.size
    ratios = []
    standard_deviations = []

    for ordered_id in range(k):
        values = filtered[
            ordered_labels == ordered_id
        ].astype(np.float32)

        ratios.append(values.size / total_pixels)
        standard_deviations.append(
            float(values.std()) if values.size else 0.0
        )

    features = np.array(
        ratios
        + ordered_centres.astype(float).tolist()
        + standard_deviations,
        dtype=np.float32,
    )

    return (
        features,
        filtered,
        ordered_labels,
        ordered_centres,
        segmented,
    )


def make_cluster_mask(ordered_labels, cluster_id):
    return np.where(
        ordered_labels == cluster_id,
        255,
        0,
    ).astype(np.uint8)


bundle = load_model_bundle()

st.title("🍌 Banana Ripeness Assessment")
st.caption(
    "Grayscale Intensity-Based K-Means Clustering for "
    "Banana Surface Segmentation + SVM"
)

with st.sidebar:
    st.header("Model")
    st.write("Classes")
    st.write("• Unripe")
    st.write("• Ripe")
    st.write("• Overripe")
    st.write("• Rotten")
    st.divider()
    st.write("Main image-processing algorithm")
    st.code("Grayscale K-Means (K = 3)", language=None)
    st.write("Classifier")
    st.code("RBF Support Vector Machine", language=None)
    st.info(
        "No colour features and no CLAHE are used. "
        "The uploaded colour image is displayed only for human viewing."
    )

if bundle is None:
    st.error("Trained model file not found: `kmeans_banana_svm.joblib`")
    st.markdown(
        "Run the supplied Colab model-export cell after training `kmeans_search`, "
        "download `kmeans_banana_svm.joblib`, and place it in the same folder as `app.py`."
    )
    st.stop()

model = bundle["model"]
image_size = tuple(bundle.get("image_size", DEFAULT_IMAGE_SIZE))
k = int(bundle.get("k", DEFAULT_K))
feature_names = list(bundle.get("feature_names", DEFAULT_FEATURE_NAMES))
random_seed = int(bundle.get("random_seed", DEFAULT_RANDOM_SEED))

uploaded_file = st.file_uploader(
    "Upload a banana image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Upload a banana image to start the prediction.")
    st.stop()

try:
    image_bgr = decode_uploaded_image(uploaded_file)
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    resized_bgr, gray = prepare_grayscale_from_bgr(
        image_bgr,
        image_size,
    )
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    (
        features,
        filtered,
        ordered_labels,
        centres,
        segmented,
    ) = extract_kmeans_features_from_gray(
        gray,
        k,
        random_seed,
    )

    feature_row = features.reshape(1, -1)
    predicted_class = model.predict(feature_row)[0]

    probabilities = None
    confidence = None
    class_names = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_row)[0]

        if hasattr(model, "named_steps") and "svm" in model.named_steps:
            class_names = list(model.named_steps["svm"].classes_)
        elif hasattr(model, "classes_"):
            class_names = list(model.classes_)

        if class_names is not None:
            predicted_index = class_names.index(predicted_class)
            confidence = float(probabilities[predicted_index])

except Exception as exc:
    st.exception(exc)
    st.stop()

st.subheader("Prediction")

left, right = st.columns([1, 1])

with left:
    st.image(
        original_rgb,
        caption="Uploaded image",
        use_container_width=True,
    )

with right:
    st.metric(
        "Predicted ripeness",
        str(predicted_class).upper(),
    )

    if confidence is not None:
        st.metric(
            "Prediction confidence",
            f"{confidence * 100:.2f}%",
        )

    st.caption(
        "Confidence is for this individual prediction. "
        "It is different from the model's overall test accuracy."
    )

st.subheader("Image-processing pipeline")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.image(
        resized_rgb,
        caption=f"1. Resize + padding ({image_size[0]}×{image_size[1]})",
        use_container_width=True,
    )

with c2:
    st.image(
        gray,
        caption="2. Grayscale",
        clamp=True,
        use_container_width=True,
    )

with c3:
    st.image(
        filtered,
        caption="3. Median filter (3×3)",
        clamp=True,
        use_container_width=True,
    )

with c4:
    st.image(
        segmented,
        caption=(
            "4. K-Means segmentation\n"
            f"centres: {centres[0]:.1f}, {centres[1]:.1f}, {centres[2]:.1f}"
        ),
        clamp=True,
        use_container_width=True,
    )

st.subheader("K-Means cluster masks")
st.caption(
    "These black/white masks are visualisations of the three K-Means "
    "intensity clusters. No additional segmentation algorithm is used."
)

mask_columns = st.columns(3)
mask_names = ["Dark region", "Medium region", "Bright region"]

for cluster_id, column in enumerate(mask_columns):
    mask = make_cluster_mask(ordered_labels, cluster_id)
    with column:
        st.image(
            mask,
            caption=(
                f"{mask_names[cluster_id]}\n"
                f"centre = {centres[cluster_id]:.1f}"
            ),
            clamp=True,
            use_container_width=True,
        )

st.subheader("Nine K-Means features sent to the SVM")

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Value": features,
})

st.dataframe(
    feature_df,
    use_container_width=True,
    hide_index=True,
)

if probabilities is not None and class_names is not None:
    st.subheader("SVM class probabilities")

    probability_df = pd.DataFrame({
        "Class": class_names,
        "Probability": probabilities,
    }).sort_values(
        "Probability",
        ascending=False,
    )

    probability_df["Probability (%)"] = (
        probability_df["Probability"] * 100
    ).round(2)

    st.dataframe(
        probability_df[["Class", "Probability (%)"]],
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        probability_df.set_index("Class")["Probability"]
    )

st.divider()
st.caption(
    "Pipeline: Upload → Resize → Grayscale → Median Filter → "
    "Grayscale K-Means (K=3) → 9 segmentation features → "
    "StandardScaler → RBF SVM → Ripeness prediction"
)
