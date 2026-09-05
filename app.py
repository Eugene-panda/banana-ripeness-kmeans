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


MODEL_PATH = Path(__file__).with_name(
    "eugene_watershed_svm.joblib"
)

DEFAULT_LABELS = [
    "unripe",
    "ripe",
    "overripe",
    "rotten",
]

DEFAULT_SVM_IMAGE_SIZE = (64, 64)
DEFAULT_PROCESSING_SIZE = (256, 256)
DEFAULT_GAUSSIAN_KERNEL = (5, 5)
DEFAULT_GAUSSIAN_SIGMA = 0

DEFAULT_WATERSHED_BORDER_RATIO = 0.06
DEFAULT_WATERSHED_CENTER_RADIUS_X = 0.43
DEFAULT_WATERSHED_CENTER_RADIUS_Y = 0.43
DEFAULT_WATERSHED_LOW_GRADIENT_PERCENTILE = 35
DEFAULT_WATERSHED_MIN_SEED_AREA_RATIO = 0.001
DEFAULT_WATERSHED_MAX_FOREGROUND_SEEDS = 8

REGION_FEATURE_NAMES = [
    "foreground_area_ratio",
    "component_count",
    "largest_component_ratio",
    "perimeter_normalised",
    "bbox_aspect_ratio",
    "extent",
    "solidity",
    "circularity",
]


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None

    loaded = joblib.load(
        MODEL_PATH
    )

    if isinstance(
        loaded,
        dict,
    ) and "model" in loaded:
        return loaded

    # Fallback for a raw scikit-learn pipeline.
    return {
        "model": loaded,
        "classes": DEFAULT_LABELS,
        "svm_image_size": DEFAULT_SVM_IMAGE_SIZE,
        "processing_size": DEFAULT_PROCESSING_SIZE,
        "gaussian_kernel": DEFAULT_GAUSSIAN_KERNEL,
        "gaussian_sigma": DEFAULT_GAUSSIAN_SIGMA,
        "watershed_border_ratio": DEFAULT_WATERSHED_BORDER_RATIO,
        "watershed_center_radius_x": DEFAULT_WATERSHED_CENTER_RADIUS_X,
        "watershed_center_radius_y": DEFAULT_WATERSHED_CENTER_RADIUS_Y,
        "watershed_low_gradient_percentile": DEFAULT_WATERSHED_LOW_GRADIENT_PERCENTILE,
        "watershed_min_seed_area_ratio": DEFAULT_WATERSHED_MIN_SEED_AREA_RATIO,
        "watershed_max_foreground_seeds": DEFAULT_WATERSHED_MAX_FOREGROUND_SEEDS,
        "region_feature_names": REGION_FEATURE_NAMES,
    }


def decode_uploaded_image(
    uploaded_file,
):
    file_bytes = np.asarray(
        bytearray(
            uploaded_file.getvalue()
        ),
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(
            "The uploaded file could not be decoded as an image."
        )

    return image_bgr


def eugene_preprocess_rgb_from_bgr(
    image_bgr,
    processing_size,
    gaussian_kernel,
    gaussian_sigma,
):
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    resized_rgb = cv2.resize(
        image_rgb,
        tuple(processing_size),
        interpolation=cv2.INTER_AREA,
    )

    gaussian_rgb = cv2.GaussianBlur(
        resized_rgb,
        tuple(gaussian_kernel),
        gaussian_sigma,
    )

    return (
        image_rgb,
        resized_rgb,
        gaussian_rgb,
    )


def sobel_gradient(
    gray,
):
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    if magnitude.max() > 0:
        magnitude = (
            magnitude
            / magnitude.max()
        )

    return np.clip(
        magnitude * 255.0,
        0,
        255,
    ).astype(np.uint8)


def create_watershed_markers(
    gray,
    gradient,
    border_ratio,
    center_radius_x,
    center_radius_y,
    low_gradient_percentile,
    min_seed_area_ratio,
    max_foreground_seeds,
):
    h, w = gray.shape

    # Background marker = image border.
    border_y = max(
        1,
        int(
            round(
                h * border_ratio
            )
        ),
    )

    border_x = max(
        1,
        int(
            round(
                w * border_ratio
            )
        ),
    )

    background_seed = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    background_seed[
        :border_y,
        :,
    ] = 255

    background_seed[
        -border_y:,
        :,
    ] = 255

    background_seed[
        :,
        :border_x,
    ] = 255

    background_seed[
        :,
        -border_x:,
    ] = 255

    # Broad central region for foreground seed search.
    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    centre_x = w / 2.0
    centre_y = h / 2.0

    centre_ellipse = (
        (
            (xx - centre_x)
            / max(
                center_radius_x * w,
                1,
            )
        ) ** 2
        +
        (
            (yy - centre_y)
            / max(
                center_radius_y * h,
                1,
            )
        ) ** 2
        <= 1.0
    )

    threshold = np.percentile(
        gradient[
            centre_ellipse
        ],
        low_gradient_percentile,
    )

    candidate_seed = (
        (
            gradient
            <= threshold
        )
        & centre_ellipse
    ).astype(
        np.uint8
    )

    count, component_labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            candidate_seed,
            connectivity=8,
        )
    )

    min_area = max(
        10,
        int(
            round(
                h
                * w
                * min_seed_area_ratio
            )
        ),
    )

    ranked_components = []

    for component_id in range(
        1,
        count,
    ):
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        cx, cy = centroids[
            component_id
        ]

        distance = np.hypot(
            (cx - centre_x) / w,
            (cy - centre_y) / h,
        )

        score = area / (
            1.0
            + 2.0 * distance
        )

        ranked_components.append(
            (
                score,
                component_id,
            )
        )

    ranked_components = sorted(
        ranked_components,
        reverse=True,
    )[
        :int(
            max_foreground_seeds
        )
    ]

    foreground_seed = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    for _, component_id in ranked_components:
        foreground_seed[
            component_labels
            == component_id
        ] = 255

    if np.count_nonzero(
        foreground_seed
    ) == 0:
        cv2.ellipse(
            foreground_seed,
            (
                int(
                    round(
                        centre_x
                    )
                ),
                int(
                    round(
                        centre_y
                    )
                ),
            ),
            (
                max(
                    3,
                    int(
                        round(
                            w * 0.08
                        )
                    ),
                ),
                max(
                    3,
                    int(
                        round(
                            h * 0.08
                        )
                    ),
                ),
            ),
            0,
            0,
            360,
            255,
            -1,
        )

    markers = np.zeros(
        (h, w),
        dtype=np.int32,
    )

    markers[
        background_seed > 0
    ] = 1

    foreground_count, foreground_labels = (
        cv2.connectedComponents(
            (
                foreground_seed
                > 0
            ).astype(
                np.uint8
            ),
            connectivity=8,
        )
    )

    for foreground_id in range(
        1,
        foreground_count,
    ):
        markers[
            foreground_labels
            == foreground_id
        ] = (
            foreground_id
            + 1
        )

    return (
        markers,
        background_seed,
        foreground_seed,
    )


def run_marker_controlled_watershed(
    preprocessed_rgb,
    bundle,
):
    gray = cv2.cvtColor(
        preprocessed_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    gradient = sobel_gradient(
        gray
    )

    (
        markers,
        background_seed,
        foreground_seed,
    ) = create_watershed_markers(
        gray=gray,
        gradient=gradient,
        border_ratio=float(
            bundle.get(
                "watershed_border_ratio",
                DEFAULT_WATERSHED_BORDER_RATIO,
            )
        ),
        center_radius_x=float(
            bundle.get(
                "watershed_center_radius_x",
                DEFAULT_WATERSHED_CENTER_RADIUS_X,
            )
        ),
        center_radius_y=float(
            bundle.get(
                "watershed_center_radius_y",
                DEFAULT_WATERSHED_CENTER_RADIUS_Y,
            )
        ),
        low_gradient_percentile=float(
            bundle.get(
                "watershed_low_gradient_percentile",
                DEFAULT_WATERSHED_LOW_GRADIENT_PERCENTILE,
            )
        ),
        min_seed_area_ratio=float(
            bundle.get(
                "watershed_min_seed_area_ratio",
                DEFAULT_WATERSHED_MIN_SEED_AREA_RATIO,
            )
        ),
        max_foreground_seeds=int(
            bundle.get(
                "watershed_max_foreground_seeds",
                DEFAULT_WATERSHED_MAX_FOREGROUND_SEEDS,
            )
        ),
    )

    watershed_input = cv2.cvtColor(
        gradient,
        cv2.COLOR_GRAY2BGR,
    )

    watershed_labels = (
        markers.copy()
    )

    cv2.watershed(
        watershed_input,
        watershed_labels,
    )

    final_mask = np.where(
        watershed_labels > 1,
        255,
        0,
    ).astype(
        np.uint8
    )

    colour_cutout = cv2.bitwise_and(
        preprocessed_rgb,
        preprocessed_rgb,
        mask=final_mask,
    )

    return {
        "gray": gray,
        "gradient": gradient,
        "markers": markers,
        "background_seed": background_seed,
        "foreground_seed": foreground_seed,
        "watershed_labels": watershed_labels,
        "final_mask": final_mask,
        "colour_cutout": colour_cutout,
    }


def extract_region_features(
    mask,
):
    binary = (
        mask > 0
    ).astype(
        np.uint8
    )

    h, w = binary.shape
    image_area = h * w

    foreground_pixels = int(
        np.count_nonzero(
            binary
        )
    )

    foreground_area_ratio = (
        foreground_pixels
        / image_area
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    component_count = max(
        count - 1,
        0,
    )

    component_areas = [
        int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )
        for component_id
        in range(
            1,
            count,
        )
    ]

    largest_component_ratio = (
        max(component_areas)
        / image_area
        if component_areas
        else 0.0
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros(
            len(
                REGION_FEATURE_NAMES
            ),
            dtype=np.float32,
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    area = float(
        cv2.contourArea(
            contour
        )
    )

    perimeter = float(
        cv2.arcLength(
            contour,
            True,
        )
    )

    x, y, bw, bh = (
        cv2.boundingRect(
            contour
        )
    )

    hull_area = float(
        cv2.contourArea(
            cv2.convexHull(
                contour
            )
        )
    )

    perimeter_normalised = (
        perimeter
        / max(
            2.0 * (h + w),
            1.0,
        )
    )

    bbox_aspect_ratio = (
        bw / max(bh, 1)
    )

    extent = (
        area
        / max(
            bw * bh,
            1,
        )
    )

    solidity = (
        area
        / max(
            hull_area,
            1.0,
        )
    )

    circularity = (
        4.0
        * np.pi
        * area
        / max(
            perimeter ** 2,
            1.0,
        )
    )

    return np.array(
        [
            foreground_area_ratio,
            component_count,
            largest_component_ratio,
            perimeter_normalised,
            bbox_aspect_ratio,
            extent,
            solidity,
            circularity,
        ],
        dtype=np.float32,
    )


def rgb_pixel_vector(
    rgb,
    svm_image_size,
):
    resized = cv2.resize(
        rgb,
        tuple(
            svm_image_size
        ),
        interpolation=cv2.INTER_AREA,
    )

    return (
        resized.astype(
            np.float32
        )
        / 255.0
    ).reshape(
        -1
    )


bundle = load_model_bundle()

st.title(
    "🍌 Banana Ripeness Assessment"
)

st.caption(
    "Marker-Controlled Watershed Segmentation "
    "+ Colour Banana Cut-Out + RBF SVM"
)


with st.sidebar:
    st.header(
        "Model"
    )

    st.write(
        "Classes"
    )

    for class_name in DEFAULT_LABELS:
        st.write(
            f"• {class_name.title()}"
        )

    st.divider()

    st.write(
        "Main image-processing algorithm"
    )

    st.code(
        "Marker-Controlled Watershed",
        language=None,
    )

    st.write(
        "Classifier"
    )

    st.code(
        "RBF Support Vector Machine",
        language=None,
    )

    st.info(
        "Grayscale and Sobel gradient are used "
        "internally to calculate Watershed boundaries. "
        "The SVM receives the final colour banana cut-out."
    )


if bundle is None:
    st.error(
        "Trained model file not found: "
        "`eugene_watershed_svm.joblib`"
    )

    st.markdown(
        "Run the final export/download cells in "
        "`Eugene_Watershed_Clean.ipynb`, then place "
        "`eugene_watershed_svm.joblib` in the same "
        "folder as this `app.py`."
    )

    st.stop()


model = bundle[
    "model"
]

svm_image_size = tuple(
    bundle.get(
        "svm_image_size",
        DEFAULT_SVM_IMAGE_SIZE,
    )
)

processing_size = tuple(
    bundle.get(
        "processing_size",
        DEFAULT_PROCESSING_SIZE,
    )
)

gaussian_kernel = tuple(
    bundle.get(
        "gaussian_kernel",
        DEFAULT_GAUSSIAN_KERNEL,
    )
)

gaussian_sigma = float(
    bundle.get(
        "gaussian_sigma",
        DEFAULT_GAUSSIAN_SIGMA,
    )
)

classes = list(
    bundle.get(
        "classes",
        DEFAULT_LABELS,
    )
)


uploaded_file = st.file_uploader(
    "Upload a banana image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
    ],
)


if uploaded_file is None:
    st.info(
        "Upload a banana image to start the prediction."
    )

    st.stop()


try:
    image_bgr = decode_uploaded_image(
        uploaded_file
    )

    (
        original_rgb,
        resized_rgb,
        gaussian_rgb,
    ) = eugene_preprocess_rgb_from_bgr(
        image_bgr=image_bgr,
        processing_size=processing_size,
        gaussian_kernel=gaussian_kernel,
        gaussian_sigma=gaussian_sigma,
    )

    watershed = run_marker_controlled_watershed(
        gaussian_rgb,
        bundle,
    )

    features = rgb_pixel_vector(
        watershed[
            "colour_cutout"
        ],
        svm_image_size,
    )

    expected_feature_length = bundle.get(
        "feature_length"
    )

    if (
        expected_feature_length
        is not None
        and len(features)
        != int(expected_feature_length)
    ):
        raise ValueError(
            "Feature length mismatch. "
            f"Expected {expected_feature_length}, "
            f"received {len(features)}."
        )

    feature_row = features.reshape(
        1,
        -1,
    )

    predicted_class = (
        model.predict(
            feature_row
        )[0]
    )

    probabilities = None
    confidence = None
    class_names = None

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = (
            model.predict_proba(
                feature_row
            )[0]
        )

        if (
            hasattr(
                model,
                "named_steps",
            )
            and "svm"
            in model.named_steps
        ):
            class_names = list(
                model.named_steps[
                    "svm"
                ].classes_
            )

        elif hasattr(
            model,
            "classes_",
        ):
            class_names = list(
                model.classes_
            )

        if (
            class_names
            is not None
            and predicted_class
            in class_names
        ):
            predicted_index = (
                class_names.index(
                    predicted_class
                )
            )

            confidence = float(
                probabilities[
                    predicted_index
                ]
            )

    region_features = (
        extract_region_features(
            watershed[
                "final_mask"
            ]
        )
    )

except Exception as exc:
    st.exception(
        exc
    )

    st.stop()


st.subheader(
    "Prediction"
)

left, right = st.columns(
    [1, 1]
)

with left:
    st.image(
        original_rgb,
        caption="Uploaded image",
        use_container_width=True,
    )

with right:
    st.metric(
        "Predicted ripeness",
        str(
            predicted_class
        ).upper(),
    )

    if confidence is not None:
        st.metric(
            "Prediction confidence",
            f"{confidence * 100:.2f}%",
        )

    if "test_accuracy" in bundle:
        st.metric(
            "Model held-out test accuracy",
            f"{float(bundle['test_accuracy']) * 100:.2f}%",
        )

    st.caption(
        "Prediction confidence is for this image only. "
        "It is different from the model's overall test accuracy."
    )


st.subheader(
    "Marker-Controlled Watershed Pipeline"
)

row1 = st.columns(
    4
)

with row1[0]:
    st.image(
        resized_rgb,
        caption=(
            f"1. Resize "
            f"({processing_size[0]}×{processing_size[1]})"
        ),
        use_container_width=True,
    )

with row1[1]:
    st.image(
        gaussian_rgb,
        caption=(
            f"2. Gaussian filter "
            f"{gaussian_kernel}"
        ),
        use_container_width=True,
    )

with row1[2]:
    st.image(
        watershed[
            "gray"
        ],
        caption="3. Grayscale",
        clamp=True,
        use_container_width=True,
    )

with row1[3]:
    st.image(
        watershed[
            "gradient"
        ],
        caption="4. Sobel gradient",
        clamp=True,
        use_container_width=True,
    )


marker_display = cv2.normalize(
    watershed[
        "markers"
    ].astype(
        np.float32
    ),
    None,
    0,
    255,
    cv2.NORM_MINMAX,
).astype(
    np.uint8
)

boundary_view = (
    gaussian_rgb.copy()
)

boundary_view[
    watershed[
        "watershed_labels"
    ]
    == -1
] = [
    255,
    255,
    255,
]


row2 = st.columns(
    4
)

with row2[0]:
    st.image(
        marker_display,
        caption="5. Foreground/background markers",
        clamp=True,
        use_container_width=True,
    )

with row2[1]:
    st.image(
        boundary_view,
        caption="6. Watershed boundaries",
        use_container_width=True,
    )

with row2[2]:
    st.image(
        watershed[
            "final_mask"
        ],
        caption="7. Final banana mask",
        clamp=True,
        use_container_width=True,
    )

with row2[3]:
    st.image(
        watershed[
            "colour_cutout"
        ],
        caption="8. Colour banana cut-out",
        use_container_width=True,
    )


st.subheader(
    "Watershed Region Features"
)

region_df = pd.DataFrame(
    {
        "Feature": REGION_FEATURE_NAMES,
        "Value": region_features,
    }
)

st.dataframe(
    region_df,
    use_container_width=True,
    hide_index=True,
)


if (
    probabilities is not None
    and class_names is not None
):
    st.subheader(
        "SVM Class Probabilities"
    )

    probability_df = pd.DataFrame(
        {
            "Class": class_names,
            "Probability": probabilities,
        }
    ).sort_values(
        "Probability",
        ascending=False,
    )

    probability_df[
        "Probability (%)"
    ] = (
        probability_df[
            "Probability"
        ]
        * 100
    ).round(
        2
    )

    st.dataframe(
        probability_df[
            [
                "Class",
                "Probability (%)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        probability_df.set_index(
            "Class"
        )[
            "Probability"
        ]
    )


st.divider()

st.caption(
    "Pipeline: Upload → Resize → Gaussian → Grayscale → "
    "Sobel Gradient → Marker-Controlled Watershed → "
    "Colour Banana Cut-Out → 64×64 RGB Features → "
    "StandardScaler → RBF SVM → Ripeness Prediction"
)
