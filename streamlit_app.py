from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Math Symbol Classifier",
    page_icon="✍️",
    layout="wide",
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    APP_DIR / ".." / "models" / "custom_cnn_deployment_model.pt"
).resolve()


# ---------------------------------------------------------
# Model architecture
# ---------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 369):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------------------------------------------
# Load model once
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Deployment model not found at: {MODEL_PATH}. "
            "Expected repository structure: "
            "app/streamlit_app.py and "
            "models/custom_cnn_deployment_model.pt"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    number_of_classes = int(checkpoint["number_of_classes"])
    image_size = int(checkpoint["image_size"])
    temperature = float(checkpoint["temperature"])
    confidence_threshold = float(checkpoint["confidence_threshold"])

    index_to_latex: Dict[int, str] = {
        int(key): str(value)
        for key, value in checkpoint["index_to_latex"].items()
    }

    model = CustomCNN(number_of_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return (
        model,
        device,
        image_size,
        temperature,
        confidence_threshold,
        index_to_latex,
    )


try:
    (
        model,
        device,
        image_size,
        temperature,
        confidence_threshold,
        index_to_latex,
    ) = load_model()
except Exception as error:
    st.error("The model could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
preprocess = transforms.Compose(
    [
        transforms.Pad(
            padding=8,
            fill=255,
            padding_mode="constant",
        ),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def predict_symbol(image: Image.Image) -> pd.DataFrame:
    input_tensor = (
        preprocess(image.convert("RGB"))
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        logits = model(input_tensor)
        calibrated_logits = logits / temperature
        probabilities = torch.softmax(
            calibrated_logits,
            dim=1,
        )[0]

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=5,
    )

    rows = []

    for rank, (probability, class_index) in enumerate(
        zip(
            top_probabilities.tolist(),
            top_indices.tolist(),
        ),
        start=1,
    ):
        rows.append(
            {
                "Rank": rank,
                "LaTeX label": index_to_latex[int(class_index)],
                "Probability": probability,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Interface
# ---------------------------------------------------------
st.title("Explainable Handwritten Mathematical Symbol Classifier")

st.markdown(
    """
Upload a cropped image containing **one handwritten mathematical symbol**.
The model returns the five most likely LaTeX labels and flags uncertain
predictions for review.

**Model:** Custom CNN trained on HASYv2  
**Number of classes:** 369  
**Confidence threshold:** 0.60
"""
)

left_column, right_column = st.columns(
    [1, 1.4],
    gap="large",
)

with left_column:
    uploaded_file = st.file_uploader(
        "Upload a symbol image",
        type=["png", "jpg", "jpeg"],
        help="Use a tightly cropped image containing one isolated symbol.",
    )

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")

        st.image(
            uploaded_image,
            caption="Uploaded symbol",
            use_container_width=True,
        )

with right_column:
    st.subheader("Prediction")

    if uploaded_file is None:
        st.info("Upload an image to begin.")
    else:
        if st.button(
            "Classify symbol",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Running inference..."):
                results = predict_symbol(uploaded_image)

            best_label = str(
                results.iloc[0]["LaTeX label"]
            )

            best_probability = float(
                results.iloc[0]["Probability"]
            )

            if best_probability >= confidence_threshold:
                st.success(
                    f"Accepted prediction: `{best_label}` "
                    f"with {best_probability * 100:.2f}% confidence."
                )
            else:
                st.warning(
                    f"Review recommended: top prediction is "
                    f"`{best_label}` with "
                    f"{best_probability * 100:.2f}% confidence, "
                    f"below the {confidence_threshold:.2f} threshold."
                )

            display_results = results.copy()
            display_results["Probability"] = (
                display_results["Probability"]
                .map(lambda value: f"{value * 100:.2f}%")
            )

            st.dataframe(
                display_results,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "The probabilities use temperature scaling fitted "
                "on the validation set."
            )


st.divider()

st.subheader("Important limitations")

st.markdown(
    """
- The prototype recognizes isolated symbols, not complete equations.
- Some notation variants are visually almost identical without equation context.
- Rare classes may be less reliable than frequent classes.
- Low-confidence predictions should be reviewed by a person.
"""
)

with st.expander("Model information"):
    st.markdown(
        f"""
- Input size: **{image_size} × {image_size} RGB**
- Deployment model: **Custom CNN**
- Parameters: **1,333,841**
- Temperature: **{temperature:.4f}**
- Acceptance threshold: **{confidence_threshold:.2f}**
- Runtime device: **{device}**
"""
    )
