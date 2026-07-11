from pathlib import Path
from typing import List, Tuple

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = (APP_DIR / ".." / "models" / "custom_cnn_deployment_model.pt").resolve()


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


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Deployment model not found at {MODEL_PATH}. "
        "Expected app/app.py and models/custom_cnn_deployment_model.pt."
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

NUM_CLASSES = int(checkpoint["number_of_classes"])
IMAGE_SIZE = int(checkpoint["image_size"])
TEMPERATURE = float(checkpoint["temperature"])
CONFIDENCE_THRESHOLD = float(checkpoint["confidence_threshold"])

index_to_latex = {
    int(key): str(value)
    for key, value in checkpoint["index_to_latex"].items()
}

model = CustomCNN(NUM_CLASSES)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

preprocess = transforms.Compose(
    [
        transforms.Pad(8, fill=255, padding_mode="constant"),
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
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


def predict_symbol(image: Image.Image) -> Tuple[str, List[List[str]]]:
    if image is None:
        return "Upload an image first.", []

    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits / TEMPERATURE, dim=1)[0]

    top_probabilities, top_indices = torch.topk(probabilities, k=5)

    rows = []
    for rank, (probability, class_index) in enumerate(
        zip(top_probabilities.tolist(), top_indices.tolist()),
        start=1,
    ):
        rows.append(
            [
                str(rank),
                index_to_latex[int(class_index)],
                f"{probability * 100:.2f}%",
            ]
        )

    best_probability = float(top_probabilities[0])
    best_label = index_to_latex[int(top_indices[0])]

    if best_probability >= CONFIDENCE_THRESHOLD:
        status = (
            f"Accepted prediction: {best_label} "
            f"({best_probability * 100:.2f}% confidence)"
        )
    else:
        status = (
            f"Review recommended: {best_label} "
            f"({best_probability * 100:.2f}% confidence), below the "
            f"{CONFIDENCE_THRESHOLD:.2f} threshold."
        )

    return status, rows


with gr.Blocks(title="Handwritten Mathematical Symbol Classifier") as demo:
    gr.Markdown(
        """
# Explainable Handwritten Mathematical Symbol Classifier

Upload one cropped handwritten mathematical symbol. The model returns the five
most likely LaTeX labels and flags low-confidence predictions for review.

**Model:** Custom CNN trained on HASYv2  
**Classes:** 369  
**Input:** one isolated symbol
"""
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload handwritten symbol")

        with gr.Column():
            status_output = gr.Textbox(label="Decision", interactive=False)
            predictions_output = gr.Dataframe(
                headers=["Rank", "LaTeX label", "Probability"],
                datatype=["str", "str", "str"],
                label="Top-5 predictions",
                interactive=False,
            )

    predict_button = gr.Button("Classify symbol", variant="primary")
    gr.ClearButton([image_input, status_output, predictions_output])

    predict_button.click(
        predict_symbol,
        inputs=image_input,
        outputs=[status_output, predictions_output],
    )

    gr.Markdown(
        """
### Limitation

This prototype recognizes isolated symbols, not complete equations.
Visually equivalent notation variants may remain difficult to distinguish.
Low-confidence predictions should be reviewed.
"""
    )


if __name__ == "__main__":
    demo.launch()
