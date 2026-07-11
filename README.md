# Explainable Handwritten Mathematical Symbol Classification

An explainable CNN-based framework for recognizing 369 handwritten mathematical symbol classes from the HASYv2 dataset. The project compares a lightweight custom CNN with EfficientNetV2-S and ConvNeXt-Tiny and includes calibration, confidence-based abstention, robustness testing, Grad-CAM, and SHAP explanations.

## Live Demo

Try the deployed classifier:

https://appapppy-xdz5vdkqnd7pdmnpmhjpls.streamlit.app/

## Project Objective

The objective is to develop a practical handwritten mathematical symbol classifier for digital homework transcription and intelligent learning support. The system predicts an isolated symbol, returns ranked alternatives, and flags uncertain predictions for human review.

## Dataset

The project uses the public HASYv2 dataset:

- 168,233 symbol images
- 369 classes
- Original image resolution: 32 × 32 pixels
- Development set: 136,116 images
- Validation set: 15,125 images
- Locked official test set: 16,992 images

The official HASYv2 fold 1 test partition was preserved for final evaluation.

## Models

Three CNN architectures were evaluated:

1. Custom CNN trained from random initialization
2. ImageNet-pretrained EfficientNetV2-S
3. ImageNet-pretrained ConvNeXt-Tiny

The transfer-learning models were trained using frozen-backbone classifier training followed by partial fine-tuning.

## Final Test Results

| Model | Top-1 Accuracy | Top-5 Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Custom CNN | **81.11%** | **98.14%** | **68.59%** | **79.49%** |
| EfficientNetV2-S | 78.35% | 96.75% | 66.50% | 77.27% |
| ConvNeXt-Tiny | 76.82% | 96.21% | 64.74% | 76.04% |

The Custom CNN achieved the strongest predictive results while also being the smallest and fastest model.

## Deployment Model

The selected deployment model is the Custom CNN:

- 1,333,841 parameters
- 5.12 MB deployment checkpoint
- Approximately 3.04 ms single-symbol GPU latency
- Approximately 7,491 symbols per second in batches of 128

## Calibration and Confidence-Aware Review

Temperature scaling reduced Expected Calibration Error:

- Validation ECE: 12.54% → 1.54%
- Test ECE: 12.78% → 1.17%

A validation-selected confidence threshold of 0.60 produced:

- Test coverage: 79.04%
- Accepted-prediction accuracy: 89.81%
- Abstention rate: 20.96%

Predictions below the threshold are intended for human review.

## Robustness Results

| Condition | Top-1 Accuracy | Macro-F1 |
|---|---:|---:|
| Clean | 80.52% | 67.22% |
| Rotation +10° | 78.80% | 64.42% |
| Gaussian blur | 80.60% | 67.29% |
| Thickened strokes | 43.58% | 29.58% |
| Partial erasure | 67.58% | 53.11% |

The model was stable under mild rotation and blur but sensitive to stroke-thickness changes and partial symbol loss.

## Explainable AI

The project includes:

- Grad-CAM explanations from the final convolutional layer
- SHAP GradientExplainer analysis
- Correct and incorrect prediction examples
- High- and low-confidence examples
- Signed positive and negative SHAP attributions

Both explanation methods primarily highlighted handwritten stroke regions and revealed cases where subtle distinguishing strokes received insufficient attention.

## Repository Structure

- `docs/` — technical results summary
- `figures/` — training, evaluation, robustness, and explainability figures
- `metadata/` — label mappings and split summaries
- `models/` — deployment-ready Custom CNN checkpoint
- `notebook/` — executed Kaggle notebook
- `results/` — CSV evaluation tables and prediction outputs

## Main Technologies

- Python
- PyTorch
- torchvision
- scikit-learn
- SHAP
- Matplotlib
- Pandas
- Kaggle GPU environment

## Reproducibility Notes

- Random seed: 42
- Image size: 96 × 96 RGB
- Class-weighted cross-entropy with label smoothing
- Validation macro-F1 used for checkpoint selection
- Locked test data was evaluated only after model development
- Confidence calibration and threshold selection used validation data only

## Limitations

- Evaluation used one public dataset and one official fold.
- The official partitions were not writer-disjoint.
- Several rare classes contained very few test examples.
- SHAP was evaluated on a small representative sample.
- Robustness to stroke thickening remained limited.
- The system recognizes isolated symbols rather than complete equations.

## Author

Emilio Alexander Zamora Wong  
University of Europe for Applied Sciences
