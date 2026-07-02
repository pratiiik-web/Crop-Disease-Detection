# Crop Disease Detection

Detect plant diseases from leaf photos using deep learning and transfer
learning, helping farmers identify crop health issues early.

## Problem
Farmers often lose significant crop yield because plant diseases go
undetected until visible damage spreads. Early identification allows
faster treatment and reduces losses. This model classifies leaf images
into 38 disease categories across multiple crop species.

## Dataset
PlantVillage Dataset — 54,305 leaf images, 38 classes
Source: Kaggle (abdallahalidev/plantvillage-dataset)
Covers: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper,
Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## Approach — Transfer Learning
Rather than training a CNN from scratch, this project uses
**MobileNetV2** pretrained on ImageNet as a frozen feature extractor,
with a custom classification head trained on top.

- Base model: MobileNetV2 (frozen, 2.26M params)
- Custom head: GlobalAveragePooling + Dropout + Dense(38, softmax)
- Trainable params: only 48,678 — fast training, low overfitting risk
- Image size: 224x224, batch size: 16
- Trained on Kaggle GPU (Tesla T4) — ~10 minutes for 10 epochs

## Results

| Metric | Score |
|---|---|
| Training Accuracy | 96.98% |
| Validation Accuracy | 96.02% |
| Training Loss | 0.088 |
| Validation Loss | 0.133 |

Training and validation accuracy stayed closely aligned throughout
training, indicating the model generalizes well rather than overfitting.

## Known Limitations
- Performs best on clean, well-lit, close-up leaf images similar to
  the training distribution. Confidence drops on blurry or distant
  real-world phone photos.
- Occasionally confuses visually similar diseases across related plant
  species (e.g., Potato vs Tomato Late Blight), since these crops
  share the same disease pathogen (Phytophthora infestans) and nearly
  identical leaf symptoms. The model correctly identifies the disease
  type even when the plant species prediction is wrong — a reflection
  of genuine visual similarity in the dataset rather than a model flaw.

## How to Run
pip install -r requirements.txt
streamlit run apps/app.py

## Tech Stack
Python · TensorFlow/Keras · MobileNetV2 · Transfer Learning ·
Streamlit · PIL

## Project Structure
Crop Disease Detection/
├── Data/
│   └── plantvillage dataset/color/
├── Models/
│   ├── crop_disease_model.h5
│   └── class_names.json
├── Notebooks/
│   └── 01_model.ipynb
└── apps/
    └── app.py

## Live Demo
https://crop-disease-detection-ixx8yuwayeqmr3afghehmw.streamlit.app/


## Key Lesson
Transfer learning makes deep learning genuinely accessible without
massive compute. Freezing 2.26M pretrained ImageNet weights and
training only 48K parameters achieved 96% accuracy in under 10 minutes
on free GPU resources — proof that production-grade image classification
doesn't always require training from scratch.
