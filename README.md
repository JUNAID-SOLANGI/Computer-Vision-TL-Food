# 🍕 Food Image Classifier

> Transfer Learning + Fine-Tuning with MobileNetV2 · 10 Food Classes · TensorFlow 2

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)](https://tensorflow.org)
[![MobileNetV2](https://img.shields.io/badge/MobileNetV2-ImageNet-green?style=flat-square)](https://keras.io/api/applications/mobilenet/)
[![Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-yellow?style=flat-square&logo=googlecolab)](https://colab.research.google.com/)
[![GPU](https://img.shields.io/badge/GPU-T4-red?style=flat-square)](https://cloud.google.com/compute/docs/gpus)

A two-stage deep learning pipeline that classifies food images into 10 categories using **MobileNetV2** pretrained on ImageNet. The project demonstrates the power of transfer learning with only 10% of training data per class, followed by selective fine-tuning of the upper layers for improved accuracy.

---

## 🗂 Dataset

The model is trained on the **Food-101** (10-class subset), with only **10% of data per class** to simulate a limited-data scenario.

| Property | Value |
|----------|-------|
| Classes | 10 food categories |
| Training split | 10% of original data |
| Image size | 224 × 224 px |
| Batch size | 32 |
| Augmentations | Horizontal flip, random rotation (±0.2 rad) |

**Classes:** Pizza · Burger · Sushi · Ramen · Steak · Ice Cream · Donuts · Salad · Noodles · Chicken Wing

---

## 🏗 Model Architecture

The pipeline uses a frozen MobileNetV2 backbone as a feature extractor with a custom classification head, then unfreezes the top layers for fine-tuning.

```
Input (224, 224, 3)
       │
       ▼
Data Augmentation (RandomFlip, RandomRotation)
       │
       ▼
MobileNetV2 Base (ImageNet weights, frozen)   ← Stage 1: Feature Extraction
       │
       ▼
GlobalAveragePooling2D
       │
       ▼
Dropout (0.2)
       │
       ▼
Dense (10, softmax)                           ← Custom Head
       │
       ▼
[Unfreeze layers 120+, retrain at lr/10]      ← Stage 2: Fine-Tuning
```

---

## ⚙️ Training Configuration

### Stage 1 — Feature Extraction

| Parameter | Value |
|-----------|-------|
| Base model | MobileNetV2 (frozen) |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Loss | SparseCategoricalCrossentropy |
| Max epochs | 100 |
| Early stopping patience | 10 |
| LR reduction patience | 3 (factor 0.5) |

### Stage 2 — Fine-Tuning

| Parameter | Value |
|-----------|-------|
| Unfrozen layers | 120 onwards (out of 155) |
| Learning rate | `1e-5` (base_lr / 10) |
| Max epochs | 100 additional |
| Model checkpoint | `best_tuned_model.keras` |

### Callbacks
- **`ModelCheckpoint`** — saves the best model by `val_loss`
- **`EarlyStopping`** — halts training if `val_loss` stagnates for 10 epochs
- **`ReduceLROnPlateau`** — halves learning rate on plateau (patience=3)

---

## 📁 Project Structure

```
.
├── TL_Food.ipynb            # Main notebook
├── best_tuned_model.keras   # Saved best model (generated after training)
└── 10_food_classes_10_percent/
    ├── train/
    │   └── {class_name}/
    │       └── *.jpg
    └── test/
        └── {class_name}/
            └── *.jpg
```

---

## 🚀 Quick Start

### 1. Open in Google Colab

Click the Colab badge above or open `TL_Food.ipynb` manually.

### 2. Upload the dataset

Upload `archive.zip` (Food-101 10-class subset) to `/content/` in your Colab environment.

```
/content/archive.zip
```

The notebook extracts it automatically.

### 3. Set the runtime

Go to **Runtime → Change runtime type → T4 GPU**.

### 4. Run all cells

```
Runtime → Run all  (Ctrl+F9)
```

Training will run in two stages automatically. The best model is saved to `best_tuned_model.keras`.

---

## 📊 Evaluation

The notebook produces the following outputs after each training stage:

- **Accuracy & loss curves** — side-by-side plots with a vertical marker showing where fine-tuning begins
- **Confusion matrix** — heatmap (Seaborn) for all 10 classes on the test set
- **Per-image predictions** — 3×3 grid with true vs. predicted labels (green = correct, red = wrong)
- **Side-by-side comparison table** — base model vs. fine-tuned model on test loss and accuracy

---

## 🧪 Dependencies

```python
tensorflow >= 2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
```

All dependencies are pre-installed in Google Colab. No `pip install` needed.

---

## 💡 Key Concepts Demonstrated

- **Transfer learning** from ImageNet to a domain-specific task with limited data
- **Feature extraction** by freezing a pretrained backbone
- **Selective fine-tuning** — only upper layers (120+) are unfrozen to preserve low-level features
- **Data augmentation** to improve generalization on small datasets
- **Callback-driven training** with early stopping, LR scheduling, and checkpointing
- **Confusion matrix analysis** to identify class-level weaknesses

---

## 📌 Notes

- The dataset uses only 10% of Food-101 — a deliberate constraint to test transfer learning's effectiveness under limited data conditions.
- MobileNetV2 was chosen for its efficiency and strong ImageNet features, making it ideal for small-data transfer learning.
- Fine-tuning at a lower learning rate (`base_lr / 10`) prevents catastrophic forgetting of pretrained features.

---

## 📄 License

This project is for educational purposes. The Food-101 dataset is subject to its own license — see the [original dataset page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).
