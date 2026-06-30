<div align="center">

# 🌿 AgriVision -- AI powered crop disease and pest diagnosis

### End-to-End Crop Disease & Pest Detection for Indian Agriculture

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

A production-ready computer vision pipeline that identifies **172 crop diseases and pests** from leaf images. Provides an actionable diagnosis with Grad-CAM heatmaps, severity assessment, estimated yield loss in ₹/acre, bilingual treatment recommendations, and auto-generated PDF reports — all from a desktop GUI or CLI.

</div>

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Multi-architecture CNN** | ConvNeXt-Tiny (default), EfficientNet-B4, Swin-T — swappable via config |
| 🗂️ **172 Classes** | Covers diseases & pests across rice, wheat, maize, cotton, potato, sugarcane, and more |
| 🔥 **Grad-CAM Heatmaps** | Visual explanation of which leaf regions triggered the prediction |
| 📊 **Damage Assessment** | Maps confidence → severity stage → estimated yield loss (₹/acre) |
| 🗺️ **ICAR Zone Filtering** | 15 agro-climatic zones mapped to Indian states for context-aware results |
| 🌐 **Regional Languages** | Treatment recommendations in Hindi, Tamil, Telugu, Kannada, Marathi |
| 📄 **PDF Reports** | Bilingual diagnosis reports generated via Jinja2 + xhtml2pdf |
| 🖥️ **Desktop GUI** | Tkinter-based GUI for upload, diagnosis, heatmap view, and report export |
| ⌨️ **CLI Pipeline** | Full pipeline runnable from the command line on a single image |

---

## 📁 Project Structure

```text
agri_disease_detector/
├── data/
│   ├── augmented/                # Augmented training images (gitignored — large)
│   ├── disease_db.json           # Disease → severity → yield_loss → solutions + translations
│   └── dataset_health.json       # Dataset class distribution summary
├── models/
│   ├── build_model.py            # ConvNeXt-Tiny / EfficientNet-B4 / Swin-T builder
│   └── saved/                    # Trained weights (gitignored — large binary files)
│       ├── best_model.pth
│       └── best_model_checkpoint.pth
├── training/
│   ├── train.py                  # Two-phase transfer learning + MixUp/CutMix pipeline
│   └── evaluate.py               # Accuracy, F1, confusion matrix, per-class report
├── preprocessing/
│   ├── validate_image.py         # Format, resolution, blur detection (Laplacian)
│   ├── preprocess.py             # Resize 224×224, ImageNet normalization
│   ├── augment.py                # Albumentations augmentation pipeline
│   └── dataset_analyzer.py       # Class balance and dataset health checker
├── inference/
│   ├── predict.py                # Load model, run inference, return result dict
│   ├── gradcam.py                # Grad-CAM heatmap generation
│   ├── explain.py                # SHAP / saliency explanations
│   └── confidence_gate.py        # Blocks predictions below confidence threshold
├── analysis/
│   ├── damage_assessment.py      # Severity stage + yield loss (₹/acre)
│   └── recommendation.py         # Fetches treatment from disease_db.json
├── pdf_generator/
│   ├── generator.py              # Renders Jinja2 template → PDF via xhtml2pdf
│   └── templates/report.html     # Bilingual HTML report template
├── gui/
│   ├── app.py                    # Main Tkinter application window
│   ├── styles.py                 # Font, color, and padding constants
│   └── components/               # Image panel, result panel, heatmap panel, action panel
├── utils/
│   ├── config.py                 # Central config: paths, thresholds, zones, languages
│   ├── logger.py                 # File + console logging setup
│   └── helpers.py                # Shared utilities (image loading, base64 encoding)
├── tests/                        # Unit tests for preprocessing, inference, PDF
├── scripts/                      # Dataset utility scripts (flatten, expand DB)
├── main.py                       # CLI entry point
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.9+
- (Recommended) NVIDIA GPU with CUDA for training

### 1. Clone the repository
```bash
git clone https://github.com/your-username/agri_disease_detector.git
cd agri_disease_detector
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above.

### 4. Configure environment
```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux / macOS
```
Edit `.env` to point `MODEL_PATH` to your trained model file.

---

## 🗃️ Dataset Preparation

1. **Obtain datasets** — IP102, Paddy Doctor, PlantVillage, or similar.
2. **Organize** images into `data/augmented/<class_name>/` folders.
3. **Augment** (recommended) to balance class distribution:
   ```bash
   python preprocessing/augment.py
   ```
4. **Check health** of the dataset:
   ```bash
   python preprocessing/dataset_analyzer.py
   ```

The model automatically discovers class names from `data/augmented/` at runtime — no manual label file editing needed.

---

## 🏋️ Training

The pipeline uses **two-phase transfer learning** with PyTorch:

| Phase | Backbone | Head | LR | Epochs |
|---|---|---|---|---|
| 1 — Feature Extraction | Frozen | Trainable | `1e-3` | up to 20 |
| 2 — Fine-Tuning | Unfrozen | Trainable | `1e-5` | up to 30 |

**Advanced training features:**
- Mixed Precision Training (AMP) for faster GPU training
- MixUp / CutMix batch augmentation
- Cosine Annealing LR with Warm Restarts
- Class-weighted loss to handle class imbalance (172 classes)
- Early stopping (patience=12) + best-model checkpointing
- TensorBoard logging

```bash
python training/train.py
```

To switch model architecture, set `MODEL_ARCHITECTURE` in `.env`:
```
MODEL_ARCHITECTURE=convnext_tiny   # default
MODEL_ARCHITECTURE=efficientnet_b4
MODEL_ARCHITECTURE=swin_t
```

Saved weights → `models/saved/best_model.pth`

---

## 🖥️ Running the Desktop GUI

```bash
python gui/app.py
```

1. Upload a leaf image
2. View the Grad-CAM heatmap overlaid on the image
3. Read the diagnosis — disease name, confidence, severity, yield loss
4. Select regional language and click **Generate PDF Report**

---

## ⌨️ Running the CLI Pipeline

```bash
python main.py --image path/to/leaf.jpg --crop rice --state "Tamil Nadu" --language ta --output report.pdf
```

| Argument | Description | Default |
|---|---|---|
| `--image` | Path to the input leaf image | *(required)* |
| `--crop` | Crop type (rice, wheat, cotton…) | `Unknown` |
| `--state` | Indian state for ICAR zone filtering | `Unknown` |
| `--language` | Regional language code | `hi` |
| `--output` | Output PDF path | *(optional)* |

**Supported language codes:** `hi` Hindi · `ta` Tamil · `te` Telugu · `kn` Kannada · `mr` Marathi

```bash
python main.py --help
```

---

## 🧪 Running Tests

```bash
python -m pytest tests/
```

---

## 🌿 Supported Crops

`rice` · `wheat` · `maize` · `cotton` · `potato` · `tomato` · `sugarcane` · `chilli` · `soybean`

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
