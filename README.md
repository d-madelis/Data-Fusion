# Data-Fusion

> **IC & Lithology Estimation** using EM & CPT data through regression and classification pipelines.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) \[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup & Installation](#setup--installation)
4. [Usage](#usage)

   * [Regression Pipeline](#regression-pipeline)
   * [Classification Pipeline](#classification-pipeline)
5. [Requirements](#requirements)
6. [Contributing](#contributing)
7. [License](#license)

---

## 📝 Project Overview

This project implements two core workflows for estimating Induction Conductivity (IC) and predicting lithology classes using:

* Electromagnetic (EM) data (`coastal_data.h5`)
* Cone Penetration Test (CPT) data (`cpt_data.h5`)

Features:

* Data preprocessing & smoothing
* Spatial alignment & interpolation
* Random Forest & PyTorch-based models
* Cross-validation and hyperparameter search
* Diagnostic plots and evaluation metrics

---

## 📁 Repository Structure

```
DData-Fusion/
├── data/
│   ├── raw/                 # Original HDF5 files
│   │   ├── coastal_data.h5
│   │   └── cpt_data.h5
│   ├── processed/           # NumPy arrays after preprocessing
│   └── splits/              # Train/val split identifiers
│       ├── train_names.npy
│       └── val_names.npy
├── results/                # Model predictions and diagnostics
│   ├── PyTorch/
│   │   ├── profiles/        # Profile plots for each well
│   │   ├── pt_pred_vs_true_per_profile_layer.png
│   │   └── pt_ratio_histogram.png
│   └── RF/
│       ├── profiles/        # Profile plots for each well
│       ├── pred_vs_true_per_profile_layer_de.png
│       └── ratio_histogram.png
├── plots/                   # Generated diagnostic and result plots
├── src/                     # Helper modules and utilities
├── regression.py            # IC regression pipeline
├── classification.py        # Lithology classification pipeline
├── requirements.txt         # Dependency list
├── license.txt              # Project license
└── README.md                # This file
```

---

## ⚙️ Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/Data-Fusion.git
   cd Data-Fusion
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # Linux/Mac
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Prepare data files**
   Place `coastal_data.h5` and `cpt_data.h5` into the `data/raw/` directory.

---

## 🚀 Usage

### Regression Pipeline

Run the full IC regression workflow:

```bash
python regression.py
```

**What it does:**

* Loads EM & CPT data
* Filters, smooths (Savitzky–Golay), and aligns profiles
* Interpolates IC onto EM layers
* Trains Random Forest with GroupKFold CV
* Performs PyTorch MLP grid search and final training
* Saves diagnostic plots under `plots/predictions/`

**Outputs:**

* Prediction plots (e.g., `pred_vs_true.png`)
* Profile comparison figures
* Serialized model artifacts (if configured)

---

### Classification Pipeline

Run the lithology classification workflow:

```bash
python classification.py
```

**What it does:**

* Loads & filters CPT data by depth and lithology frequency
* Assigns `New_Litho` labels via EM neighborhood mode
* Splits profiles into train/validation sets
* Constructs features (`X`) and labels (`y`), applies log-transform & scaling
* Trains Random Forest classifier (with class weights)
* Trains PyTorch MLP classifier
* Reports metrics & confusion matrix

**Outputs:**

* Printed classification report & confusion matrix
* Saved result plots in `plots/classification/`

---

## 🛠️ Requirements

* **Python**: 3.8+
* **HDF5 data**: `coastal_data.h5`, `cpt_data.h5` in `data/raw/`
* **Libraries** (from `requirements.txt`):

  ```
  numpy>=1.20
  pandas>=1.3
  scipy>=1.7
  matplotlib>=3.4
  scikit-learn>=1.0
  torch>=1.10
  h5py>=3.1
  ```

> **Optional**: For GPU support in PyTorch, install CUDA-enabled wheels:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
> ```

---

## 🤝 Contributing

We welcome bug fixes, enhancements, and new features!

1. Fork the repo and clone locally.
2. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature
   ```
3. Implement changes, add tests, and follow code style.
4. Commit and push:

   ```bash
   git add .
   git commit -m "Brief description"
   git push origin feature/your-feature
   ```
5. Open a Pull Request against the main repository.

Please include:

* A clear title & description
* Issue references (if applicable)
* Screenshots or examples (for visual changes)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

