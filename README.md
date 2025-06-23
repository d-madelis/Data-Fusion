# Data-Fusion: IC & Lithology Estimation with EM & CPT Data

A Python project implementing both **regression** (IC prediction) and **classification** (lithology prediction) pipelines using Electromagnetic (EM) and Cone Penetration Test (CPT) data. Features include data preprocessing, spatial correlation, interpolation, Random Forest models and PyTorch neural nets.

---

## 📁 Repository Structure

Data-Fusion/
├── data/
│   ├── raw/                 # Πρωτογενή HDF5 αρχεία
│   │   ├── coastal_data.h5
│   │   └── cpt_data.h5
│   ├── processed/           # Επεξεργασμένα NumPy arrays
│   │   ├── X_train.npy, y_train.npy, …
│   │   └── depth_train.npy, layer_train.npy, …
│   └── splits/              # Train/Val split ονομάτων προφίλ
│       ├── train_names.npy
│       └── val_names.npy
│
├── plots/
│   ├── interp_checks/       # Εικόνες interpolation (CPT vs EM)
│   └── predictions/
│       ├── pred_vs_true.png
│       └── profiles/        # Προφίλ IC (true vs pred)
│
├── src/                     # (προαιρετικά) helper modules
│
├── classification.py        # Pipeline classification λιθολογίας
├── regression.py            # Pipeline πρόβλεψης IC (regression)
├── README.md                # Αυτό το αρχείο
└── requirements.txt         # Python dependencies
