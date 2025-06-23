# -*- coding: utf-8 -*-
"""
Created on Tue May 13 09:23:16 2025

@author: madel
"""
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Global configuration and default parameters
# =============================================================================
LITHO_MIN_PCT = 5       # minimum percentage threshold for lithology classes
RANDOM_STATE = 0        # random seed for reproducibility
BATCH_SIZE = 64         # batch size for PyTorch DataLoader
LEARNING_RATE = 5e-4    # learning rate for optimizer
NUM_EPOCHS = 40         # number of training epochs for neural network
HIDDEN_SIZE = 128       # hidden layer size for MLP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. Data loading and initial filtering
# =============================================================================
def load_data(geo_path, cpt_path, depth_threshold=-15):
    """
    Load geographic and CPT data from HDF5 files and apply a shallow depth filter.

    Parameters:
        geo_path (str): Path to geographic data .h5 file.
        cpt_path (str): Path to CPT data .h5 file.
        depth_threshold (float): Minimum allowed depth_to_reference.

    Returns:
        geo_df (pd.DataFrame): Geographic features.
        cpt_df (pd.DataFrame): CPT entries filtered by depth.
    """
    geo_df = pd.read_hdf(geo_path)
    cpt_df = pd.read_hdf(cpt_path)
    # Remove CPT samples above the specified depth threshold
    cpt_df = cpt_df[cpt_df['depth_to_reference'] > depth_threshold].copy()
    return geo_df, cpt_df

# =============================================================================
# 2. Lithology distribution and class filtering
# =============================================================================
def filter_lithology(cpt_df, min_pct=LITHO_MIN_PCT):
    """
    Compute lithology distribution and remove classes below the threshold.

    Parameters:
        cpt_df (pd.DataFrame): CPT data with lithology column.
        min_pct (float): Minimum percentage for classes to keep.

    Returns:
        filtered_df (pd.DataFrame): CPT data with infrequent classes removed.
    """
    # Compute percent distribution of each lithology class
    litho_perc = (cpt_df['lithology'].value_counts(normalize=True) * 100).sort_index()
    # Identify valid classes
    valid_classes = litho_perc[litho_perc >= min_pct].index.astype(float).tolist()
    # Retain only valid classes
    filtered_df = cpt_df[cpt_df['lithology'].isin(valid_classes)].copy()
    return filtered_df, valid_classes

# =============================================================================
# 3. Assign "New_Litho" based on majority within 50m radius
# =============================================================================
def compute_new_litho(cpt_df, geo_df, radius=50):
    """
    For each CPT profile, find nearby geo points and assign New_Litho by mode.

    Parameters:
        cpt_df (pd.DataFrame): CPT data with 'name', 'X', 'Y', 'lithology'.
        geo_df (pd.DataFrame): Geographic features including DEP_TOP_i, DEP_BOT_i.
        radius (float): Search radius in meters.

    Returns:
        cpt_df (pd.DataFrame): Updated CPT data with New_Litho column.
    """
    # Pre-extract depth arrays and coordinates
    depth_top = geo_df[[f"DEP_TOP_{i}" for i in range(1, 15)]].to_numpy()
    depth_bot = geo_df[[f"DEP_BOT_{i}" for i in range(1, 15)]].to_numpy()
    coords_geo = geo_df[['X', 'Y']].to_numpy()

    cpt_df['New_Litho'] = np.nan
    # Process each unique profile
    for profile in cpt_df['name'].unique():
        prof_df = cpt_df[cpt_df['name'] == profile]
        if prof_df.empty:
            continue
        x0, y0 = prof_df[['X', 'Y']].iloc[0].to_numpy()
        # Compute distances to all geo points
        distances = np.linalg.norm(coords_geo - np.array([x0, y0]), axis=1)
        nearby_idx = np.where(distances < radius)[0]
        if not nearby_idx.size:
            continue
        # Select the closest geo point
        nearest = nearby_idx[np.argmin(distances[nearby_idx])]
        # Assign New_Litho by mode within each interval
        for k in range(depth_top.shape[1]):
            top_depth = depth_top[nearest, k]
            bot_depth = depth_bot[nearest, k]
            mask = (
                (cpt_df['name'] == profile) &
                (cpt_df['depth_to_reference'] >= top_depth) &
                (cpt_df['depth_to_reference'] < bot_depth)
            )
            vals = cpt_df.loc[mask, 'lithology']
            if not vals.empty:
                cpt_df.loc[mask, 'New_Litho'] = vals.mode().iloc[0]
    return cpt_df

# =============================================================================
# 4. Compute average depths for each interval
# =============================================================================
def compute_avg_depth(geo_df):
    """
    Add AVG_DEPTH_i features to geo_df as midpoint between DEP_TOP and DEP_BOT.

    Parameters:
        geo_df (pd.DataFrame): Geographic data with TOPO, DEP_TOP_i, DEP_BOT_i.

    Returns:
        geo_df (pd.DataFrame): Updated with AVG_DEPTH_i columns.
    """
    topo = geo_df['TOPO'].to_numpy()[:, None]
    for i in range(1, 15):
        avg_col = f"AVG_DEPTH_{i}"
        top_col = f"DEP_TOP_{i}"
        bot_col = f"DEP_BOT_{i}"
        geo_df[avg_col] = topo.squeeze() - (geo_df[top_col] + geo_df[bot_col]) / 2
    return geo_df

# =============================================================================
# 5. Split profiles into train/test sets
# =============================================================================
def split_profiles(cpt_df, test_size=0.2, random_state=RANDOM_STATE):
    """
    Split unique profile names into training and testing sets.

    Parameters:
        cpt_df (pd.DataFrame): CPT data with 'name' column.
        test_size (float): Fraction of profiles for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_df, test_df (pd.DataFrame): Split CPT data.
    """
    all_names = cpt_df['name'].unique()
    train_names, test_names = train_test_split(
        all_names, test_size=test_size, random_state=random_state, shuffle=True
    )
    return (
        cpt_df[cpt_df['name'].isin(train_names)].copy(),
        cpt_df[cpt_df['name'].isin(test_names)].copy()
    )

# =============================================================================
# 6. Feature matrix and label vector builder
# =============================================================================
def build_X_y(df, geo_df, coords_geo):
    """
    Construct feature matrix X and label vector y from profiles.

    Parameters:
        df (pd.DataFrame): CPT subset (train or test).
        geo_df (pd.DataFrame): Geographic data with features.
        coords_geo (np.ndarray): Nx2 array of geo coordinates.

    Returns:
        X, y (np.ndarray): Features and labels.
    """
    X_list, y_list = [], []
    for profile in df['name'].unique():
        prof_df = df[df['name'] == profile]
        if prof_df.empty:
            continue
        x0, y0 = prof_df[['X', 'Y']].iloc[0].to_numpy()
        distances = np.linalg.norm(coords_geo - np.array([x0, y0]), axis=1)
        idxs = np.where(distances < 50)[0]
        if not idxs.size:
            continue
        local_geo = geo_df.iloc[idxs]
        for _, geo_row in local_geo.iterrows():
            for i in range(1, 15):
                top, bot = geo_row[f"DEP_TOP_{i}"], geo_row[f"DEP_BOT_{i}"]
                arr = prof_df.loc[
                    (prof_df['depth_to_reference'] >= top) &
                    (prof_df['depth_to_reference'] < bot),
                    'New_Litho'
                ]
                if arr.empty:
                    continue
                # Features: X, Y, AVG_DEPTH, RHO_I, DISTANCE_COASTLINE
                X_list.append([
                    geo_row['X'], geo_row['Y'],
                    geo_row[f"AVG_DEPTH_{i}"],
                    geo_row[f"RHO_I_{i}"],
                    geo_row['DISTANCE_COASTLINE']
                ])
                y_list.append(arr.mode().iloc[0])
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    # Remove any rows with NaNs
    valid_mask = ~np.isnan(X).any(axis=1)
    return X[valid_mask], y[valid_mask]

# =============================================================================
# 7. Data preprocessing: log-transform and standardization
# =============================================================================
def preprocess_features(X_train, X_test, em_feature_idx=3):
    """
    Apply log-transform to EM feature and standardize all features.

    Parameters:
        X_train, X_test (np.ndarray): Raw feature arrays.
        em_feature_idx (int): Index of EM feature in X columns.

    Returns:
        X_train_scaled, X_test_scaled, scaler (StandardScaler): Standardized data and fitted scaler.
    """
    eps = 1e-9
    X_train[:, em_feature_idx] = np.log(X_train[:, em_feature_idx] + eps)
    X_test[:,  em_feature_idx] = np.log(X_test[:,  em_feature_idx] + eps)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# =============================================================================
# 8. Train and evaluate a Random Forest classifier
# =============================================================================
def train_random_forest(X_train, y_train, class_weights, random_state=RANDOM_STATE):
    """
    Train a RandomForest classifier with provided class weights.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        class_weight=class_weights,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_classifier(clf, X_test, y_test, classes, title="Classifier Results"):
    """
    Print classification report and plot confusion matrix.
    """
    y_pred = clf.predict(X_test)
    print(f"\n--- {title} Evaluation ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title(f"Confusion Matrix – {title}", fontweight='bold')
    plt.tight_layout()
    plt.show()
    return y_pred

# =============================================================================
# 9. PyTorch dataset and model definitions
# =============================================================================
class LitDataset(Dataset):
    """
    Custom Dataset for lithology data.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    """
    Simple MLP with two hidden layers and dropout.
    """
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# =============================================================================
# 10. Train and evaluate neural network
# =============================================================================
def train_nn(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    """
    Training loop for PyTorch neural network.
    """
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs} – Loss: {avg_loss:.4f}")


def evaluate_nn(model, test_loader, idx2cls):
    """
    Evaluate trained PyTorch model on test data and plot results.
    """
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())

    y_pred_idx = np.concatenate(all_preds)
    y_true_idx = np.concatenate(all_true)
    # Map indices back to class labels
    y_pred = np.array([idx2cls[i] for i in y_pred_idx])
    y_true = np.array([idx2cls[i] for i in y_true_idx])

    print("\n--- Neural Network Evaluation ---")
    print(classification_report(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(idx2cls.values()))
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=list(idx2cls.values()))
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title("Confusion Matrix – Neural Network", fontweight='bold')
    plt.tight_layout()
    plt.show()

# =============================================================================
# Main execution flow
# =============================================================================
def main():
    # Paths to HDF5 data files
    geo_path = r"D:\Διπλωματική ΠΜΣ\git\coastal_data.h5"
    cpt_path = r"D:\Διπλωματική ΠΜΣ\git\cpt_data.h5"

    # 1. Load data
    geo_df, cpt_df = load_data(geo_path, cpt_path)

    # 2. Filter based on lithology frequency
    cpt_df, valid_classes = filter_lithology(cpt_df)

    # 3. Compute New_Litho labels
    cpt_df = compute_new_litho(cpt_df, geo_df)
    # Drop any samples with New_Litho == 3 as per initial logic
    cpt_df = cpt_df[cpt_df['New_Litho'] != 3].copy()

    # 4. Compute average depth intervals
    geo_df = compute_avg_depth(geo_df)

    # 5. Split profiles into train and test sets
    train_df, test_df = split_profiles(cpt_df)

    # 6. Build feature matrices and labels
    coords_geo = geo_df[['X', 'Y']].to_numpy()
    X_train, y_train = build_X_y(train_df, geo_df, coords_geo)
    X_test,  y_test  = build_X_y(test_df,  geo_df, coords_geo)
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # 7. Preprocess features
    X_train, X_test, scaler = preprocess_features(X_train, X_test)

    # 8. Class distribution and manual weights
    classes = sorted(pd.Series(y_train).unique())
    class_weights = {cls: w for cls, w in zip(classes, [1.0, 2.0, 1.5])}

    # 9. Train and evaluate Random Forest
    rf_clf = train_random_forest(X_train, y_train, class_weights)
    evaluate_classifier(rf_clf, X_test, y_test, classes, title="Random Forest")

    # 10. Prepare data for PyTorch
    cls2idx = {cls: i for i, cls in enumerate(classes)}
    idx2cls = {i: cls for cls, i in cls2idx.items()}
    y_train_idx = np.array([cls2idx[c] for c in y_train], dtype=np.int64)
    y_test_idx  = np.array([cls2idx[c] for c in y_test],  dtype=np.int64)
    weight_list = [class_weights[idx2cls[i]] for i in range(len(classes))]
    class_weights_tensor = torch.tensor(weight_list, dtype=torch.float32, device=DEVICE)

    train_ds = LitDataset(X_train, y_train_idx)
    test_ds  = LitDataset(X_test,  y_test_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # 11. Define model, loss, and optimizer
    model = MLP(X_train.shape[1], HIDDEN_SIZE, len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 12. Train neural network
    train_nn(model, train_loader, criterion, optimizer)

    # 13. Evaluate neural network
    evaluate_nn(model, test_loader, idx2cls)

if __name__ == "__main__":
    main()
