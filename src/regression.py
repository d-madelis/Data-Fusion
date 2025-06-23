# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 19:25:56 2025

@author: madel
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, ParameterSampler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List
import itertools

def load_data(geo_path: Path, cpt_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load geophysical and CPT datasets from HDF5 files.

    Parameters:
    - geo_path: Path to the geophysical data HDF5 file.
    - cpt_path: Path to the CPT data HDF5 file.

    Returns:
    - geo: DataFrame containing geophysical measurements.
    - cpt: DataFrame containing Cone Penetration Test data.
    """
    # Read HDF5 files into pandas DataFrames
    geo = pd.read_hdf(geo_path)
    cpt = pd.read_hdf(cpt_path)
    return geo, cpt


def preprocess_cpt(
    cpt: pd.DataFrame,
    depth_thresh: float = -15,
    window_length: int = 151,
    polyorder: int = 2
) -> pd.DataFrame:
    """
    Filter CPT observations by depth, smooth IC profiles, and remove invalid entries.

    Steps:
    1. Keep measurements shallower than the specified depth threshold.
    2. Preserve original IC values in a new column 'IC_raw'.
    3. Apply Savitzky-Golay smoothing to the IC profile for each unique probe.
    4. Replace infinite values with NaN and drop any rows with missing data.

    Parameters:
    - cpt: Raw CPT DataFrame with columns ['name', 'depth_to_reference', 'IC', ...].
    - depth_thresh: Minimum depth (negative value) to include in the dataset.
    - window_length: Window size for the Savitzky–Golay filter (must be odd).
    - polyorder: Polynomial order for the filter.

    Returns:
    - cpt: Cleaned and smoothed CPT DataFrame.
    """
    # Select only profiles shallower than threshold
    cpt = cpt[cpt['depth_to_reference'] > depth_thresh].copy()
    # Keep a copy of the raw IC measurements
    cpt['IC_raw'] = cpt['IC']

    # Smooth the IC profile for each probe name
    for name in cpt['name'].unique():
        mask = cpt['name'] == name
        cpt.loc[mask, 'IC_smooth'] = savgol_filter(
            cpt.loc[mask, 'IC'], window_length=window_length, polyorder=polyorder
        )

    # Convert infinite values to NaN, then drop all rows with NaNs
    cpt.replace([np.inf, -np.inf], np.nan, inplace=True)
    cpt.dropna(inplace=True)
    return cpt


def compute_em_layer_depths(
    geo: pd.DataFrame,
    num_layers: int = 14
) -> pd.DataFrame:
    """
    Compute the mean depth of each electromagnetic (EM) layer.

    The average depth is calculated as: TOP - (DEP_TOP + DEP_BOT) / 2

    Parameters:
    - geo: DataFrame containing topography (TOPO) and EM layer bounds.
    - num_layers: Number of EM layers to process.

    Returns:
    - geo: Original DataFrame augmented with columns ['AVG_DEPTH_1', ..., 'AVG_DEPTH_n'].
    """
    # Extract topography values and reshape for broadcasting
    topo = geo['TOPO'].values[:, None]
    # Define column names for layer top and bottom depths
    top_cols = [f'DEP_TOP_{i}' for i in range(1, num_layers+1)]
    bot_cols = [f'DEP_BOT_{i}' for i in range(1, num_layers+1)]

    # Retrieve top and bottom depth arrays
    dt = geo[top_cols].values
    db = geo[bot_cols].values

    # Calculate average depth for each layer
    avg_depth = topo - (dt + db) / 2

    # Add new columns back to the DataFrame
    for i in range(num_layers):
        geo[f'AVG_DEPTH_{i+1}'] = avg_depth[:, i]
    return geo


def build_kdtree(
    geo: pd.DataFrame
) -> tuple[cKDTree, np.ndarray]:
    """
    Construct a KD-tree for efficient spatial queries of EM station locations.

    Parameters:
    - geo: DataFrame containing X, Y coordinates of each station.

    Returns:
    - tree: cKDTree built from the station coordinates.
    - coords: Numpy array of original [X, Y] points.
    """
    coords = geo[['X', 'Y']].values  # Extract coordinate pairs
    tree = cKDTree(coords)           # Build KD-tree for fast neighbor lookup
    return tree, coords


def filter_cpt_by_proximity(
    cpt: pd.DataFrame,
    tree: cKDTree,
    coords: np.ndarray,
    max_dist: float = 50
) -> pd.DataFrame:
    """
    Retain only CPT probes with at least one EM measurement within max_dist meters.

    For each probe (unique 'name'), we check if any kd-tree points fall within
    the specified radius of the probe's XY location.

    Parameters:
    - cpt: CPT DataFrame with probe locations (X, Y).
    - tree: KD-tree of EM station locations.
    - coords: Array of EM station coordinates used to build the tree.
    - max_dist: Maximum search radius in meters.

    Returns:
    - Subset of the original CPT DataFrame with nearby EM data.
    """
    prox_names = []
    # Iterate through each unique probe name
    for name in cpt['name'].unique():
        # Get the first recorded location for this probe
        x0, y0 = cpt[cpt['name'] == name][['X', 'Y']].iloc[0]
        # Find EM stations within the specified distance
        idxs = tree.query_ball_point((x0, y0), max_dist)
        if idxs:
            prox_names.append(name)

    # Return only probes with at least one nearby EM station
    return cpt[cpt['name'].isin(prox_names)].copy()


def load_split_names(
    split_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training and validation profile names from saved NumPy arrays.

    Expects two files in split_dir:
    - 'train_names.npy'
    - 'val_names.npy'

    Parameters:
    - split_dir: Directory containing the split files.

    Returns:
    - train_names: Array of training profile identifiers.
    - val_names: Array of validation profile identifiers.
    """
    train_file = split_dir / 'train_names.npy'
    val_file = split_dir / 'val_names.npy'
    train_names = np.load(train_file, allow_pickle=True)
    val_names = np.load(val_file, allow_pickle=True)
    return train_names, val_names


def prepare_output_dirs(
    base_dir: Path
) -> dict[str, Path]:
    """
    Create directory structure for saving plots and model outputs.

    Directories created under base_dir:
    - 'plots/interp_checks'
    - 'plots/predictions'
    - 'plots/predictions/profiles'

    Parameters:
    - base_dir: Root directory for all outputs.

    Returns:
    - dirs: Dictionary mapping output types to their Path objects.
    """
    dirs = {
        'interp': base_dir / 'plots' / 'interp_checks',
        'pred': base_dir / 'plots' / 'predictions',
        'profiles': base_dir / 'plots' / 'predictions' / 'profiles'
    }
    # Ensure all directories exist
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def interpolate_and_plot(
    profiles: np.ndarray,
    filtered_cpt: pd.DataFrame,
    geo: pd.DataFrame,
    coords: np.ndarray,
    tree: cKDTree,
    em_depth_cols: list[str],
    em_res_cols: list[str],
    num_layers: int,
    output_dir: Path,
    is_train: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Interpolate smoothed CPT IC values at EM layer depths and generate diagnostic plots.

    For each profile name:
    1. Select nearby EM stations within 50m (limit to 10 closest if many).
    2. Interpolate CPT IC_smooth vs depth onto EM layer depth points.
    3. Compute median IC per layer and attach to local EM DataFrame.
    4. Plot individual interpolated CPT curves, original CPT profile, and mean resistivity profile.

    Parameters:
    - profiles: Array of unique CPT profile identifiers (names).
    - filtered_cpt: CPT DataFrame preprocessed (with IC_smooth).
    - geo: Geophysical DataFrame with EM readings & coordinates.
    - coords: Array of EM station [X,Y] coordinates used by tree.
    - tree: KD-tree built on EM station locations.
    - em_depth_cols: List of column names for EM layer depths.
    - em_res_cols: List of column names for EM resistivity logs.
    - num_layers: Number of EM layers.
    - output_dir: Directory to save generated figures.
    - is_train: Flag to control storing interpolation for validation.

    Returns:
    - all_data: Concatenated DataFrame of local EM+IC_smooth for all profiles.
    - interp_val: Dict of raw layer depths and mean IC for validation profiles (if is_train=False).
    """
    all_records = []
    interp_val = {} if not is_train else None

    # Loop through each profile
    for name in profiles:
        # Subset CPT for this profile
        df_cpt = filtered_cpt[filtered_cpt['name'] == name]
        # Get profile coordinates
        x0, y0 = df_cpt[['X', 'Y']].iloc[0]
        # Find EM stations within 50m
        idxs = tree.query_ball_point((x0, y0), 50)
        # If many stations, keep 10 nearest by Euclidean distance
        if len(idxs) > 10:
            dists = np.linalg.norm(coords[idxs] - np.array([x0, y0]), axis=1)
            nearest = np.argsort(dists)[:10]
            idxs = [idxs[i] for i in nearest]
        # Local EM subset for this profile
        local = geo.iloc[idxs].copy()
        local['name'] = name  # Tag with profile ID

        # Skip if insufficient stations
        if local.shape[0] <= 1:
            continue

        # Prepare CPT depth and IC arrays, removing NaNs
        depths = pd.to_numeric(df_cpt['depth_to_reference'], errors='coerce').values
        icv = pd.to_numeric(df_cpt['IC_smooth'], errors='coerce').values
        mask = ~np.isnan(depths) & ~np.isnan(icv)
        depths, icv = depths[mask], icv[mask]
        if len(depths) < 2:
            continue

        # Sort by depth for interpolation
        order = np.argsort(depths)
        depths, icv = depths[order], icv[order]

        n_geo = local.shape[0]
        # Matrix to hold interpolated IC for each EM station
        M = np.zeros((num_layers, n_geo))
        for k in range(n_geo):
            ed = local.iloc[k][em_depth_cols].astype(float).values
            # Interpolate IC at EM layer depths
            M[:, k] = np.interp(ed, depths, icv, left=np.nan, right=np.nan)

        # Compute median IC across stations for each layer
        mean_ic = np.nanmedian(M, axis=1)
        # Layer depths median for sorting
        layer_d = np.nanmedian(local[em_depth_cols].astype(float).values, axis=0)
        sidx = np.argsort(layer_d)
        layer_d, mean_ic = layer_d[sidx], mean_ic[sidx]

        # Save for validation if needed
        if not is_train:
            interp_val[name] = {'layer_d': layer_d.copy(), 'mean_ic': mean_ic.copy()}

        # Attach mean IC to local DataFrame
        for j in range(num_layers):
            local[f'IC_smooth{j+1}'] = mean_ic[j]
        all_records.append(local)

        # Plotting section
        title_prefix = 'Train' if is_train else 'Val'
        fig, ax = plt.subplots(figsize=(5, 6))
        # Plot individual interpolated CPT curves in light gray
        for k in range(n_geo):
            ax.plot(M[sidx, k], layer_d, lw=1, alpha=0.7)
        # Original CPT vs depth
        ax.plot(icv, depths, 'b-', lw=2, label='Original CPT')
        # Mean IC profile
        ax.plot(mean_ic, layer_d, 'orange', lw=2, label='Mean IC')
        ax.set_xlabel('IC_smooth')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{title_prefix} Interp: {name}', fontsize=14, fontweight='bold')
        ax.grid(True)
        # Secondary x-axis for EM resistivity logs
        ax2 = ax.twiny()
        # Plot individual EM resistivity curves
        for k in range(n_geo):
            ax2.plot(
                local.iloc[k][em_res_cols].astype(float).values,
                local.iloc[k][em_depth_cols].astype(float).values,
                alpha=0.6
            )
        # Mean resistivity profile
        mean_log = np.nanmedian(local[em_res_cols].astype(float).values, axis=0)
        ax2.plot(mean_log, mean_dep := np.nanmedian(local[em_depth_cols].astype(float).values, axis=0), lw=2)

        # Save figure and close
        outfile = output_dir / f"{title_prefix.lower()}_{name}.png"
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Combine all local EM records into one DataFrame
    all_data = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    return all_data, interp_val


def build_feature_matrix(
    df: pd.DataFrame,
    em_res_cols: list[str],
    em_depth_cols: list[str],
    num_layers: int,
    is_validation: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Assemble feature matrix X and target y for Random Forest training or validation.

    Each EM layer becomes one row in X with predictors:
    [X_coord, Y_coord, layer_depth, resistivity, distance_to_coast].
    Target y is the interpolated IC value for that layer.

    For validation, also return the layer indices.

    Returns:
    - X: Feature array of shape (n_samples, 5).
    - y: Target IC values (n_samples, ).
    - depth: Depth array corresponding to each sample.
    - prof_idx: Integer index of profile for grouping.
    - layer_idx (optional): Layer index for each sample (validation only).
    """
    uniq_names = df['name'].unique()
    X_rows, y_rows, depth_rows, prof_rows, layer_rows = [], [], [], [], []

    # Iterate over rows to expand by layer
    for _, row in df.iterrows():
        res_vals = row[em_res_cols].astype(float).values
        dep_vals = row[em_depth_cols].astype(float).values
        ic_vals = row[[f'IC_smooth{i+1}' for i in range(num_layers)]].values
        coast = row['DISTANCE_COASTLINE']
        x, y = row['X'], row['Y']
        # Profile ID for grouping
        prof_id = np.where(uniq_names == row['name'])[0][0]

        # Create one sample per layer
        for j in range(num_layers):
            X_rows.append([x, y, dep_vals[j], res_vals[j], coast])
            y_rows.append(ic_vals[j])
            depth_rows.append(dep_vals[j])
            prof_rows.append(prof_id)
            if is_validation:
                layer_rows.append(j)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=float)
    depth = np.array(depth_rows, dtype=float)
    prof_idx = np.array(prof_rows, dtype=int)
    layer_idx = np.array(layer_rows, dtype=int) if is_validation else None

    # Remove any samples with missing values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[mask], y[mask], depth[mask], prof_idx[mask], (layer_idx[mask] if is_validation else None)


def train_and_evaluate_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prof_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> tuple[RandomForestRegressor, np.ndarray, dict]:
    """
    Perform RandomizedSearchCV over hyperparameters, train Random Forest, and evaluate.

    Uses GroupKFold to prevent data leakage across profiles.

    Returns:
    - best_model: Trained RandomForestRegressor with optimal params.
    - y_pred: Predictions for test set.
    - metrics: Dict with RMSE and R² on test data.
    """
    # Hyperparameter search space
    param_dist = {
        'n_estimators': np.arange(100, 1001, 100),
        'max_depth': [None] + list(range(5, 51, 5)),
        'min_samples_split': np.arange(2, 21),
        'min_samples_leaf': np.arange(1, 11),
        'max_features': ['sqrt', 'log2', 0.2, 0.5, 0.8],
        'bootstrap': [True, False]
    }
    # Grouped CV to respect profiles
    gkf = GroupKFold(n_splits=5)

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=0),
        param_distributions=param_dist,
        n_iter=50,
        cv=gkf.split(X_train, y_train, prof_train),
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=0
    )
    # Fit search and extract best model
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Predict and compute metrics
    y_pred = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE = {rmse:.4f}, R² = {r2:.4f}")

    return best_model, y_pred, {'rmse': rmse, 'r2': r2}


def plot_predictions(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    depth_test: np.ndarray,
    prof_test: np.ndarray,
    layer_test: np.ndarray,
    output_dir: Path
) -> None:
    """
    Generate scatter and histogram plots comparing predicted vs actual IC values.

    1. Scatter: y_true vs y_pred colored by depth, with 1:1 reference line.
    2. Histogram: distribution of y_true/y_pred ratios.

    Saves figures to output_dir.
    """
    # Build DataFrame for aggregated plotting
    df = pd.DataFrame({
        'profile': prof_test,
        'layer': layer_test,
        'y_true': y_test,
        'y_pred': y_pred,
        'depth': depth_test
    })
    # Average per profile-layer combination
    df_avg = df.groupby(['profile', 'layer']).mean().reset_index()

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df_avg['y_true'], df_avg['y_pred'], c=df_avg['depth'], cmap='rainbow', s=10)
    lims = [df_avg[['y_true', 'y_pred']].min().min(), df_avg[['y_true', 'y_pred']].max().max()]
    ax.plot(lims, lims, 'k--', lw=1)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Depth (m)')
    ax.text(
        0.05, 0.95, f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}",
        ha='left', va='top', transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )
    ax.set_xlabel('Actual (y_true)')
    ax.set_ylabel('Predicted (y_pred)')
    ax.set_title('Pred vs True by Depth', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(output_dir / 'pred_vs_true.png', dpi=300)
    plt.close(fig)

    # Histogram of ratios
    ratios = df_avg['y_true'] / df_avg['y_pred']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(ratios, bins=50, range=(0.2, 2), edgecolor='black')
    ax.set_title('Histogram of y_true/y_pred', fontsize=14, fontweight='bold')
    ax.set_xlabel('True/Pred Ratio')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(output_dir / 'ratio_hist.png', dpi=300)
    plt.close(fig)


def plot_profile_curves(
    interp_val: dict,
    y_pred: np.ndarray,
    prof_test: np.ndarray,
    layer_test: np.ndarray,
    output_dir: Path,
    names: np.ndarray
) -> None:
    """
    Plot true vs predicted IC profiles for each validation profile.

    For each profile:
    - Reverse depth order for downward plots.
    - Plot true median IC vs depth and predicted vs depth.
    - Save each figure under output_dir.
    """
    for name, info in interp_val.items():
        # Get profile index for this name
        pid = np.where(names == name)[0][0]
        mask = (prof_test == pid)
        if not mask.any():
            continue

        true_ic = info['mean_ic']
        layer_d = info['layer_d']
        # Compute predicted IC by averaging predictions per layer
        pred_ic = np.array([
            np.nanmean(y_pred[(prof_test == pid) & (layer_test == j)])
            for j in range(len(layer_d))
        ])

        # Reverse for plotting top-down
        plot_d = layer_d[::-1]
        plot_true = true_ic[::-1]
        plot_pred = pred_ic[::-1]

        fig, ax = plt.subplots(figsize=(5, 6))
        ax.plot(plot_true, plot_d, 'b-', lw=2, label='True')
        ax.plot(plot_pred, plot_d, 'r-', lw=2, label='Pred')
        ax.set_xlabel('IC')
        ax.set_ylabel('Depth (m)')
        # Clean up profile name for title
        clean_name = name.replace('_IMBRO_A.gef', '')
        ax.set_title(f'Profile: {clean_name} (Random Forest)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True)
        fig.savefig(output_dir / f'profile_{name}.png', dpi=300)
        plt.close(fig)




def set_random_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


class IcPredictor(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size1: int = 64,
                 hidden_size2: int = 32,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


def train_model(params: dict,
                train_dataset: TensorDataset,
                X_val: torch.Tensor,
                y_val: torch.Tensor,
                device: torch.device,
                input_size: int,
                num_epochs: int = 100,
                patience: int = 10) -> float:
    train_loader = DataLoader(train_dataset,
                              batch_size=params['batch_size'],
                              shuffle=True)
    model = IcPredictor(input_size,
                        params['hidden_size1'],
                        params['hidden_size2'],
                        params['dropout_rate']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss


def grid_search_random(param_grid: dict,
                       train_dataset: TensorDataset,
                       val_dataset: Subset,
                       device: torch.device,
                       input_size: int,
                       n_iter: int = 20) -> dict:
    # Sample a limited number of parameter combinations randomly
    param_list = list(ParameterSampler(param_grid,
                                       n_iter=n_iter,
                                       random_state=42))
    # Pre-stack validation tensors once
    X_val = torch.stack([x for x, _ in val_dataset]).to(device)
    y_val = torch.stack([y for _, y in val_dataset]).to(device)

    best_score = float('inf')
    best_params = None

    for params in param_list:
        print(f"Testing params: {params}")
        val_loss = train_model(params,
                               train_dataset,
                               X_val,
                               y_val,
                               device,
                               input_size)
        print(f"Validation RMSE: {np.sqrt(val_loss):.4f}")
        if val_loss < best_score:
            best_score = val_loss
            best_params = params

    print(f"Best parameters: {best_params}")
    print(f"Best validation RMSE: {np.sqrt(best_score):.4f}")
    return best_params


def train_final_model(params: dict,
                      train_dataset: TensorDataset,
                      val_dataset: Subset,
                      device: torch.device,
                      input_size: int,
                      num_epochs: int = 200,
                      patience: int = 20,
                      save_path: str = 'best_model.pth') -> nn.Module:
    # Pre-stack validation tensors once
    X_val = torch.stack([x for x, _ in val_dataset]).to(device)
    y_val = torch.stack([y for _, y in val_dataset]).to(device)

    train_loader = DataLoader(train_dataset,
                              batch_size=params['batch_size'],
                              shuffle=True)
    model = IcPredictor(input_size,
                        params['hidden_size1'],
                        params['hidden_size2'],
                        params['dropout_rate']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()

        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load(save_path))
    return model


def evaluate_model(model: nn.Module,
                   test_dataset: TensorDataset,
                   device: torch.device,
                   batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0).flatten()
    y_true = test_dataset.tensors[1].cpu().numpy().flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Final Test RMSE: {rmse:.4f}")
    return y_true, y_pred, rmse

def main():
    # — Project base & output dirs —
    base = Path(r"D:/Διπλωματική ΠΜΣ/git")
    dirs = prepare_output_dirs(base)
    torch_output_dir = base / 'plots' / 'pytorch'
    torch_output_dir.mkdir(parents=True, exist_ok=True)

    # — 1) Load & preprocess data —
    geo, cpt = load_data(base/'coastal_data.h5', base/'cpt_data.h5')
    cpt       = preprocess_cpt(cpt)
    geo       = compute_em_layer_depths(geo)

    # — 2) Define layers & EM columns —
    num_layers    = 14
    em_depth_cols = [f'AVG_DEPTH_{i}' for i in range(1, num_layers+1)]
    em_res_cols   = [f'RHO_I_{i}'   for i in range(1, num_layers+1)]

    # — 3) Spatial filtering & splits —
    tree, coords            = build_kdtree(geo)
    cpt                     = filter_cpt_by_proximity(cpt, tree, coords, max_dist=50)
    train_names, val_names  = load_split_names(base/'splits')

    # — 4) Interpolation & RF inputs —
    train_df, _        = interpolate_and_plot(train_names, cpt, geo, coords, tree,
                                              em_depth_cols, em_res_cols, num_layers,
                                              dirs['interp'], is_train=True)
    val_df, interp_val = interpolate_and_plot(val_names,  cpt, geo, coords, tree,
                                              em_depth_cols, em_res_cols, num_layers,
                                              dirs['interp'], is_train=False)

    X_train, y_train, _, prof_train, _ = build_feature_matrix(
        train_df, em_res_cols, em_depth_cols, num_layers, is_validation=False
    )
    X_test,  y_test,  depth_test, prof_test, layer_test = build_feature_matrix(
        val_df,   em_res_cols, em_depth_cols, num_layers, is_validation=True
    )

    # — 5) Random Forest training & plotting —
    rf_model, y_pred_rf, metrics_rf = train_and_evaluate_rf(
        X_train, y_train, prof_train,
        X_test,  y_test
    )
    plot_predictions(y_test, y_pred_rf, depth_test, prof_test, layer_test, dirs['pred'])
    plot_profile_curves(interp_val, y_pred_rf, prof_test, layer_test, dirs['profiles'], val_names)

    set_random_seeds(42)
    device = get_device()

    # ---- Preprocess once ----
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_tr_s).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1,1).to(device)
    X_test_t  = torch.FloatTensor(X_te_s).to(device)
    y_test_t  = torch.FloatTensor(y_test).reshape(-1,1).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t,  y_test_t)

    # ---- Split for validation ----
    idxs = np.arange(len(train_ds))
    tr_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42)
    train_sub = Subset(train_ds, tr_idx)
    val_sub   = Subset(train_ds, val_idx)

    # ---- Hyperparameter search ----
    param_grid = {
        'hidden_size1': [32, 64, 128],
        'hidden_size2': [16, 32, 64],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [32, 64, 128]
    }
    best_params = grid_search_random(
        param_grid,
        train_sub,
        val_sub,
        device,
        input_size=X_train.shape[1],
        n_iter=20  # only 20 random trials instead of 243
    )

    # ---- Final training ----
    final_model = train_final_model(
        best_params,
        train_sub,
        val_sub,
        device,
        input_size=X_train.shape[1],
        save_path='best_model.pth'
    )

    # ---- Evaluation ----
    y_true, y_pred, rmse = evaluate_model(final_model, test_ds, device)

    # ---- Optional: plotting, saving results etc. ----

if __name__ == '__main__':
    main()

