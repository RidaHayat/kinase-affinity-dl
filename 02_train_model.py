#!/usr/bin/env python3
"""
Step 2: Train Deep Learning (PyTorch) Model & Benchmark Against Random Forest
Evaluates prediction of kinase inhibitor affinity (pIC50).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from src.features import compute_molecular_features
from src.model import KinaseAffinityNet

def run_training(
    data_path="data/egfr_bioactivity_clean.csv",
    epochs=60,
    batch_size=32,
    lr=1e-3,
    output_dir="results"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"[!] Dataset not found at {data_path}. Running 01_fetch_data.py first...")
        from importlib.machinery import SourceFileLoader
        fetcher = SourceFileLoader("fetch", "01_fetch_data.py").load_module()
        df_raw = fetcher.fetch_chembl_egfr()
        fetcher.preprocess_and_save(df_raw, data_path)

    df = pd.read_csv(data_path)
    print(f"[*] Processing {len(df)} compounds with RDKit molecular featurization...")

    # 2. Featurize
    X_list, y_list, valid_indices = [], [], []
    for idx, row in df.iterrows():
        feat = compute_molecular_features(row["smiles"])
        if feat is not None:
            X_list.append(feat)
            y_list.append(row["pic50"])
            valid_indices.append(idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    df_valid = df.iloc[valid_indices].copy().reset_index(drop=True)
    print(f"[+] Featurization complete. Feature matrix shape: {X.shape}")

    # 3. Train / Test Split (80/20)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize the continuous descriptors (last 6 columns)
    scaler = StandardScaler()
    X_train[:, -6:] = scaler.fit_transform(X_train[:, -6:])
    X_test[:, -6:] = scaler.transform(X_test[:, -6:])

    # 4. Baseline Model: Random Forest Regressor
    print("\n--- Benchmarking Baseline: Random Forest Regressor ---")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
    rf_mae = float(mean_absolute_error(y_test, rf_preds))
    rf_r2 = float(r2_score(y_test, rf_preds))
    rf_pearson, _ = pearsonr(y_test, rf_preds)
    print(f"Random Forest  ->  RMSE: {rf_rmse:.3f} | MAE: {rf_mae:.3f} | R²: {rf_r2:.3f} | Pearson r: {rf_pearson:.3f}")

    # 5. Deep Learning: PyTorch Architecture
    print("\n--- Training Deep Neural Network (PyTorch) ---")
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))

    # Prevent batch norm crash if final batch has size 1
    drop_last = (len(train_dataset) > batch_size and len(train_dataset) % batch_size == 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = KinaseAffinityNet(in_features=X.shape[1], hidden_dims=[512, 256, 64], dropout_rate=0.25).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = model(x_b)
                loss = criterion(preds, y_b)
                val_loss += loss.item() * len(y_b)
        val_loss /= len(test_loader.dataset)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_kinase_model.pt"))

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    # 6. Final Evaluation on Test Set
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_kinase_model.pt")))
    model.eval()
    with torch.no_grad():
        nn_preds = model(torch.tensor(X_test).to(device)).cpu().numpy().flatten()

    nn_rmse = float(np.sqrt(mean_squared_error(y_test, nn_preds)))
    nn_mae = float(mean_absolute_error(y_test, nn_preds))
    nn_r2 = float(r2_score(y_test, nn_preds))
    nn_pearson, _ = pearsonr(y_test, nn_preds)

    print("\n================ FINAL BENCHMARK SUMMARY ================")
    print(f"PyTorch DNN    ->  RMSE: {nn_rmse:.3f} | MAE: {nn_mae:.3f} | R²: {nn_r2:.3f} | Pearson r: {nn_pearson:.3f}")
    print(f"Random Forest  ->  RMSE: {rf_rmse:.3f} | MAE: {rf_mae:.3f} | R²: {rf_r2:.3f} | Pearson r: {rf_pearson:.3f}")
    print("=========================================================\n")

    # 7. Save Predictions & Metrics
    results_df = df_valid.iloc[test_idx].copy().reset_index(drop=True)
    results_df["predicted_pic50_pytorch"] = np.round(nn_preds, 3)
    results_df["predicted_pic50_rf"] = np.round(rf_preds, 3)
    results_df["residual_pytorch"] = np.round(results_df["pic50"] - results_df["predicted_pic50_pytorch"], 3)
    results_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    metrics = {
        "pytorch": {"rmse": nn_rmse, "mae": nn_mae, "r2": nn_r2, "pearson": float(nn_pearson)},
        "random_forest": {"rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2, "pearson": float(rf_pearson)},
        "epochs": epochs,
        "n_samples": len(X),
        "history": history
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[+] All outputs successfully saved to '{output_dir}/'")

if __name__ == "__main__":
    run_training()
