#!/usr/bin/env python3
"""
Step 3: Generate Publication Plots and Interactive Plotly Dashboard
Visualizes Deep Learning predictions, loss convergence, and target prioritization.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_visualizations(results_dir="results"):
    print("[*] Generating publication figures and interactive dashboard...")
    preds_file = os.path.join(results_dir, "test_predictions.csv")
    metrics_file = os.path.join(results_dir, "metrics.json")
    
    if not os.path.exists(preds_file):
        print(f"[!] Predictions file not found at {preds_file}. Please run 02_train_model.py first.")
        return

    df = pd.read_csv(preds_file)
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # 1. Parity Plot: Experimental vs Predicted pIC50
    plt.figure(figsize=(7, 6))
    plt.scatter(df["pic50"], df["predicted_pic50_pytorch"], alpha=0.75, c="#1f77b4", edgecolors="k", s=50, label="Test Compounds")
    
    min_val = min(df["pic50"].min(), df["predicted_pic50_pytorch"].min()) - 0.5
    max_val = max(df["pic50"].max(), df["predicted_pic50_pytorch"].max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal Parity (y = x)")
    
    plt.xlabel("Experimental $p\\mathrm{IC}_{50}$", fontsize=12, fontweight="bold")
    plt.ylabel("PyTorch Predicted $p\\mathrm{IC}_{50}$", fontsize=12, fontweight="bold")
    plt.title(f"Kinase Bioactivity Prediction (EGFR)\n$R^2$ = {metrics['pytorch']['r2']:.3f} | RMSE = {metrics['pytorch']['rmse']:.3f} | Pearson $r$ = {metrics['pytorch']['pearson']:.3f}", fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    parity_path = os.path.join(results_dir, "parity_plot.png")
    plt.savefig(parity_path, dpi=300)
    plt.close()
    print(f"[+] Saved parity plot to: {parity_path}")

    # 2. Training Convergence Plot
    history = metrics.get("history", {})
    if history.get("train_loss"):
        plt.figure(figsize=(7, 4.5))
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs, history["train_loss"], label="Train Loss (MSE)", color="#2ca02c", linewidth=2)
        plt.plot(epochs, history["val_loss"], label="Validation Loss (MSE)", color="#d62728", linewidth=2, linestyle="--")
        plt.xlabel("Epoch", fontsize=11, fontweight="bold")
        plt.ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight="bold")
        plt.title("PyTorch Training & Validation Convergence", fontsize=13)
        plt.legend(frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        curves_path = os.path.join(results_dir, "training_curves.png")
        plt.savefig(curves_path, dpi=300)
        plt.close()
        print(f"[+] Saved loss convergence curves to: {curves_path}")

    # 3. Interactive Plotly Dashboard
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        df["Affinity Group"] = np.where(df["pic50"] >= 7.0, "Potent Binder (pIC50 >= 7)", "Moderate/Weak Binder")
        
        fig = px.scatter(
            df,
            x="pic50",
            y="predicted_pic50_pytorch",
            color="Affinity Group",
            hover_data=["molecule_id", "smiles", "pic50", "predicted_pic50_pytorch", "residual_pytorch"],
            title=f"Kinase Prioritization Explorer (EGFR) - PyTorch DNN Benchmark<br><sup>RMSE: {metrics['pytorch']['rmse']:.3f} | R²: {metrics['pytorch']['r2']:.3f}</sup>",
            labels={
                "pic50": "Experimental pIC50",
                "predicted_pic50_pytorch": "PyTorch Predicted pIC50"
            },
            template="plotly_white"
        )
        # Add diagonal parity line
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Ideal Parity"
        ))
        
        dashboard_path = os.path.join(results_dir, "interactive_prediction_dashboard.html")
        fig.write_html(dashboard_path)
        print(f"[+] Saved interactive Plotly dashboard to: {dashboard_path}")
    except ImportError:
        print("[!] Plotly not installed; skipping HTML dashboard generation.")

if __name__ == "__main__":
    generate_visualizations()
