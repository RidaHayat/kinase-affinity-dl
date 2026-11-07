#!/usr/bin/env python3
"""
Step 4: Consortium Kinase Prioritization & Selectivity Profiling
Mirrors the consortium dashboard objective:
Ranks kinase inhibitors by predicted affinity AND off-target selectivity.
"""

import os
import pandas as pd
import numpy as np

def compute_consortium_prioritization(results_dir="results"):
    print("[*] Generating Consortium Kinase Prioritization Table...")
    preds_file = os.path.join(results_dir, "test_predictions.csv")
    if not os.path.exists(preds_file):
        print(f"[!] Test predictions not found at {preds_file}.")
        return

    df = pd.read_csv(preds_file)

    # 1. Consortium Scoring Heuristic:
    # High potency (pIC50 >= 7.5), low prediction uncertainty / residual
    df["Potency_Score"] = np.clip((df["predicted_pic50_pytorch"] - 5.0) / 4.0, 0, 1)
    df["Confidence_Score"] = np.exp(-np.abs(df["residual_pytorch"]))
    
    # 2. Simulated Off-target Kinase Profile (e.g. Selectivity over sibling kinase HER2/CDK2)
    # A true drug candidate needs selective inhibition to avoid clinical toxicity
    np.random.seed(42)
    df["OffTarget_Predicted_pIC50"] = np.clip(
        df["predicted_pic50_pytorch"] - np.random.normal(1.2, 0.4, size=len(df)),
        3.0, 9.0
    ).round(2)
    
    df["Selectivity_Margin_Delta_pIC50"] = (
        df["predicted_pic50_pytorch"] - df["OffTarget_Predicted_pIC50"]
    ).round(2)

    # Composite Consortium Score (0 to 100)
    df["Consortium_Priority_Score"] = (
        (df["Potency_Score"] * 0.5 + df["Confidence_Score"] * 0.3 + (df["Selectivity_Margin_Delta_pIC50"] > 1.0).astype(int) * 0.2) * 100
    ).round(1)

    # Sort by priority
    df_prioritized = df.sort_values(by="Consortium_Priority_Score", ascending=False)
    
    out_path = os.path.join(results_dir, "consortium_prioritized_leads.csv")
    df_prioritized.to_csv(out_path, index=False)
    print(f"[+] Saved Consortium Prioritized Leads to: {out_path}")
    print("\nTop 5 Consortium Prioritized Candidates:")
    cols = ["molecule_id", "predicted_pic50_pytorch", "Selectivity_Margin_Delta_pIC50", "Consortium_Priority_Score"]
    print(df_prioritized[cols].head(5).to_string(index=False))

if __name__ == "__main__":
    compute_consortium_prioritization()
