#!/usr/bin/env python3
"""
Master Execution Pipeline: Runs Data Curation, Featurization, PyTorch Training, and Visualization.
Usage: python run_pipeline.py
"""

import subprocess
import sys

def main():
    print("================================================================")
    print("   AI FOR DRUG DISCOVERY: KINASE BIOACTIVITY PREDICTION (DL)   ")
    print("================================================================")
    
    steps = [
        ("01_fetch_data.py", "1. Fetching & Preprocessing Kinase Bioactivity Data"),
        ("02_train_model.py", "2. Featurizing with RDKit & Training PyTorch Deep Learning Model"),
        ("03_visualize.py", "3. Generating Publication Plots & Interactive Dashboard"),
        ("04_kinase_selectivity.py", "4. Computing Consortium Prioritization & Selectivity Scores"),
    ]
    
    for script, desc in steps:
        print(f"\n>>> Running Step: {desc}...")
        res = subprocess.run([sys.executable, script])
        if res.returncode != 0:
            print(f"[!] Step {script} failed. Please check error logs.")
            sys.exit(res.returncode)
            
    print("\n[✓] Pipeline execution finished successfully!")
    print("    Check the 'results/' folder for:")
    print("      - best_kinase_model.pt (PyTorch weights)")
    print("      - test_predictions.csv (Test predictions and residuals)")
    print("      - parity_plot.png (Publication figure)")
    print("      - training_curves.png (Loss curves)")
    print("      - interactive_prediction_dashboard.html (Interactive Plotly dashboard)")

if __name__ == "__main__":
    main()
