# Kinase Bioactivity Prediction with PyTorch & RDKit

A reproducible deep-learning benchmark predicting small-molecule binding affinity ($p\mathrm{IC}_{50}$) against **Human Epidermal Growth Factor Receptor (EGFR) Kinase** ([ChEMBL203](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL203/)).

This repository demonstrates an end-to-end computational drug discovery workflow:
1. **Bioactivity Data Curation:** Programmatic retrieval and standardization of kinase inhibitor data from the ChEMBL database.
2. **Cheminformatics Featurization:** High-dimensional 1024-bit Morgan Fingerprints (ECFP4 equivalent) paired with 2D physicochemical descriptors using **RDKit**.
3. **Deep Learning Model (PyTorch):** Deep Multi-Layer Perceptron (MLP) with Batch Normalization, Dropout regularization, and learning rate scheduling.
4. **Structure-Based Docking Bridge:** Integration of 3D receptor-ligand interaction parameters (AutoDock binding energy, hinge-region H-bonds, and pocket occlusion inspired by CASTp analysis) to explore hybrid proteochemometric representations.
5. **Consortium Prioritization & Selectivity:** Automated scoring module computing off-target kinome selectivity margins and prioritization rankings for lead optimization.
6. **Interactive Prioritization Dashboard:** Interactive Plotly explorer for compound ranking and residual inspection.

---

## Architecture Overview

```text
SMILES String
      │
      ▼
┌───────────────────────────────────────────────┐
│              RDKit Featurization              │
│  - Morgan Fingerprint (1024-bit, radius=2)    │
│  - Physicochemical: [MW, LogP, HBD, HBA, TPSA]│
└───────────────────────┬───────────────────────┘
                        │ Feature Vector (1030-D)
                        ▼
┌───────────────────────────────────────────────┐
│        KinaseAffinityNet (PyTorch DNN)        │
│  Linear(1030 -> 512) + BatchNorm + Dropout    │
│  Linear(512  -> 256) + BatchNorm + Dropout    │
│  Linear(256  -> 64)  + BatchNorm + ReLU       │
│  Linear(64   -> 1)   [Regression Head]        │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
          Predicted Kinase Bioactivity (pIC50)
```

---

## Benchmark Results (Test Set)

| Model Architecture | RMSE | MAE | $R^2$ Score | Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch Deep Neural Network (MLP)** | **0.687** | **0.514** | **0.766** | **0.876** |
| Random Forest Regressor (Baseline) | 0.725 | 0.565 | 0.740 | 0.860 |

*Evaluation on 20% held-out test split of diverse kinase inhibitors.*

---

## Repository Structure

```text
kinase-affinity-dl/
├── data/
│   └── egfr_bioactivity_clean.csv       # Curated kinase bioactivity dataset
├── src/
│   ├── features.py                      # RDKit Morgan fingerprints & descriptors
│   └── model.py                         # PyTorch KinaseAffinityNet architecture
├── results/
│   ├── best_kinase_model.pt             # Trained PyTorch model checkpoint
│   ├── test_predictions.csv             # Test set predictions and residuals
│   ├── parity_plot.png                  # Publication-ready parity plot
│   ├── training_curves.png              # Loss convergence curves
│   └── interactive_prediction_dashboard.html  # Interactive Plotly dashboard
├── 01_fetch_data.py                     # ChEMBL query and data sanitation
├── 02_train_model.py                    # PyTorch training & baseline benchmark
├── 03_visualize.py                      # Matplotlib & Plotly visualization
├── run_pipeline.py                      # Single-command end-to-end execution
└── requirements.txt                     # Python dependencies
```

---

## Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/RidaHayat/kinase-affinity-dl.git
cd kinase-affinity-dl
```

### 2. Create and activate a virtual environment
```bash
# Using conda (recommended for RDKit):
conda create -n kinase-dl python=3.10 -y
conda activate kinase-dl
pip install -r requirements.txt

# Or using standard venv:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the complete pipeline
```bash
python run_pipeline.py
```

All models will train, metrics will be calculated, and publication figures + interactive HTML dashboards will be generated in `results/`.

---

## Interactive Dashboard

Open `results/interactive_prediction_dashboard.html` in any browser to inspect candidate molecules, examine predicted vs. experimental $p\mathrm{IC}_{50}$, and prioritize high-affinity binders for downstream docking and structure-based validation.

---

## Author
**Rida Hayat**  
MSc in Medical Bioinformatics, University of Verona  
[GitHub Profile](https://github.com/RidaHayat) • [LinkedIn](https://www.linkedin.com/in/rida-hayat)
