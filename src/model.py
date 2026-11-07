"""
Deep Learning Architecture: PyTorch Deep Neural Network for Kinase Bioactivity Prediction
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class KinaseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class KinaseAffinityNet(nn.Module):
    """
    Multi-layer Perceptron (MLP) Regressor with Batch Normalization and Dropout Regularization.
    Maps molecular representations (Morgan FP + physicochemical descriptors) to pIC50 affinity.
    """
    def __init__(self, in_features: int = 1030, hidden_dims=[512, 256, 64], dropout_rate: float = 0.25):
        super().__init__()
        
        layers = []
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        # Final regression head
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
