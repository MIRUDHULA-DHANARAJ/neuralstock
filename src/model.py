"""
Model Definitions Module for NeuralStock.
Implements the Stacked LSTM Network and Feedforward MLP Baseline.
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Stacked LSTM network for multivariate time-series demand forecasting.
    Expects input shape: (Batch, Seq_Length, Features)
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            input_size (int): Total number of feature columns.
            hidden_size (int): Internal recurrent vector dimensions.
            num_layers (int): Depth level for stacking LSTM units.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes 3D sequential frames into a 1D forecast array."""
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  
        return self.fc(last_step)


class MLPModel(nn.Module):
    """
    Feedforward Multi-Layer Perceptron Baseline.
    Flattens sequence windows entirely to capture static historical relations.
    """
    def __init__(self, input_size: int, seq_length: int = 8):
        
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size * seq_length, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        return self.network(x)
