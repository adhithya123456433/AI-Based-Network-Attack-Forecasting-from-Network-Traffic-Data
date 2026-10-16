"""
Standard Reactive Baseline Model (Traditional LSTM-IDS).
Detects attack stages purely reactively at the current timestep t (x_t -> y_t)
without latent transition dynamics or future forecasting capabilities.
Used for academic benchmarking against the proposed World Model.
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ReactiveLSTMClassifier(nn.Module):
    def __init__(
        self,
        in_features: int = config.NUM_FEATURES,
        hidden_dim: int = 64,
        num_stages: int = config.NUM_STAGES,
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_stages)
        )
        self.alarm_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_hist: torch.Tensor):
        """
        Args:
            x_hist: (batch_size, window_size, in_features)
        Returns:
            stage_logits: (batch_size, num_stages) for current time t
            alarm_prob: (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x_hist)
        last_step = lstm_out[:, -1, :] # representation at time t
        stage_logits = self.classifier(last_step)
        alarm_prob = self.alarm_head(last_step)
        return stage_logits, alarm_prob
