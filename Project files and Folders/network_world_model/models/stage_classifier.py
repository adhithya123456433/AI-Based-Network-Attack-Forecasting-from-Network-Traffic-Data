"""
Attack Stage & Early Warning Forecasting Head.
Takes future latent state vectors z_{t+k} (or combined with hidden recurrent state h_{t+k})
and classifies the predicted Cyber Kill Chain stage:
- 0: Benign Normal
- 1: Reconnaissance (Probe / Scan)
- 2: Initial Compromise (Exploit / Brute Force)
- 3: Lateral Movement
- 4: Exfiltration & Impact
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AttackStageForecaster(nn.Module):
    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_stages: int = config.NUM_STAGES,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        in_features = latent_dim + hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, num_stages)
        )

        # Binary Early Warning head (0: Normal, 1: Under Attack)
        self.early_warning_head = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, z_future: torch.Tensor, h_future: torch.Tensor):
        """
        Args:
            z_future: (batch, K, latent_dim)
            h_future: (batch, K, hidden_dim)
        Returns:
            stage_logits: (batch, K, num_stages)
            early_warning_prob: (batch, K, 1)
        """
        combined = torch.cat([z_future, h_future], dim=-1)
        stage_logits = self.classifier(combined)
        early_warning_prob = self.early_warning_head(combined)
        return stage_logits, early_warning_prob
