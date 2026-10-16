"""
State Encoder Module.
Encodes multi-dimensional network flow metrics into a compact latent representation z_t.
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FlowStateEncoder(nn.Module):
    def __init__(self, in_features: int = config.NUM_FEATURES, latent_dim: int = config.LATENT_DIM, dropout: float = config.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, ..., in_features)
        Returns:
            z: (batch, ..., latent_dim)
        """
        return self.net(x)
