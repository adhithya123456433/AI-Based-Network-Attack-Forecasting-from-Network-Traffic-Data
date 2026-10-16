"""
State Decoder / Reconstructor Module.
Maps latent vectors z_t back into physical network flow telemetry features x_t.
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FlowStateDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        out_features: int = config.NUM_FEATURES,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, out_features),
            nn.Sigmoid()  # Flow metrics are normalized in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, ..., latent_dim)
        Returns:
            x_rec: (batch, ..., out_features)
        """
        return self.net(z)
