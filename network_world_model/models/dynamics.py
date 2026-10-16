"""
World Model Transition Dynamics Module.
Models the temporal evolution of network states in latent space:
z_{t+1:t+K} = Dynamics(z_{<=t})
"""
import torch
import torch.nn as nn
from typing import Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LatentWorldDynamics(nn.Module):
    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # History context encoder (GRU)
        self.history_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        # Autoregressive transition cell for future rollouts
        self.transition_cell = nn.GRUCell(
            input_size=latent_dim,
            hidden_size=hidden_dim
        )

        # Transition prior: maps recurrent state h_t to forecasted latent state z_{t+1}
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, z_hist: torch.Tensor, forecast_horizon: int = config.FORECAST_HORIZON) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_hist: (batch_size, window_size, latent_dim) - history of encoded latent states
            forecast_horizon: K - number of steps into the future to imagine
        Returns:
            z_future_forecast: (batch_size, forecast_horizon, latent_dim)
            h_future: (batch_size, forecast_horizon, hidden_dim)
        """
        batch_size = z_hist.shape[0]

        # Process historical context
        gru_out, h_last = self.history_gru(z_hist)
        # h_last shape: (num_layers, batch_size, hidden_dim) -> use top layer
        h_current = h_last[-1] # (batch_size, hidden_dim)
        z_current = z_hist[:, -1, :] # Last observed state

        forecasted_latents = []
        forecasted_hiddens = []

        # Autoregressive latent rollout into the future
        for k in range(forecast_horizon):
            # Predict next latent state from current recurrent state
            z_next = self.transition_head(h_current)
            # Update recurrent state with predicted state
            h_next = self.transition_cell(z_next, h_current)

            forecasted_latents.append(z_next.unsqueeze(1))
            forecasted_hiddens.append(h_next.unsqueeze(1))

            h_current = h_next
            z_current = z_next

        z_future_forecast = torch.cat(forecasted_latents, dim=1) # (batch, K, latent_dim)
        h_future = torch.cat(forecasted_hiddens, dim=1)          # (batch, K, hidden_dim)

        return z_future_forecast, h_future
