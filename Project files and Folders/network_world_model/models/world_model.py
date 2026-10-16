"""
Unified Deep Learning World Model for Network Attack Forecasting.
Combines:
1. FlowStateEncoder: x_t -> z_t
2. LatentWorldDynamics: z_{<=t} -> z_{t+1:t+K}
3. FlowStateDecoder: z_{t+k} -> \hat{x}_{t+k}
4. AttackStageForecaster: (z_{t+k}, h_{t+k}) -> \hat{y}_{t+k}, Early Warning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.encoder import FlowStateEncoder
from models.dynamics import LatentWorldDynamics
from models.decoder import FlowStateDecoder
from models.stage_classifier import AttackStageForecaster


class NetworkWorldModel(nn.Module):
    def __init__(
        self,
        num_features: int = config.NUM_FEATURES,
        latent_dim: int = config.LATENT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_stages: int = config.NUM_STAGES,
        forecast_horizon: int = config.FORECAST_HORIZON,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages
        self.forecast_horizon = forecast_horizon

        # World Model components
        self.encoder = FlowStateEncoder(in_features=num_features, latent_dim=latent_dim, dropout=dropout)
        self.dynamics = LatentWorldDynamics(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.decoder = FlowStateDecoder(latent_dim=latent_dim, out_features=num_features, dropout=dropout)
        self.forecaster = AttackStageForecaster(latent_dim=latent_dim, hidden_dim=hidden_dim, num_stages=num_stages, dropout=dropout)

    def forward(self, x_hist: torch.Tensor, horizon: int = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_hist: (batch_size, window_size, num_features)
            horizon: forecast horizon K (default config.FORECAST_HORIZON)
        Returns:
            Dictionary containing:
                - z_hist: encoded historical latent states
                - z_future_forecast: forecasted future latent states
                - x_future_forecast: decoded forecasted flow metrics
                - stage_logits: forecasted stage classification logits
                - early_warning_prob: proactive attack alert probability
        """
        if horizon is None:
            horizon = self.forecast_horizon

        # 1. Encode historical flow observation window
        z_hist = self.encoder(x_hist) # (batch, W, latent_dim)

        # 2. Rollout latent dynamics into the future
        z_future_forecast, h_future = self.dynamics(z_hist, forecast_horizon=horizon)

        # 3. Decode forecasted latent states into future flow metrics
        x_future_forecast = self.decoder(z_future_forecast) # (batch, K, num_features)

        # 4. Predict attack stage progression & early warning
        stage_logits, early_warning_prob = self.forecaster(z_future_forecast, h_future)

        return {
            "z_hist": z_hist,
            "z_future_forecast": z_future_forecast,
            "h_future": h_future,
            "x_future_forecast": x_future_forecast,
            "stage_logits": stage_logits,
            "early_warning_prob": early_warning_prob
        }

    def compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        alpha_recon: float = config.ALPHA_RECON,
        beta_trans: float = config.BETA_TRANS,
        gamma_stage: float = config.GAMMA_STAGE
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Multi-task loss:
        - Reconstruction loss on future flow telemetry
        - Latent transition consistency loss
        - Attack stage forecasting loss (CrossEntropy)
        - Early warning binary classification loss (BCE)
        """
        x_hist = batch["x_hist"]
        x_future = batch["x_future"]
        y_future = batch["y_future"]

        batch_size, K, D = x_future.shape

        outputs = self.forward(x_hist, horizon=K)
        x_forecast = outputs["x_future_forecast"]
        z_forecast = outputs["z_future_forecast"]
        stage_logits = outputs["stage_logits"]
        alarm_prob = outputs["early_warning_prob"].squeeze(-1)

        # 1. State Reconstruction Loss (MSE)
        loss_recon = F.mse_loss(x_forecast, x_future)

        # 2. Latent Transition Consistency Loss:
        # Compare imagined future latents with direct encodings of actual future data
        with torch.no_grad():
            z_future_true = self.encoder(x_future)
        loss_trans = F.mse_loss(z_forecast, z_future_true)

        # 3. Stage Classification Loss (CrossEntropy)
        loss_stage = F.cross_entropy(
            stage_logits.view(-1, self.num_stages),
            y_future.view(-1)
        )

        # 4. Proactive Early Warning Loss (BCE)
        alarm_target = (y_future > 0).float()
        loss_alarm = F.binary_cross_entropy(alarm_prob, alarm_target)

        total_loss = (
            alpha_recon * loss_recon +
            beta_trans * loss_trans +
            gamma_stage * (loss_stage + 0.5 * loss_alarm)
        )

        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": loss_recon.item(),
            "trans_loss": loss_trans.item(),
            "stage_loss": loss_stage.item(),
            "alarm_loss": loss_alarm.item()
        }

        return total_loss, metrics
