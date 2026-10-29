"""
models.py
---------
Deep learning model architectures:
1. NetworkAttackWorldModel: Latent Dynamics World Model with 1D-CNN Encoder,
   Residual GRU Transition Core, and Dual Prediction Heads (Forecasting + Attack Stages).
2. StaticMLPBaseline: Non-temporal baseline.
3. VanillaLSTM: Standard recurrent sequence-to-sequence baseline without latent dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """
    Observation Encoder:
    Compresses raw sliding window of network traffic X[t-W : t]
    into a compact latent bottleneck representation z[t].
    """
    def __init__(self, in_features: int, latent_dim: int = 64, dropout: float = 0.25):
        super(TemporalEncoder, self).__init__()
        # in_features -> 64 -> 128 -> latent_dim
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=3, padding=1)
        self.ln1 = nn.GroupNorm(1, 64)  # LayerNorm equivalent for 1D conv
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.ln2 = nn.GroupNorm(1, 128)
        
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_latent = nn.Linear(128, latent_dim)
        self.ln_latent = nn.LayerNorm(latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, window_size, in_features]
        # Transpose for Conv1d: [batch, in_features, window_size]
        h = x.transpose(1, 2)
        
        h = F.leaky_relu(self.ln1(self.conv1(h)), negative_slope=0.1)
        h = self.dropout(h)
        h = F.leaky_relu(self.ln2(self.conv2(h)), negative_slope=0.1)
        h = self.dropout(h)
        
        h = self.pool(h).squeeze(-1)  # [batch, 128]
        z = self.ln_latent(self.fc_latent(h))  # [batch, latent_dim]
        return z


class LatentTransitionWorldModel(nn.Module):
    """
    World Model Dynamics Engine:
    Simulates the forward temporal evolution of the network inside the latent space:
    z[t+1] = f_theta(z[t])
    Uses a Residual GRU block to preserve gradient flow and prevent underfitting.
    """
    def __init__(self, latent_dim: int = 64, horizon: int = 5, dropout: float = 0.20):
        super(LatentTransitionWorldModel, self).__init__()
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.transition_cell = nn.GRUCell(latent_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z_init: torch.Tensor):
        # z_init: [batch, latent_dim]
        batch_size = z_init.size(0)
        latent_trajectory = []
        
        z_current = z_init
        for _ in range(self.horizon):
            # Recurrent latent transition with residual connection
            z_next = self.transition_cell(z_current, z_current)
            z_next = self.layer_norm(z_current + z_next)  # Residual skip
            z_next = self.dropout(z_next)
            latent_trajectory.append(z_next.unsqueeze(1))
            z_current = z_next
            
        # [batch, horizon, latent_dim]
        latent_seq = torch.cat(latent_trajectory, dim=1)
        return latent_seq, z_current


class NetworkAttackWorldModel(nn.Module):
    """
    End-to-End Latent Dynamics World Model:
    1. Observation Encoder -> z_t
    2. Latent Dynamics Transition Engine -> z_{t+1:t+k}
    3. Future Telemetry Forecaster Head -> X_hat_{t+1:t+k}
    4. Multi-Stage Attack Classifier Head -> Stage Logits (Kill-Chain phase)
    """
    def __init__(self, in_features: int = 10, latent_dim: int = 64, horizon: int = 5,
                 num_classes: int = 5, dropout: float = 0.25):
        super(NetworkAttackWorldModel, self).__init__()
        self.in_features = in_features
        self.horizon = horizon
        self.num_classes = num_classes
        
        # 1. Observation Encoder
        self.encoder = TemporalEncoder(in_features=in_features, latent_dim=latent_dim, dropout=dropout)
        
        # 2. World Model Latent Dynamics Core
        self.dynamics = LatentTransitionWorldModel(latent_dim=latent_dim, horizon=horizon, dropout=dropout)
        
        # 3. Head A: Continuous Future Telemetry Forecaster
        self.telemetry_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, in_features)
        )
        
        # 4. Head B: Attack Stage Classifier
        self.stage_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x_past: torch.Tensor):
        # x_past: [batch, window_size, in_features]
        z_t = self.encoder(x_past)  # [batch, latent_dim]
        
        # Simulate forward trajectory in latent space
        latent_trajectory, z_final = self.dynamics(z_t)  # [batch, horizon, latent_dim]
        
        # Predict continuous future metrics for all horizon steps
        pred_future = self.telemetry_decoder(latent_trajectory)  # [batch, horizon, in_features]
        
        # Predict attack stage at final horizon state
        stage_logits = self.stage_classifier(z_final)  # [batch, num_classes]
        
        return pred_future, stage_logits, z_t


# ==========================================
# BASELINE MODELS (FOR EMPIRICAL COMPARISON)
# ==========================================

class StaticMLPBaseline(nn.Module):
    """
    Non-temporal baseline: evaluates only the most recent network snapshot,
    ignoring temporal dynamics. Demonstrates why sequential modeling is vital.
    """
    def __init__(self, in_features: int = 10, horizon: int = 5, num_classes: int = 5, dropout: float = 0.25):
        super(StaticMLPBaseline, self).__init__()
        self.in_features = in_features
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_future = nn.Linear(64, horizon * in_features)
        self.fc_stage = nn.Linear(64, num_classes)
        
    def forward(self, x_past: torch.Tensor):
        # Only take the latest time step snapshot: X[:, -1, :]
        x_latest = x_past[:, -1, :]
        h = self.net(x_latest)
        pred_future = self.fc_future(h).view(-1, self.horizon, self.in_features)
        stage_logits = self.fc_stage(h)
        return pred_future, stage_logits, h


class VanillaSeq2SeqLSTM(nn.Module):
    """
    Standard sequence-to-sequence LSTM without a latent world model.
    Directly encodes the sequence and decodes predictions.
    """
    def __init__(self, in_features: int = 10, hidden_dim: int = 64, horizon: int = 5,
                 num_classes: int = 5, dropout: float = 0.25):
        super(VanillaSeq2SeqLSTM, self).__init__()
        self.in_features = in_features
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.fc_future = nn.Linear(hidden_dim, in_features)
        self.fc_stage = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x_past: torch.Tensor):
        out, (h_n, _) = self.lstm(x_past)
        last_hidden = h_n[-1]  # [batch, hidden_dim]
        
        # Forecast future by repeating last hidden state
        rep_hidden = last_hidden.unsqueeze(1).repeat(1, self.horizon, 1)
        pred_future = self.fc_future(rep_hidden)
        stage_logits = self.fc_stage(last_hidden)
        return pred_future, stage_logits, last_hidden
