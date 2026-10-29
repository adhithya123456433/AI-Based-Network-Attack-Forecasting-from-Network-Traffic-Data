"""
data_pipeline.py
----------------
Handles temporal network traffic dataset creation, chronological splitting,
normalization without look-ahead bias, and sliding-window PyTorch Dataset creation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler


# 5-Stage Cyber Kill-Chain mapping
STAGE_NAMES = [
    "Normal",
    "Reconnaissance",
    "Delivery",
    "Exploitation",
    "Exfiltration"
]

FEATURE_NAMES = [
    "packet_rate",          # Packets per second
    "byte_volume",          # Bytes per second
    "syn_flag_ratio",       # SYN flags / total packets (spikes during scans/syn floods)
    "ack_flag_ratio",       # ACK flags / total packets
    "rst_flag_ratio",       # Reset flags (spikes during port scanning / teardown)
    "flow_duration",        # Average flow duration in ms
    "dst_port_entropy",     # Entropy of destination ports (high = port scan)
    "src_ip_entropy",       # Entropy of source IPs (high = distributed traffic/DDoS)
    "failed_logins_rate",   # Authentication failures (delivery/brute force)
    "packet_size_var"       # Variance in packet sizes (high in exfiltration/payload delivery)
]


def generate_synthetic_attack_telemetry(total_timesteps: int = 4000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic sequential multi-stage network attack scenarios over time.
    Simulates transitions: Normal -> Reconnaissance -> Delivery -> Exploitation -> Exfiltration.
    """
    np.random.seed(random_seed)
    
    # Baseline normal traffic distributions
    time_index = np.arange(total_timesteps)
    
    # Initialize base metrics with diurnal periodic waves + Gaussian noise
    diurnal = 0.3 * np.sin(2 * np.pi * time_index / 500)
    
    packet_rate = 100 + 20 * diurnal + np.random.normal(0, 8, total_timesteps)
    byte_volume = 40000 + 8000 * diurnal + np.random.normal(0, 3000, total_timesteps)
    syn_flag_ratio = 0.05 + np.random.normal(0, 0.01, total_timesteps)
    ack_flag_ratio = 0.85 + np.random.normal(0, 0.02, total_timesteps)
    rst_flag_ratio = 0.02 + np.random.normal(0, 0.005, total_timesteps)
    flow_duration = 250 + np.random.normal(0, 30, total_timesteps)
    dst_port_entropy = 0.2 + np.random.normal(0, 0.04, total_timesteps)
    src_ip_entropy = 0.3 + np.random.normal(0, 0.05, total_timesteps)
    failed_logins_rate = 0.01 + np.random.exponential(0.005, total_timesteps)
    packet_size_var = 150 + np.random.normal(0, 20, total_timesteps)
    
    stages = np.zeros(total_timesteps, dtype=int)  # 0: Normal
    
    # Inject sequential attack campaigns dynamically based on total_timesteps
    if total_timesteps >= 4000:
        campaign_starts = [500, 1600, 2500, 3350, 4000]
        campaign_durations = [140, 160, 140, 160, 180]
    else:
        # Scale for shorter sequences
        campaign_starts = [int(total_timesteps * 0.25)]
        campaign_durations = [min(120, int(total_timesteps * 0.4))]
    
    for start, dur in zip(campaign_starts, campaign_durations):
        # Stage 1: Reconnaissance (Port Scanning)
        p1 = int(dur * 0.25)
        idx1 = slice(start, start + p1)
        stages[idx1] = 1
        syn_flag_ratio[idx1] += np.random.uniform(0.35, 0.55, p1)
        dst_port_entropy[idx1] += np.random.uniform(0.55, 0.75, p1)
        rst_flag_ratio[idx1] += np.random.uniform(0.15, 0.30, p1)
        packet_rate[idx1] += np.random.uniform(50, 100, p1)
        
        # Stage 2: Delivery & Weaponization (Brute-force / probing)
        p2 = int(dur * 0.25)
        idx2 = slice(start + p1, start + p1 + p2)
        stages[idx2] = 2
        failed_logins_rate[idx2] += np.random.uniform(0.40, 0.80, p2)
        flow_duration[idx2] *= 0.5
        packet_size_var[idx2] += np.random.uniform(80, 150, p2)
        
        # Stage 3: Exploitation & Lateral Movement
        p3 = int(dur * 0.25)
        idx3 = slice(start + p1 + p2, start + p1 + p2 + p3)
        stages[idx3] = 3
        flow_duration[idx3] += np.random.uniform(200, 400, p3)
        src_ip_entropy[idx3] += np.random.uniform(0.30, 0.50, p3)
        ack_flag_ratio[idx3] -= 0.25
        
        # Stage 4: Exfiltration & Impact (Data theft / DoS burst)
        p4 = dur - (p1 + p2 + p3)
        idx4 = slice(start + p1 + p2 + p3, start + dur)
        stages[idx4] = 4
        byte_volume[idx4] += np.random.uniform(80000, 160000, p4)
        packet_rate[idx4] += np.random.uniform(250, 500, p4)
        packet_size_var[idx4] += np.random.uniform(300, 600, p4)
        
    df = pd.DataFrame({
        "packet_rate": np.clip(packet_rate, 5, None),
        "byte_volume": np.clip(byte_volume, 100, None),
        "syn_flag_ratio": np.clip(syn_flag_ratio, 0.0, 1.0),
        "ack_flag_ratio": np.clip(ack_flag_ratio, 0.0, 1.0),
        "rst_flag_ratio": np.clip(rst_flag_ratio, 0.0, 1.0),
        "flow_duration": np.clip(flow_duration, 10, None),
        "dst_port_entropy": np.clip(dst_port_entropy, 0.0, 1.0),
        "src_ip_entropy": np.clip(src_ip_entropy, 0.0, 1.0),
        "failed_logins_rate": np.clip(failed_logins_rate, 0.0, 1.0),
        "packet_size_var": np.clip(packet_size_var, 10, None),
        "stage": stages
    })
    return df


class TemporalSequenceDataset(Dataset):
    """
    Sliding window dataset for sequence-to-sequence future forecasting
    and multi-stage attack classification.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, window_size: int = 30, forecast_horizon: int = 5):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        
        # Number of valid sequences
        self.num_samples = len(features) - window_size - forecast_horizon + 1
        assert self.num_samples > 0, "Dataset too short for specified window size and horizon."
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Historical observation window: X[t - W : t]
        x_past = self.features[idx : idx + self.window_size]
        
        # Future ground truth traffic: Y[t : t + k]
        y_future = self.features[idx + self.window_size : idx + self.window_size + self.forecast_horizon]
        
        # Attack stage label at forecast horizon (target future state)
        # Using the maximum severity stage in the future forecast window
        future_labels = self.labels[idx + self.window_size : idx + self.window_size + self.forecast_horizon]
        y_stage = torch.max(future_labels).item()
        
        return x_past, y_future, torch.tensor(y_stage, dtype=torch.long)


def prepare_temporal_dataloaders(df: pd.DataFrame, window_size: int = 30, forecast_horizon: int = 5, batch_size: int = 32):
    """
    Splits data chronologically (70% train, 15% validation, 15% test).
    Fits RobustScaler strictly on the training segment to avoid look-ahead data leakage.
    Returns PyTorch DataLoaders, the fitted scaler, and class distribution weights.
    """
    feature_cols = [c for c in df.columns if c != "stage"]
    raw_features = df[feature_cols].values
    raw_labels = df["stage"].values
    
    total_len = len(df)
    train_end = int(total_len * 0.70)
    val_end = int(total_len * 0.85)
    
    # 1. Fit scaler ONLY on training data
    scaler = RobustScaler()
    scaled_train = scaler.fit_transform(raw_features[:train_end])
    scaled_val = scaler.transform(raw_features[train_end:val_end])
    scaled_test = scaler.transform(raw_features[val_end:])
    
    scaled_features = np.vstack([scaled_train, scaled_val, scaled_test])
    
    # 2. Slice into chronological datasets
    train_feats = scaled_features[:train_end]
    train_labels = raw_labels[:train_end]
    
    val_feats = scaled_features[train_end:val_end]
    val_labels = raw_labels[train_end:val_end]
    
    test_feats = scaled_features[val_end:]
    test_labels = raw_labels[val_end:]
    
    train_dataset = TemporalSequenceDataset(train_feats, train_labels, window_size, forecast_horizon)
    val_dataset = TemporalSequenceDataset(val_feats, val_labels, window_size, forecast_horizon)
    test_dataset = TemporalSequenceDataset(test_feats, test_labels, window_size, forecast_horizon)
    
    # Compute inverse class frequencies for handling class imbalance
    unique, counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)
    num_classes = len(STAGE_NAMES)
    class_weights = np.ones(num_classes, dtype=np.float32)
    for u, c in zip(unique, counts):
        class_weights[u] = total_samples / (num_classes * c)
        
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler, class_weights
