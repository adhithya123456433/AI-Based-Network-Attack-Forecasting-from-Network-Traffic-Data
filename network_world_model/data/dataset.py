"""
PyTorch Dataset and DataLoader builder for Network World Model.
Implements temporal sliding window slicing for historical context and future forecasting horizons.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.killchain_simulator import NetworkKillChainSimulator


class NetworkSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, window_size: int = config.WINDOW_SIZE, forecast_horizon: int = config.FORECAST_HORIZON):
        """
        Args:
            X: (num_episodes, episode_len, num_features)
            Y: (num_episodes, episode_len)
            window_size: W (length of past observations)
            forecast_horizon: K (length of future forecast horizon)
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

        self.samples_x_hist = []      # (W, D)
        self.samples_x_future = []    # (K, D)
        self.samples_y_curr = []      # Scalar current stage at time t
        self.samples_y_future = []    # (K,) future stage labels

        num_episodes, ep_len, _ = X.shape

        for ep in range(num_episodes):
            for t in range(window_size, ep_len - forecast_horizon):
                # Past observation window: [t - W, t)
                x_hist = X[ep, t - window_size:t]
                # Future states to forecast: [t, t + K)
                x_future = X[ep, t:t + forecast_horizon]
                # Current stage at time t-1
                y_curr = Y[ep, t - 1]
                # Future stages: [t, t + K)
                y_future = Y[ep, t:t + forecast_horizon]

                self.samples_x_hist.append(x_hist)
                self.samples_x_future.append(x_future)
                self.samples_y_curr.append(y_curr)
                self.samples_y_future.append(y_future)

        self.samples_x_hist = torch.tensor(np.array(self.samples_x_hist), dtype=torch.float32)
        self.samples_x_future = torch.tensor(np.array(self.samples_x_future), dtype=torch.float32)
        self.samples_y_curr = torch.tensor(np.array(self.samples_y_curr), dtype=torch.long)
        self.samples_y_future = torch.tensor(np.array(self.samples_y_future), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples_x_hist)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_hist": self.samples_x_hist[idx],         # (W, D)
            "x_future": self.samples_x_future[idx],     # (K, D)
            "y_curr": self.samples_y_curr[idx],         # scalar
            "y_future": self.samples_y_future[idx]      # (K,)
        }


def get_dataloaders(
    num_train_episodes: int = 70,
    num_val_episodes: int = 15,
    num_test_episodes: int = 15,
    episode_length: int = 160,
    batch_size: int = config.BATCH_SIZE,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Synthesizes multi-stage episodes and prepares Train, Val, and Test DataLoaders.
    """
    sim = NetworkKillChainSimulator(random_seed=seed)
    
    total_episodes = num_train_episodes + num_val_episodes + num_test_episodes
    X_all, Y_all = sim.generate_dataset(num_episodes=total_episodes, episode_length=episode_length)

    # Train split
    idx_train = num_train_episodes
    idx_val = idx_train + num_val_episodes

    train_ds = NetworkSequenceDataset(X_all[:idx_train], Y_all[:idx_train])
    val_ds = NetworkSequenceDataset(X_all[idx_train:idx_val], Y_all[idx_train:idx_val])
    test_ds = NetworkSequenceDataset(X_all[idx_val:], Y_all[idx_val:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        num_train_episodes=10, num_val_episodes=3, num_test_episodes=3, episode_length=100
    )
    batch = next(iter(train_loader))
    print(f"Batch x_hist: {batch['x_hist'].shape}")
    print(f"Batch x_future: {batch['x_future'].shape}")
    print(f"Batch y_curr: {batch['y_curr'].shape}")
    print(f"Batch y_future: {batch['y_future'].shape}")
