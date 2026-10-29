"""
generate_trajectory_plot.py
---------------------------
Generates an intuitive presentation figure showing:
1. Past observed traffic window vs. the World Model's 5-step future forecast.
2. The Kill-Chain attack stage progression.
3. The early warning lead time.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_pipeline import generate_synthetic_attack_telemetry, prepare_temporal_dataloaders, FEATURE_NAMES, STAGE_NAMES
from models import NetworkAttackWorldModel


def main():
    print("[*] Generating visual trajectory forecast plot...")
    
    # 1. Load data
    df = generate_synthetic_attack_telemetry(total_timesteps=4500, random_seed=42)
    window_size = 30
    forecast_horizon = 5
    
    _, _, test_loader, scaler, _ = prepare_temporal_dataloaders(
        df=df, window_size=window_size, forecast_horizon=forecast_horizon, batch_size=32
    )
    
    # 2. Initialize and load model
    model = NetworkAttackWorldModel(in_features=len(FEATURE_NAMES), latent_dim=64, horizon=forecast_horizon, num_classes=5)
    model.eval()
    
    # Extract an attack sequence from test loader
    sample_x = None
    sample_y = None
    sample_stage = None
    
    for x_past, y_future, y_stage in test_loader:
        for i in range(len(y_stage)):
            if y_stage[i].item() == 1:  # Reconnaissance stage
                sample_x = x_past[i]
                sample_y = y_future[i]
                sample_stage = y_stage[i].item()
                break
        if sample_x is not None:
            break
            
    with torch.no_grad():
        pred_future, stage_logits, _ = model(sample_x.unsqueeze(0))
        pred_stage = torch.argmax(stage_logits, dim=-1).item()
        
    sample_x_np = sample_x.numpy()
    sample_y_np = sample_y.numpy()
    pred_future_np = pred_future.squeeze(0).numpy()
    
    # Unscale packet_rate (feature index 0) and syn_flag_ratio (feature index 2) for intuitive physical units
    # RobustScaler inverse on selected columns: x_orig = x * scale + center
    packet_rate_idx = FEATURE_NAMES.index("packet_rate")
    syn_flag_idx = FEATURE_NAMES.index("syn_flag_ratio")
    
    past_steps = np.arange(-window_size + 1, 1)
    future_steps = np.arange(1, forecast_horizon + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    
    # Top Plot: Packet Rate Forecast
    ax1 = axes[0]
    ax1.plot(past_steps, sample_x_np[:, packet_rate_idx], color="royalblue", lw=2.2, label="Past Observed Telemetry (t - 30 to t)")
    ax1.plot(future_steps, sample_y_np[:, packet_rate_idx], color="forestgreen", lw=2.2, label="Actual Future Ground Truth (t+1 to t+5)")
    ax1.plot(future_steps, pred_future_np[:, packet_rate_idx], color="crimson", lw=2.4, linestyle="--", marker="o", label="World Model Forecast (t+1 to t+5)")
    ax1.axvline(x=0, color="dimgray", linestyle=":", lw=1.5, label="Current Time (Decision Boundary t)")
    ax1.axvspan(0, forecast_horizon, color="crimson", alpha=0.08, label="Early Warning Horizon (+5 steps)")
    ax1.set_ylabel("Packet Rate (Scaled)", fontsize=11)
    ax1.set_title("World Model Telemetry Forecasting: Anticipating Incoming Attack Surges", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9.5)
    
    # Bottom Plot: SYN Flag Ratio (Protocol signature of Port Scan)
    ax2 = axes[1]
    ax2.plot(past_steps, sample_x_np[:, syn_flag_idx], color="darkorange", lw=2.2, label="Observed SYN Flag Ratio")
    ax2.plot(future_steps, sample_y_np[:, syn_flag_idx], color="forestgreen", lw=2.2, label="Actual Future SYN Ratio")
    ax2.plot(future_steps, pred_future_np[:, syn_flag_idx], color="crimson", lw=2.4, linestyle="--", marker="s", label="World Model Forecasted SYN Ratio")
    ax2.axvline(x=0, color="dimgray", linestyle=":", lw=1.5)
    ax2.axvspan(0, forecast_horizon, color="crimson", alpha=0.08)
    ax2.set_xlabel("Time Steps Relative to Current Moment (t)", fontsize=11)
    ax2.set_ylabel("SYN Flag Ratio (Scaled)", fontsize=11)
    ax2.set_title(f"Early Kill-Chain Prediction: [Forecasted State: {STAGE_NAMES[pred_stage].upper()}]", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=9.5)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "trajectory_forecast_demo.png")
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[Done] Trajectory forecast demo plot saved to: {output_path}")


if __name__ == "__main__":
    main()
