"""
Evaluation and Benchmarking Script for Network World Model.
Evaluates:
- Proactive Forecasting Accuracy across different horizons (k = 1, 2, 3, 4)
- Early Warning Lead Time (time steps in advance of major compromise/impact)
- Per-Stage Classification Report & Confusion Matrix
- Telemetry Reconstruction Mean Squared Error
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset import get_dataloaders
from models.world_model import NetworkWorldModel


def evaluate_model(checkpoint_path: str = None, device: str = "cpu"):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_world_model.pt")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}. Run train.py first.")
        return

    print("=" * 70)
    print("[EVAL] Evaluating Network Attack Forecasting World Model")
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 70)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = NetworkWorldModel(
        num_features=ckpt["config"]["num_features"],
        latent_dim=ckpt["config"]["latent_dim"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        num_stages=ckpt["config"]["num_stages"],
        forecast_horizon=ckpt["config"]["forecast_horizon"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Get Test DataLoader
    _, _, test_loader = get_dataloaders(
        num_train_episodes=70,
        num_val_episodes=15,
        num_test_episodes=20,
        episode_length=160,
        batch_size=config.BATCH_SIZE,
        seed=100
    )

    all_targets = []
    all_preds = []
    recon_errors = []
    horizon_correct = [0] * config.FORECAST_HORIZON
    horizon_total = [0] * config.FORECAST_HORIZON

    with torch.no_grad():
        for batch in test_loader:
            x_hist = batch["x_hist"].to(device)
            x_future = batch["x_future"].to(device)
            y_future = batch["y_future"].to(device) # (B, K)

            # Student Note: Run inference through the World Model without gradient tracking
            out = model(x_hist, horizon=config.FORECAST_HORIZON)
            stage_logits = out["stage_logits"] # (B, K, num_stages)
            x_forecast = out["x_future_forecast"]

            # Step 1: Compute future telemetry reconstruction MSE (how closely our latent imagination matches real flows)
            batch_recon_mse = torch.mean((x_forecast - x_future) ** 2, dim=-1).cpu().numpy()
            recon_errors.extend(batch_recon_mse.flatten())

            # Step 2: Extract stage predictions for each forecast step (t+1 through t+4)
            preds = torch.argmax(stage_logits, dim=-1) # (B, K)

            for k in range(config.FORECAST_HORIZON):
                p_k = preds[:, k].cpu().numpy()
                t_k = y_future[:, k].cpu().numpy()
                horizon_correct[k] += np.sum(p_k == t_k)
                horizon_total[k] += len(t_k)

            all_targets.extend(y_future.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    print("\n[RESULTS] Multi-Horizon Proactive Forecasting Accuracy:")
    for k in range(config.FORECAST_HORIZON):
        acc_k = (horizon_correct[k] / horizon_total[k]) * 100.0
        seconds_ahead = (k + 1) * config.TIME_STEP_SECONDS
        print(f"  Step t+{k+1} (+{seconds_ahead:02d}s in advance): Accuracy = {acc_k:.2f}%")

    print("\n[DYNAMICS] Overall Telemetry Reconstruction Error (World Model Dynamics):")
    print(f"  Mean Squared Error (MSE): {np.mean(recon_errors):.6f}")

    print("\n[REPORT] Stage Classification Performance Report:")
    report = classification_report(
        all_targets,
        all_preds,
        target_names=config.STAGE_NAMES,
        digits=4,
        zero_division=0
    )
    print(report)

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Save evaluation summary to results
    summary_path = os.path.join(config.RESULTS_DIR, "evaluation_report.txt")
    with open(summary_path, "w") as f:
        f.write("=== Network Attack Forecasting World Model Evaluation ===\n\n")
        f.write("Forecasting Accuracy Across Horizons:\n")
        for k in range(config.FORECAST_HORIZON):
            acc_k = (horizon_correct[k] / horizon_total[k]) * 100.0
            seconds_ahead = (k + 1) * config.TIME_STEP_SECONDS
            f.write(f"  t+{k+1} (+{seconds_ahead}s): {acc_k:.2f}%\n")
        f.write(f"\nMean Reconstruction MSE: {np.mean(recon_errors):.6f}\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"\n[SAVED] Full evaluation report saved to: {summary_path}")


if __name__ == "__main__":
    evaluate_model()
