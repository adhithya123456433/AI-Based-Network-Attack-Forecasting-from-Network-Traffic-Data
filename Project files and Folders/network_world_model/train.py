"""
Training Pipeline for Network Attack Forecasting World Model.
Trains the latent dynamics, state decoder, and proactive attack stage forecaster.
"""
import torch
import torch.optim as optim
import numpy as np
import time
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset import get_dataloaders
from models.world_model import NetworkWorldModel


def train_model(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    device: str = "cpu"
):
    print("=" * 70)
    print("[TRAIN] Training AI-Based Network Attack Forecasting World Model")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"Observation Window W={config.WINDOW_SIZE} | Forecast Horizon K={config.FORECAST_HORIZON}")
    print("=" * 70)

    # 1. Prepare Data
    print("Generating multi-stage attack and benign traffic datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        num_train_episodes=80,
        num_val_episodes=15,
        num_test_episodes=15,
        episode_length=160,
        batch_size=batch_size
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    # 2. Instantiate Model
    model = NetworkWorldModel(
        num_features=config.NUM_FEATURES,
        latent_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_stages=config.NUM_STAGES,
        forecast_horizon=config.FORECAST_HORIZON,
        dropout=config.DROPOUT
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_recon_mse": []
    }

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_world_model.pt")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total_targets = 0

        # Student Note: Iterate through each training batch
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            # Step 1: Forward pass through the World Model and compute multi-task loss
            # (Reconstruction Loss + Latent Transition Consistency + Stage Cross-Entropy + Threat BCE)
            loss, metrics = model.compute_losses(batch)
            loss.backward()

            # Step 2: Clip gradient norm to 1.0 to prevent exploding gradients in recurrent GRU dynamics
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += loss.item() * batch["x_hist"].size(0)

            # Step 3: Compute attack stage forecasting accuracy across the horizon (K steps ahead)
            with torch.no_grad():
                out = model(batch["x_hist"], horizon=config.FORECAST_HORIZON)
                preds = torch.argmax(out["stage_logits"], dim=-1) # (batch, K)
                targets = batch["y_future"]                      # (batch, K)
                train_correct += (preds == targets).sum().item()
                train_total_targets += targets.numel()

        train_loss = train_loss_accum / len(train_loader.dataset)
        train_acc = (train_correct / train_total_targets) * 100.0

        # Validation loop
        model.eval()
        val_loss_accum = 0.0
        val_recon_accum = 0.0
        val_correct = 0
        val_total_targets = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, metrics = model.compute_losses(batch)
                val_loss_accum += loss.item() * batch["x_hist"].size(0)
                val_recon_accum += metrics["recon_loss"] * batch["x_hist"].size(0)

                out = model(batch["x_hist"], horizon=config.FORECAST_HORIZON)
                preds = torch.argmax(out["stage_logits"], dim=-1)
                targets = batch["y_future"]
                val_correct += (preds == targets).sum().item()
                val_total_targets += targets.numel()

        val_loss = val_loss_accum / len(val_loader.dataset)
        val_recon = val_recon_accum / len(val_loader.dataset)
        val_acc = (val_correct / val_total_targets) * 100.0

        scheduler.step(val_loss)
        epoch_dur = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_recon_mse"].append(val_recon)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_dur:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Recon MSE: {val_recon:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": {
                    "num_features": config.NUM_FEATURES,
                    "latent_dim": config.LATENT_DIM,
                    "hidden_dim": config.HIDDEN_DIM,
                    "num_stages": config.NUM_STAGES,
                    "forecast_horizon": config.FORECAST_HORIZON
                }
            }, checkpoint_path)
            print(f"  [SAVED] Saved new best checkpoint (Val Loss: {best_val_loss:.4f}) -> {checkpoint_path}")

    print("=" * 70)
    print("[SUCCESS] Training Completed Successfully!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return model, history


if __name__ == "__main__":
    train_model(epochs=15, batch_size=64)
