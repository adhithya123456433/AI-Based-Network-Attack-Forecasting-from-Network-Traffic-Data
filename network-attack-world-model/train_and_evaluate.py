"""
train_and_evaluate.py
---------------------
Training loop, early stopping, validation monitoring, baseline benchmarking,
and metric plotting for the Network Attack World Model.
"""

import os
import copy
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

from losses import JointMultiTaskLoss
from data_pipeline import STAGE_NAMES


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    num_epochs: int = 15,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 4,
    device: str = "cpu"
):
    """
    Trains the model with:
    - Joint Multi-Task Loss (Huber + Focal)
    - Weight decay (L2 regularization)
    - Cosine Annealing learning rate schedule
    - Early Stopping monitored on Validation Macro-F1
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = JointMultiTaskLoss(alpha=0.4, beta=1.0, class_weights=class_weights.to(device), gamma=2.0)
    
    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    
    print(f"\n[Training] Starting {model.__class__.__name__} on {device} (Epochs={num_epochs}, Patience={patience})...")
    
    for epoch in range(1, num_epochs + 1):
        # 1. Train step
        model.train()
        train_loss_accum = 0.0
        for x_past, y_future, y_stage in train_loader:
            x_past = x_past.to(device)
            y_future = y_future.to(device)
            y_stage = y_stage.to(device)
            
            optimizer.zero_grad()
            pred_future, stage_logits, _ = model(x_past)
            loss, _, _ = criterion(pred_future, y_future, stage_logits, y_stage)
            loss.backward()
            
            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss_accum += loss.item()
            
        avg_train_loss = train_loss_accum / len(train_loader)
        scheduler.step()
        
        # 2. Validation step
        model.eval()
        val_loss_accum = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_past, y_future, y_stage in val_loader:
                x_past = x_past.to(device)
                y_future = y_future.to(device)
                y_stage = y_stage.to(device)
                
                pred_future, stage_logits, _ = model(x_past)
                loss, _, _ = criterion(pred_future, y_future, stage_logits, y_stage)
                val_loss_accum += loss.item()
                
                preds = torch.argmax(stage_logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_stage.cpu().numpy())
                
        avg_val_loss = val_loss_accum / len(val_loader)
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        val_acc = accuracy_score(all_targets, all_preds)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        
        print(f"  Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1 (Macro): {val_f1:.4f} | Val Acc: {val_acc*100:.1f}%")
        
        # Early Stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Early Stopping Triggered] Validation F1 did not improve for {patience} consecutive epochs.")
                break
                
    # Restore best weights
    model.load_state_dict(best_weights)
    print(f"  [Model Saved] Loaded best model checkpoint with Val Macro-F1 = {best_val_f1:.4f}")
    return history


def evaluate_model(model: torch.nn.Module, test_loader, device: str = "cpu"):
    """
    Evaluates the model on unseen test data for both:
    1. Telemetry forecasting (MAE, RMSE)
    2. Attack stage classification (Accuracy, Macro-F1, Precision, Recall, Confusion Matrix)
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    future_errors = []
    
    with torch.no_grad():
        for x_past, y_future, y_stage in test_loader:
            x_past = x_past.to(device)
            y_future = y_future.to(device)
            
            pred_future, stage_logits, _ = model(x_past)
            
            # Forecast error
            err = torch.abs(pred_future - y_future).mean(dim=[1, 2]).cpu().numpy()
            future_errors.extend(err)
            
            preds = torch.argmax(stage_logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_stage.cpu().numpy())
            
    metrics = {
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "macro_f1": float(f1_score(all_targets, all_preds, average='macro', zero_division=0)),
        "precision": float(precision_score(all_targets, all_preds, average='macro', zero_division=0)),
        "recall": float(recall_score(all_targets, all_preds, average='macro', zero_division=0)),
        "forecast_mae": float(np.mean(future_errors)),
        "confusion_matrix": confusion_matrix(all_targets, all_preds, labels=list(range(len(STAGE_NAMES))))
    }
    return metrics, all_preds, all_targets


def plot_evaluation_report(history: dict, metrics: dict, output_path: str = "training_metrics.png"):
    """
    Plots training convergence curves and test confusion matrix.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Plot 1: Loss & F1 curves
    ax1 = axes[0]
    ax1.plot(history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    ax1.plot(history["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title("Training & Validation Loss Convergence", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Secondary axis for F1
    ax1_twin = ax1.twinx()
    ax1_twin.plot(history["val_f1"], label="Val Macro-F1", color="forestgreen", lw=1.8, linestyle=":")
    ax1_twin.set_ylabel("Macro-F1 Score", color="forestgreen", fontsize=10)
    ax1_twin.tick_params(axis='y', labelcolor="forestgreen")
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    
    # Plot 2: Confusion Matrix
    ax2 = axes[1]
    cm = metrics["confusion_matrix"]
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title("Test Set Attack Stage Confusion Matrix", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(STAGE_NAMES))
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(STAGE_NAMES, rotation=35, ha="right", fontsize=9)
    ax2.set_yticklabels(STAGE_NAMES, fontsize=9)
    
    # Annotate numbers in confusion matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    ax2.set_ylabel("True Stage", fontsize=10)
    ax2.set_xlabel("Predicted Stage", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot Saved] Report visualization saved to: {output_path}")
