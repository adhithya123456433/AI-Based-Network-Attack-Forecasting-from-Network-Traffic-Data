"""
main.py
-------
Main execution script for AI-Based Network Attack Forecasting World Model:
1. Generates / Prepares temporal network traffic data with zero look-ahead bias.
2. Trains the proposed Latent Dynamics World Model.
3. Benchmarks against Static MLP and Vanilla LSTM baselines.
4. Generates an Explainability report using Integrated Gradients for an attack alert.
5. Saves visualization plots.
"""

import os
import torch
import pandas as pd
# standard formatting without external dependencies

from data_pipeline import (
    generate_synthetic_attack_telemetry,
    prepare_temporal_dataloaders,
    FEATURE_NAMES,
    STAGE_NAMES
)
from models import NetworkAttackWorldModel, StaticMLPBaseline, VanillaSeq2SeqLSTM
from train_and_evaluate import train_model, evaluate_model, plot_evaluation_report
from explainability import ExplainabilityEngine


def print_comparison_table(results: dict):
    print("\n" + "="*80)
    print("                     EMPIRICAL BENCHMARK COMPARISON TABLE")
    print("="*80)
    header = f"{'Model':<28} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Forecast MAE':<14} | {'Status':<12}"
    print(header)
    print("-" * 80)
    for model_name, m in results.items():
        row = f"{model_name:<28} | {m['accuracy']*100:>8.2f}% | {m['macro_f1']:>10.4f} | {m['forecast_mae']:>14.4f} | {'Evaluated':<12}"
        print(row)
    print("="*80 + "\n")


def main():
    print("=" * 80)
    print(" AI-BASED NETWORK ATTACK FORECASTING USING LATENT WORLD MODEL")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Execution device: {device}")
    
    # 1. Prepare Data
    print("\n[Step 1/5] Synthesizing continuous multi-stage temporal network telemetry...")
    df = generate_synthetic_attack_telemetry(total_timesteps=4500, random_seed=42)
    print(f"    Total traffic records: {len(df)}")
    print(f"    Features: {len(FEATURE_NAMES)} {FEATURE_NAMES[:4]}...")
    print(f"    Attack stages distribution:\n{df['stage'].value_counts().to_dict()}")
    
    window_size = 30
    forecast_horizon = 5
    batch_size = 32
    
    print("\n[Step 2/5] Creating zero-leakage chronological train/val/test splits...")
    train_loader, val_loader, test_loader, scaler, class_weights = prepare_temporal_dataloaders(
        df=df,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size
    )
    print(f"    Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    
    # 2. Train Proposed World Model
    print("\n[Step 3/5] Training Proposed Latent Dynamics World Model...")
    world_model = NetworkAttackWorldModel(
        in_features=len(FEATURE_NAMES),
        latent_dim=64,
        horizon=forecast_horizon,
        num_classes=len(STAGE_NAMES),
        dropout=0.25
    )
    
    wm_history = train_model(
        model=world_model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        num_epochs=12,
        lr=0.001,
        weight_decay=1e-4,
        patience=4,
        device=device
    )
    
    wm_metrics, wm_preds, wm_targets = evaluate_model(world_model, test_loader, device=device)
    print(f"\n[World Model Test Results] Accuracy: {wm_metrics['accuracy']*100:.2f}% | Macro-F1: {wm_metrics['macro_f1']:.4f} | Forecast MAE: {wm_metrics['forecast_mae']:.4f}")
    
    # Save training metrics plot
    metrics_plot_path = os.path.join(os.path.dirname(__file__), "training_metrics.png")
    plot_evaluation_report(wm_history, wm_metrics, output_path=metrics_plot_path)
    
    # 3. Train Baselines for Comparison
    print("\n[Step 4/5] Training Baseline Models for Comparative Evaluation...")
    
    # Baseline 1: Static MLP
    print("\n  -> Training Baseline 1: Static Feed-Forward MLP (Non-temporal)...")
    mlp_baseline = StaticMLPBaseline(
        in_features=len(FEATURE_NAMES),
        horizon=forecast_horizon,
        num_classes=len(STAGE_NAMES),
        dropout=0.25
    )
    _ = train_model(mlp_baseline, train_loader, val_loader, class_weights, num_epochs=8, lr=0.001, patience=3, device=device)
    mlp_metrics, _, _ = evaluate_model(mlp_baseline, test_loader, device=device)
    
    # Baseline 2: Vanilla Seq2Seq LSTM
    print("\n  -> Training Baseline 2: Vanilla Seq2Seq LSTM...")
    lstm_baseline = VanillaSeq2SeqLSTM(
        in_features=len(FEATURE_NAMES),
        hidden_dim=64,
        horizon=forecast_horizon,
        num_classes=len(STAGE_NAMES),
        dropout=0.25
    )
    _ = train_model(lstm_baseline, train_loader, val_loader, class_weights, num_epochs=8, lr=0.001, patience=3, device=device)
    lstm_metrics, _, _ = evaluate_model(lstm_baseline, test_loader, device=device)
    
    # Print Benchmark Comparison Table
    results_summary = {
        "Static MLP (Non-Temporal)": mlp_metrics,
        "Vanilla Seq2Seq LSTM": lstm_metrics,
        "Proposed Latent World Model": wm_metrics
    }
    print_comparison_table(results_summary)
    
    # 4. Explainable AI Demo
    print("\n[Step 5/5] Generating Explainability Alert for a Test Attack Window...")
    explainer = ExplainabilityEngine(world_model, FEATURE_NAMES, STAGE_NAMES)
    
    # Find a test sample where an attack stage is accurately forecasted by the World Model
    alert_sample = None
    target_stage = 0
    world_model.eval()
    for x_past, y_future, y_stage in test_loader:
        with torch.no_grad():
            _, logits, _ = world_model(x_past)
            preds = torch.argmax(logits, dim=-1)
        for i in range(len(y_stage)):
            if preds[i].item() > 0:  # Successfully forecasted malicious stage
                alert_sample = x_past[i]
                target_stage = y_stage[i].item()
                break
        if alert_sample is not None:
            break
            
    if alert_sample is not None:
        explain_plot_path = os.path.join(os.path.dirname(__file__), "explainability_alert.png")
        report = explainer.generate_security_alert_report(alert_sample, output_image_path=explain_plot_path)
        
        print("\n" + "="*60)
        print("               AUTOMATED SECURITY ALERT REPORT")
        print("="*60)
        print(f"  Forecasted Cyber Stage : {report['predicted_attack_stage'].upper()} (Stage ID {report['stage_id']})")
        print(f"  Ground Truth Severity  : {STAGE_NAMES[target_stage].upper()}")
        print(f"  Threat Status          : {'MALICIOUS (High Priority)' if report['is_malicious'] else 'BENIGN'}")
        print("\n  Key Drivers Identified by Explainability Engine (Integrated Gradients):")
        for rank, item in enumerate(report["top_contributing_features"], 1):
            print(f"    {rank}. {item['feature']:<20} -> Attribution: {item['attribution_pct']:>5.2f}%")
        print("="*60)
        print(f"\n[Explainability Chart Saved] Saved to: {explain_plot_path}")
    
    print("\n[COMPLETE] End-to-end Deep Learning pipeline successfully finished.")


if __name__ == "__main__":
    main()
