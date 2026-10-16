"""
Live Interactive Demonstration & Simulation of the Network Attack Forecasting World Model.
Simulates a live streaming network session, performs proactive multi-step attack forecasting,
and invokes the XAI engine to explain the features influencing the prediction.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.killchain_simulator import NetworkKillChainSimulator
from models.world_model import NetworkWorldModel
from explainability.attribution import IntegratedGradientsExplainer


def run_live_simulation(checkpoint_path: str = None, save_plot: bool = True):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_world_model.pt")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}. Train the model first using train.py.")
        return

    print("=" * 80)
    print("[SIMULATION] REAL-TIME NETWORK ATTACK FORECASTING WORLD MODEL")
    print("=" * 80)

    # Step 1: Load our best trained World Model weights from the checkpoint directory
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = NetworkWorldModel(
        num_features=ckpt["config"]["num_features"],
        latent_dim=ckpt["config"]["latent_dim"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        num_stages=ckpt["config"]["num_stages"],
        forecast_horizon=ckpt["config"]["forecast_horizon"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Step 2: Initialize the Integrated Gradients Explainer for Explainable AI (XAI)
    # This explains WHICH network features (like port entropy or byte ratios) caused the prediction
    explainer = IntegratedGradientsExplainer(model, steps=20)

    # Step 3: Generate a live network attack scenario to simulate incoming network traffic
    sim = NetworkKillChainSimulator(random_seed=777)
    episode_len = 100
    X_stream, Y_stream = sim.generate_episode(episode_type="full_attack", length=episode_len)

    print(f"Simulating live network stream of {episode_len} timesteps ({episode_len * config.TIME_STEP_SECONDS} seconds total)...")
    print("-" * 80)

    # Step 4: Define our observation window (W=12 steps = past 60s) and forecast horizon (K=4 steps = +20s ahead)
    W = config.WINDOW_SIZE
    K = config.FORECAST_HORIZON

    timesteps = []
    current_stages = []
    forecasted_stages = []
    early_warning_scores = []

    # Stream timesteps
    sample_alerts_triggered = 0

    for t in range(W, episode_len - K):
        x_window = torch.tensor(X_stream[t - W:t], dtype=torch.float32).unsqueeze(0)
        curr_stage = Y_stream[t - 1]
        future_true_stage = Y_stream[t + K - 1]

        with torch.no_grad():
            out = model(x_window, horizon=K)
            pred_logits = out["stage_logits"] # (1, K, num_stages)
            alarm_prob = out["early_warning_prob"][0, -1, 0].item()
            pred_future_stage = int(torch.argmax(pred_logits[0, -1]).item())

        timesteps.append(t)
        current_stages.append(curr_stage)
        forecasted_stages.append(pred_future_stage)
        early_warning_scores.append(alarm_prob)

        # Trigger proactive alert if future attack is forecasted while current stage is still early or benign
        if pred_future_stage > curr_stage and alarm_prob > 0.65 and sample_alerts_triggered < 3:
            sample_alerts_triggered += 1
            print(f"\n[ALERT] PROACTIVE EARLY WARNING @ t={t * config.TIME_STEP_SECONDS}s")
            print(f"  Current Status:    {config.STAGE_NAMES[curr_stage]}")
            print(f"  Forecasted Stage:  {config.STAGE_NAMES[pred_future_stage]} (in +{K * config.TIME_STEP_SECONDS}s)")
            print(f"  Threat Confidence: {alarm_prob * 100:.1f}%")

            # Explain the prediction using XAI
            explanation = explainer.explain_future_stage(x_window, target_stage=pred_future_stage, forecast_step=-1)
            narrative = explainer.generate_narrative_explanation(explanation, stage_idx=pred_future_stage)
            print(narrative)
            print("-" * 80)

    print("\n[COMPLETE] Simulation Stream Completed.")

    # 3. Generate Visual Dashboard Artifact
    if save_plot:
        plt.figure(figsize=(14, 8))

        time_axis = np.array(timesteps) * config.TIME_STEP_SECONDS

        # Subplot 1: Stages progression
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, current_stages, label="Actual Current Stage", color="#1f77b4", linewidth=2.5)
        plt.plot(time_axis, forecasted_stages, label=f"Forecasted Stage (+{K * config.TIME_STEP_SECONDS}s ahead)", color="#d62728", linestyle="--", linewidth=2)
        plt.yticks(range(config.NUM_STAGES), [s.split("(")[0].strip() for s in config.STAGE_NAMES])
        plt.title("AI-Based Network Attack Forecasting: World Model Temporal Trajectory", fontsize=14, fontweight="bold")
        plt.ylabel("Cyber Kill Chain Stage", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left")

        # Subplot 2: Threat Alarm Probability
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, early_warning_scores, label="Early Warning Alarm Probability", color="#ff7f0e", linewidth=2)
        plt.axhline(0.65, color="red", linestyle=":", label="Proactive Alert Threshold (0.65)")
        plt.xlabel("Simulation Time (Seconds)", fontsize=11)
        plt.ylabel("Threat Probability", fontsize=11)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left")

        plt.tight_layout()
        plot_path = os.path.join(config.RESULTS_DIR, "forecast_timeline.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[PLOT] Visual forecast trajectory chart saved to: {plot_path}")


if __name__ == "__main__":
    run_live_simulation()
