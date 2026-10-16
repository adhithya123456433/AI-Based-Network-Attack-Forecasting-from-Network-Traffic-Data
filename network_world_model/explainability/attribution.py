"""
Explainable AI (XAI) Attribution Module.
Implements Integrated Gradients and Temporal Feature Attribution to explain:
1. Which network features triggered the transition into a high-risk future state.
2. How each feature contributes to the predicted Cyber Kill Chain attack stage.
"""
import torch
import numpy as np
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class IntegratedGradientsExplainer:
    def __init__(self, model: torch.nn.Module, steps: int = 25):
        """
        Args:
            model: Trained NetworkWorldModel
            steps: Number of Riemann interpolation steps for path integral
        """
        self.model = model
        self.steps = steps
        self.feature_names = config.FEATURE_NAMES

    def explain_future_stage(
        self,
        x_hist: torch.Tensor,
        target_stage: Optional[int] = None,
        forecast_step: int = -1,
        baseline: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Computes feature importance attributions for a predicted future stage.
        Args:
            x_hist: Tensor of shape (1, window_size, num_features)
            target_stage: Stage index (0 to 4) to explain. If None, uses top predicted stage.
            forecast_step: Horizon index to explain (e.g. -1 for the furthest forecast step)
            baseline: Neutral baseline tensor (default: zeros)
        Returns:
            Dictionary mapping feature names to their attribution scores (percentages)
        """
        self.model.eval()
        if x_hist.dim() == 2:
            x_hist = x_hist.unsqueeze(0)

        device = next(self.model.parameters()).device
        x_input = x_hist.clone().detach().to(device)

        if baseline is None:
            baseline = torch.zeros_like(x_input).to(device)
        else:
            baseline = baseline.to(device)

        # 1. Determine target stage if not provided
        with torch.no_grad():
            outputs = self.model(x_input)
            stage_logits = outputs["stage_logits"] # (1, K, num_stages)
            target_logits_at_step = stage_logits[0, forecast_step]
            if target_stage is None:
                target_stage = int(torch.argmax(target_logits_at_step).item())

        # 2. Integrated Gradients path interpolation
        alphas = torch.linspace(0.0, 1.0, self.steps, device=device)
        total_gradients = torch.zeros_like(x_input)

        diff = x_input - baseline

        for alpha in alphas:
            interpolated = baseline + alpha * diff
            interpolated.requires_grad_(True)

            out = self.model(interpolated)
            logits = out["stage_logits"]
            score = logits[0, forecast_step, target_stage]

            self.model.zero_grad()
            score.backward(retain_graph=False)

            total_gradients += interpolated.grad.detach()

        # Average gradient along the path * input difference
        avg_gradients = total_gradients / self.steps
        attributions = avg_gradients * diff # (1, W, D)

        # Aggregate across time window W: sum of absolute attribution
        feature_importance = torch.sum(torch.abs(attributions), dim=(0, 1)).cpu().numpy()

        # Normalize to percentage
        total_imp = np.sum(feature_importance) + 1e-9
        normalized_importance = (feature_importance / total_imp) * 100.0

        explanation_dict = {
            self.feature_names[i]: float(normalized_importance[i])
            for i in range(len(self.feature_names))
        }

        # Sort descending by importance
        sorted_explanation = dict(
            sorted(explanation_dict.items(), key=lambda item: item[1], reverse=True)
        )

        return sorted_explanation

    def generate_narrative_explanation(self, explanation: Dict[str, float], stage_idx: int) -> str:
        """
        Produces human-readable security alert insights for SOC teams.
        """
        stage_name = config.STAGE_NAMES[stage_idx]
        top_features = list(explanation.items())[:3]

        narrative = [
            f"[XAI] Threat Forecasting Analysis for Predicted Stage: [{stage_name}]",
            "Key Influential Network Telemetry Signatures:"
        ]

        for rank, (feat, score) in enumerate(top_features, 1):
            narrative.append(f"  {rank}. {feat}: {score:.2f}% influence")

        if stage_idx == 1:
            narrative.append("  [ACTION] Proactive Recommendation: High scanning activity detected. Apply temporary rate limits and check perimeter firewall blocklists.")
        elif stage_idx == 2:
            narrative.append("  [ACTION] Proactive Recommendation: Exploit/Auth surge predicted. Inspect target service logs and isolate vulnerable endpoint.")
        elif stage_idx == 3:
            narrative.append("  [ACTION] Proactive Recommendation: Internal spreading signature. Sever lateral session tokens and enforce strict micro-segmentation.")
        elif stage_idx == 4:
            narrative.append("  [ACTION] Proactive Recommendation: Imminent data exfiltration/impact. Throttle outbound egress gateways and alert incident responders.")
        else:
            narrative.append("  [STATUS] System state is within normal operating baselines.")

        return "\n".join(narrative)
