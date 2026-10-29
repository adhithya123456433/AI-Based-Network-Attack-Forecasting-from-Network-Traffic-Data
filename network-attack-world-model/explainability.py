"""
explainability.py
-----------------
PyTorch Autograd-based Feature Attribution engine using Integrated Gradients.
Identifies which network features (SYN rate, port entropy, packet variance, etc.)
caused the World Model to forecast a specific attack stage.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Dict, Any


class ExplainabilityEngine:
    """
    Computes Integrated Gradients feature attribution on the input sequence:
    Attribution_i = (x_i - x_baseline_i) * \int_0^1 \frac{\partial F(x_{baseline} + \alpha(x - x_{baseline}))}{\partial x_i} d\alpha
    """
    def __init__(self, model: torch.nn.Module, feature_names: List[str], stage_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.stage_names = stage_names
        
    def attribute(self, input_tensor: torch.Tensor, target_class: int = None, steps: int = 25) -> np.ndarray:
        """
        Computes feature importance for a single input sequence [1, window_size, in_features].
        Returns attribution score per feature: shape [in_features].
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x = input_tensor.clone().detach().to(device)
        
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dim [1, W, D]
            
        # Baseline is a neutral zero-vector (median/neutral traffic in standardized space)
        baseline = torch.zeros_like(x)
        
        # If target_class is not specified, explain the model's top predicted class
        if target_class is None:
            with torch.no_grad():
                _, stage_logits, _ = self.model(x)
                target_class = torch.argmax(stage_logits, dim=-1).item()
                
        # Generate linearly scaled paths from baseline to input
        alphas = torch.linspace(0.0, 1.0, steps, device=device)
        interpolated_inputs = torch.cat([baseline + alpha * (x - baseline) for alpha in alphas], dim=0)
        interpolated_inputs.requires_grad_(True)
        
        # Forward pass through the model
        _, stage_logits, _ = self.model(interpolated_inputs)
        target_scores = stage_logits[:, target_class]
        
        # Compute gradients of target output with respect to inputs
        grads = torch.autograd.grad(
            outputs=target_scores.sum(),
            inputs=interpolated_inputs,
            create_graph=False
        )[0]
        
        # Riemann sum approximation of the integral
        avg_grads = torch.mean(grads, dim=0, keepdim=True)  # [1, W, D]
        delta = (x - baseline)
        attributions = (delta * avg_grads).squeeze(0)  # [W, D]
        
        # Aggregate across time window: sum of absolute attribution per feature
        feature_importance = torch.sum(torch.abs(attributions), dim=0).detach().cpu().numpy()
        
        # Normalize to percentage (0 - 100%)
        total_imp = np.sum(feature_importance) + 1e-9
        normalized_importance = (feature_importance / total_imp) * 100.0
        
        return normalized_importance, target_class

    def generate_security_alert_report(self, input_tensor: torch.Tensor, output_image_path: str = None) -> Dict[str, Any]:
        """
        Generates a human-readable security alert explanation for network administrators.
        """
        importance_scores, predicted_class_idx = self.attribute(input_tensor)
        predicted_stage = self.stage_names[predicted_class_idx]
        
        # Rank features by importance
        ranked_indices = np.argsort(importance_scores)[::-1]
        ranked_features = [
            {"feature": self.feature_names[i], "attribution_pct": float(np.round(importance_scores[i], 2))}
            for i in ranked_indices
        ]
        
        report = {
            "predicted_attack_stage": predicted_stage,
            "stage_id": int(predicted_class_idx),
            "is_malicious": bool(predicted_class_idx > 0),
            "top_contributing_features": ranked_features[:3],
            "all_feature_attributions": ranked_features
        }
        
        if output_image_path:
            plt.figure(figsize=(9, 4.5))
            y_pos = np.arange(len(self.feature_names))
            top_feats = [rf["feature"] for rf in ranked_features][::-1]
            top_scores = [rf["attribution_pct"] for rf in ranked_features][::-1]
            
            colors = ['#d9534f' if predicted_class_idx > 0 else '#5cb85c'] * len(top_feats)
            plt.barh(y_pos, top_scores, color=colors, alpha=0.85)
            plt.yticks(y_pos, top_feats, fontsize=10)
            plt.xlabel("Attribution Contribution (%)", fontsize=11)
            plt.title(f"Explainable AI Alert: Forecasted State = [{predicted_stage.upper()}]", fontsize=12, fontweight='bold')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_image_path, dpi=150)
            plt.close()
            
        return report
