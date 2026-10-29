"""
losses.py
---------
Custom loss functions designed for network security deep learning:
1. Multi-class Focal Loss for handling extreme class imbalance.
2. Smooth Huber Loss for robust future state regression.
3. Joint Multi-Task Loss combining forecasting and attack classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Prevents the vast majority of easy 'Normal' samples from dominating gradients,
    forcing the network to focus on hard, rare multi-stage attack transitions.
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [batch_size, num_classes]
        # targets: [batch_size]
        probs = F.softmax(logits, dim=-1)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Gather predicted probability for the target class
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - target_probs) ** self.gamma
        focal_loss = modulating_factor * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_factors = self.alpha.gather(0, targets)
            focal_loss = alpha_factors * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class JointMultiTaskLoss(nn.Module):
    """
    Balances future telemetry forecasting error (Huber loss)
    and attack stage classification error (Focal loss).
    
    L_total = alpha * L_forecast + beta * L_stage
    """
    def __init__(self, alpha: float = 0.5, beta: float = 1.0, class_weights: torch.Tensor = None, gamma: float = 2.0):
        super(JointMultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.forecast_loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber Loss
        self.stage_loss_fn = FocalLoss(alpha=class_weights, gamma=gamma)
        
    def forward(self, pred_future: torch.Tensor, true_future: torch.Tensor,
                pred_stage_logits: torch.Tensor, true_stage: torch.Tensor):
        l_forecast = self.forecast_loss_fn(pred_future, true_future)
        l_stage = self.stage_loss_fn(pred_stage_logits, true_stage)
        total_loss = (self.alpha * l_forecast) + (self.beta * l_stage)
        return total_loss, l_forecast, l_stage
