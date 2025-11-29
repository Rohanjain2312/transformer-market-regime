"""Focal Loss for Binary Classification - Handles Class Imbalance"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification
    
    FL(pt) = -α(1-pt)^γ * log(pt)
    
    Args:
        alpha (float): Weighting factor for class imbalance (default: 0.75)
                      Higher alpha = more weight to minority class
        gamma (float): Focusing parameter (default: 2.0)
                      Higher gamma = more focus on hard examples
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits), shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)
        
        Returns:
            Focal loss value
        """
        # Get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = probability of correct class
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CostSensitiveFocalLoss(nn.Module):
    """
    Focal Loss with Cost-Sensitive Learning
    
    Assigns different costs to different types of misclassification:
    - Missing Bearish (False Negative): High cost (miss market downturn)
    - False Bearish (False Positive): Medium cost (false alarm)
    
    Args:
        alpha (float): Base weighting factor for class imbalance
        gamma (float): Focusing parameter
        cost_fn (float): Cost of False Negative (missing Bearish)
        cost_fp (float): Cost of False Positive (false Bearish alarm)
    """
    
    def __init__(self, alpha=0.75, gamma=2.0, cost_fn=2.0, cost_fp=1.0):
        super(CostSensitiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cost_fn = cost_fn  # Cost of missing Bearish (worse!)
        self.cost_fp = cost_fp  # Cost of false Bearish alarm
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits), shape (batch_size, 2)
            targets: Ground truth labels (0=Bearish, 1=Bullish), shape (batch_size,)
        """
        # Get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Base focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply cost-sensitive weights
        predictions = inputs.argmax(dim=1)
        
        # False Negatives: Predicted Bullish (1) but actually Bearish (0)
        false_negatives = (predictions == 1) & (targets == 0)
        
        # False Positives: Predicted Bearish (0) but actually Bullish (1)
        false_positives = (predictions == 0) & (targets == 1)
        
        # Apply costs
        cost_weights = torch.ones_like(focal_loss)
        cost_weights[false_negatives] = self.cost_fn
        cost_weights[false_positives] = self.cost_fp
        
        # Final loss
        cost_sensitive_loss = focal_loss * cost_weights
        
        return cost_sensitive_loss.mean()


def get_focal_loss(loss_type='focal', **kwargs):
    """
    Factory function to get focal loss
    
    Args:
        loss_type (str): 'focal' or 'cost_sensitive'
        **kwargs: Parameters for the loss function
    
    Returns:
        Loss function instance
    """
    if loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.75)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'cost_sensitive':
        alpha = kwargs.get('alpha', 0.75)
        gamma = kwargs.get('gamma', 2.0)
        cost_fn = kwargs.get('cost_fn', 2.0)
        cost_fp = kwargs.get('cost_fp', 1.0)
        return CostSensitiveFocalLoss(alpha=alpha, gamma=gamma, 
                                     cost_fn=cost_fn, cost_fp=cost_fp)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FOCAL LOSS TEST")
    print("="*70)
    
    # Test Focal Loss
    batch_size = 32
    num_classes = 2
    
    # Dummy data
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Standard Focal Loss
    print("\n1. Standard Focal Loss (α=0.75, γ=2.0)")
    focal_loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    loss_focal = focal_loss_fn(inputs, targets)
    print(f"   Loss: {loss_focal.item():.4f}")
    
    # Cost-Sensitive Focal Loss
    print("\n2. Cost-Sensitive Focal Loss (FN cost=2.0, FP cost=1.0)")
    cost_focal_loss_fn = CostSensitiveFocalLoss(alpha=0.75, gamma=2.0, 
                                                cost_fn=2.0, cost_fp=1.0)
    loss_cost = cost_focal_loss_fn(inputs, targets)
    print(f"   Loss: {loss_cost.item():.4f}")
    
    # Compare with standard CrossEntropyLoss
    print("\n3. Standard CrossEntropyLoss (for comparison)")
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    loss_ce = ce_loss_fn(inputs, targets)
    print(f"   Loss: {loss_ce.item():.4f}")
    
    print("\n" + "="*70)
    print("FOCAL LOSS TEST COMPLETE")
    print("="*70 + "\n")
