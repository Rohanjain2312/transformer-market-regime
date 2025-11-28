"""Focal Loss: Addresses class imbalance by focusing on hard examples"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Addresses class imbalance by:
    1. Down-weighting easy examples (high p_t)
    2. Focusing on hard examples (low p_t)
    
    Args:
        alpha: Weighting factor for each class (list or tensor)
        gamma: Focusing parameter (default: 2.0)
               - gamma = 0: equivalent to CrossEntropyLoss
               - gamma > 0: focuses more on hard examples
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha)
            self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        
        Returns:
            loss: Focal loss value
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of correct class for each sample
        batch_size = inputs.size(0)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)  # (batch_size,)
        
        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross entropy: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss: Automatically adjusts alpha based on class distribution
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, gamma=2.0, reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = None
    
    def update_alpha(self, class_counts):
        """
        Update alpha based on class distribution
        
        Args:
            class_counts: Tensor of class counts (num_classes,)
        """
        total = class_counts.sum()
        alpha = total / (class_counts * len(class_counts))
        self.alpha = alpha / alpha.sum() * len(class_counts)
    
    def forward(self, inputs, targets):
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of correct class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        
        # Focal term
        focal_weight = (1 - p_t) ** self.gamma
        
        # Cross entropy
        ce_loss = -torch.log(p_t + 1e-8)
        
        # Apply adaptive alpha
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    # Test focal loss
    print("\n" + "="*70)
    print("FOCAL LOSS TEST")
    print("="*70)
    
    batch_size = 32
    num_classes = 3
    
    # Create dummy data
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test standard focal loss
    print("\n1. Standard Focal Loss (no class weights):")
    criterion = FocalLoss(gamma=2.0)
    loss = criterion(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test with class weights
    print("\n2. Focal Loss with Class Weights:")
    alpha = [1.0, 2.0, 1.5]  # Weight Neutral (class 1) more
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    loss = criterion(logits, targets)
    print(f"   Alpha: {alpha}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Test adaptive focal loss
    print("\n3. Adaptive Focal Loss:")
    class_counts = torch.tensor([100, 50, 120])  # Imbalanced
    criterion = AdaptiveFocalLoss(gamma=2.0)
    criterion.update_alpha(class_counts)
    loss = criterion(logits, targets)
    print(f"   Class counts: {class_counts.tolist()}")
    print(f"   Computed alpha: {criterion.alpha.tolist()}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Compare with standard CE
    print("\n4. Comparison with Standard CrossEntropyLoss:")
    ce_criterion = nn.CrossEntropyLoss()
    ce_loss = ce_criterion(logits, targets)
    focal_criterion = FocalLoss(gamma=2.0)
    focal_loss = focal_criterion(logits, targets)
    print(f"   CE Loss:    {ce_loss.item():.4f}")
    print(f"   Focal Loss: {focal_loss.item():.4f}")
    
    print("\n" + "="*70)
    print("FOCAL LOSS TEST COMPLETE")
    print("="*70 + "\n")