"""Train with Improvements: Focal Loss + Balanced Sampling + Better Regularization"""

import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config
from src.dataset import MarketRegimeDataset
from src.model import create_model
from src.focal_loss import FocalLoss, AdaptiveFocalLoss
from src.balanced_sampler import RegimeBalancedBatchSampler
from torch.utils.data import DataLoader

# Try to import Colab display
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False


class ImprovedTrainer:
    """
    Enhanced trainer with:
    - Focal Loss for hard example mining
    - Balanced batch sampling
    - Better learning rate scheduling
    """
    
    def __init__(self, model, train_loader, val_loader, device, model_name, 
                 use_focal_loss=True, gamma=2.0, use_adaptive_alpha=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        # Training configuration
        self.is_gpu = device.type == 'cuda'
        self.num_epochs = config.NUM_EPOCHS_GPU if self.is_gpu else config.NUM_EPOCHS_CPU
        
        # Loss function
        if use_focal_loss:
            if use_adaptive_alpha:
                # Compute class distribution
                class_counts = torch.zeros(config.NUM_CLASSES)
                for _, y in train_loader:
                    for label in y:
                        class_counts[label] += 1
                
                self.criterion = AdaptiveFocalLoss(gamma=gamma)
                self.criterion.update_alpha(class_counts)
                print(f"\nUsing Adaptive Focal Loss (gamma={gamma}):")
                print(f"  Class counts: {class_counts.tolist()}")
                print(f"  Alpha weights: {[f'{a:.3f}' for a in self.criterion.alpha.tolist()]}")
            else:
                # Use fixed alpha based on inverse frequency
                class_counts = torch.zeros(config.NUM_CLASSES)
                for _, y in train_loader:
                    for label in y:
                        class_counts[label] += 1
                
                total = class_counts.sum()
                alpha = total / (class_counts * config.NUM_CLASSES)
                alpha = alpha / alpha.sum() * config.NUM_CLASSES
                
                self.criterion = FocalLoss(alpha=alpha.tolist(), gamma=gamma)
                print(f"\nUsing Focal Loss (gamma={gamma}):")
                print(f"  Class counts: {class_counts.tolist()}")
                print(f"  Alpha weights: {[f'{a:.3f}' for a in alpha.tolist()]}")
        else:
            # Standard cross-entropy with class weights
            class_counts = torch.zeros(config.NUM_CLASSES)
            for _, y in train_loader:
                for label in y:
                    class_counts[label] += 1
            
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES
            class_weights = class_weights.to(device)
            
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            print(f"\nUsing CrossEntropyLoss with class weights:")
            print(f"  Weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")
        
        # Optimizer with slightly lower LR for stability
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE * 0.5,  # Lower LR for more stable training
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = None
        if config.USE_LR_SCHEDULER and self.is_gpu:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=config.LR_SCHEDULER_FACTOR,
                patience=config.LR_SCHEDULER_PATIENCE,
                min_lr=config.LR_SCHEDULER_MIN_LR
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'per_regime_val_acc': {regime: [] for regime in config.REGIME_NAMES}
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Per-regime tracking
        regime_correct = {i: 0 for i in range(config.NUM_CLASSES)}
        regime_total = {i: 0 for i in range(config.NUM_CLASSES)}
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                # Per-regime accuracy
                for regime_idx in range(config.NUM_CLASSES):
                    mask = y == regime_idx
                    regime_total[regime_idx] += mask.sum().item()
                    regime_correct[regime_idx] += (pred[mask] == y[mask]).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Per-regime accuracy
        per_regime_acc = {}
        for regime_idx, regime_name in enumerate(config.REGIME_NAMES):
            if regime_total[regime_idx] > 0:
                acc = 100. * regime_correct[regime_idx] / regime_total[regime_idx]
            else:
                acc = 0.0
            per_regime_acc[regime_name] = acc
        
        return avg_loss, accuracy, per_regime_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = config.CHECKPOINT_DIR / self.model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': self.history['val_acc'][-1],
            'history': self.history,
            'model_name': self.model_name
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            best_path = checkpoint_dir / 'best_improved.pth'
            torch.save(checkpoint, best_path)
            print(f"    ✓ Saved best model: {best_path}")
    
    def train(self):
        """Complete training loop"""
        print("\n" + "="*70)
        print("TRAINING START (WITH IMPROVEMENTS)")
        print("="*70)
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, per_regime_acc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            for regime_name, acc in per_regime_acc.items():
                self.history['per_regime_val_acc'][regime_name].append(acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"Per-Regime Val Acc:")
            for regime_name, acc in per_regime_acc.items():
                print(f"  {regime_name}: {acc:.2f}%")
            print(f"LR: {current_lr:.6f}")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_acc)
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                print(f"*** New best validation accuracy: {val_acc:.2f}% ***")
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            if is_best:
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping
            if self.is_gpu and self.epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return self.history


def plot_training_history(history, model_name, save_path=None):
    """Plot training curves including per-regime accuracy"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Per-Regime Accuracy
    ax3 = fig.add_subplot(gs[1, :])
    colors = ['red', 'gray', 'green']
    for regime_name, color in zip(config.REGIME_NAMES, colors):
        ax3.plot(epochs, history['per_regime_val_acc'][regime_name], 
                color=color, label=regime_name, linewidth=2, marker='o', markersize=4)
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Per-Regime Validation Accuracy', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning Rate
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Training History (Improved)', 
                 fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved training curves: {save_path}")
        if IN_COLAB:
            display(Image(save_path))
    
    if not IN_COLAB:
        plt.show()
    plt.close()


def main(args):
    """Main training pipeline with improvements"""
    
    print("\n" + "="*70)
    print("TRAINING WITH IMPROVEMENTS")
    print("="*70)
    print(f"\nModel: {args.model_id}")
    print(f"Use Focal Loss: {args.use_focal_loss}")
    print(f"Focal Gamma: {args.focal_gamma}")
    print(f"Use Balanced Sampling: {args.use_balanced_sampling}")
    
    # Load data
    print("\nLoading data...")
    train_data = pd.read_csv(
        config.REGIME_DATA_DIR / "train_labeled_engineered.csv",
        index_col=0, parse_dates=True
    )
    val_data = pd.read_csv(
        config.REGIME_DATA_DIR / "val_labeled_engineered.csv",
        index_col=0, parse_dates=True
    )
    test_data = pd.read_csv(
        config.REGIME_DATA_DIR / "test_labeled_engineered.csv",
        index_col=0, parse_dates=True
    )
    
    # Get model config
    model_config = config.get_model_config(args.model_id)
    feature_list = model_config['features']
    
    # Create datasets
    train_dataset = MarketRegimeDataset(train_data, feature_list)
    val_dataset = MarketRegimeDataset(val_data, feature_list)
    test_dataset = MarketRegimeDataset(test_data, feature_list)
    
    # Create dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.BATCH_SIZE_GPU if device.type == 'cuda' else config.BATCH_SIZE_CPU
    
    if args.use_balanced_sampling and device.type == 'cuda':
        # Use balanced batch sampler
        print("\nUsing RegimeBalancedBatchSampler...")
        # Get labels from dataset (after sequence windowing)
        # The dataset creates sequences, so we need labels for valid sequence positions
        train_labels = []
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            train_labels.append(label.item())
        train_labels = np.array(train_labels)
        
        batch_sampler = RegimeBalancedBatchSampler(
            train_labels, 
            batch_size=batch_size,
            drop_last=True
        )
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    else:
        # Standard sampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"\nInitializing {model_config['name']}...")
    model, device = create_model(args.model_id)
    
    # Train with improvements
    model_name_improved = f"{args.model_id}_improved"
    trainer = ImprovedTrainer(
        model, train_loader, val_loader, device,
        model_name=model_name_improved,
        use_focal_loss=args.use_focal_loss,
        gamma=args.focal_gamma,
        use_adaptive_alpha=True
    )
    
    history = trainer.train()
    
    # Plot training history
    plot_path = config.DATA_DIR / f"{model_name_improved}_history.png"
    plot_training_history(history, model_config['name'], save_path=plot_path)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    model.eval()
    correct = 0
    total = 0
    regime_correct = {i: 0 for i in range(config.NUM_CLASSES)}
    regime_total = {i: 0 for i in range(config.NUM_CLASSES)}
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = outputs.argmax(dim=1)
            
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            for regime_idx in range(config.NUM_CLASSES):
                mask = y == regime_idx
                regime_total[regime_idx] += mask.sum().item()
                regime_correct[regime_idx] += (pred[mask] == y[mask]).sum().item()
    
    test_acc = 100. * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"\nPer-Regime Test Accuracy:")
    for regime_idx, regime_name in enumerate(config.REGIME_NAMES):
        if regime_total[regime_idx] > 0:
            acc = 100. * regime_correct[regime_idx] / regime_total[regime_idx]
        else:
            acc = 0.0
        print(f"  {regime_name}: {acc:.2f}%")
    
    print("\n" + "="*70)
    print("TRAINING WITH IMPROVEMENTS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with improvements')
    
    parser.add_argument('--model_id', type=str, default='model_2_large_capacity',
                       help='Model to train (default: model_2_large_capacity)')
    
    parser.add_argument('--use_focal_loss', type=bool, default=True,
                       help='Use focal loss instead of CE (default: True)')
    
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter (default: 2.0)')
    
    parser.add_argument('--use_balanced_sampling', type=bool, default=True,
                       help='Use balanced batch sampling (default: True)')
    
    args = parser.parse_args()
    
    main(args)