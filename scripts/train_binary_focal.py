"""Train Binary Classification with Focal Loss: Bearish vs Bullish"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import seaborn as sns

import config
from src.dataset import MarketRegimeDataset
from src.model import create_model
from src.focal_loss import get_focal_loss
from torch.utils.data import DataLoader

# Try to import Colab display
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False


class BinaryTrainerFocal:
    """Trainer for binary classification with Focal Loss"""
    
    def __init__(self, model, train_loader, val_loader, device, model_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        # Training configuration
        self.is_gpu = device.type == 'cuda'
        self.num_epochs = config.NUM_EPOCHS_GPU if self.is_gpu else config.NUM_EPOCHS_CPU
        self.num_classes = 2
        
        # Loss function - Focal Loss or standard
        if config.USE_FOCAL_LOSS:
            self.criterion = get_focal_loss(
                loss_type=config.FOCAL_LOSS_TYPE,
                alpha=config.FOCAL_ALPHA,
                gamma=config.FOCAL_GAMMA,
                cost_fn=config.COST_FN,
                cost_fp=config.COST_FP
            )
            print(f"Using {config.FOCAL_LOSS_TYPE.upper()} Focal Loss:")
            print(f"  Alpha: {config.FOCAL_ALPHA}")
            print(f"  Gamma: {config.FOCAL_GAMMA}")
            if config.FOCAL_LOSS_TYPE == 'cost_sensitive':
                print(f"  Cost FN (miss Bearish): {config.COST_FN}")
                print(f"  Cost FP (false Bearish): {config.COST_FP}")
        else:
            # Standard CrossEntropyLoss with class weights
            class_counts = torch.zeros(self.num_classes)
            for _, y in train_loader:
                for label in y:
                    class_counts[label] += 1
            
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * self.num_classes
            class_weights = class_weights.to(device)
            
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using Standard CrossEntropyLoss with class weights")
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = None
        if config.USE_LR_SCHEDULER and self.is_gpu:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
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
            'val_mcc': [],  # Track MCC
            'learning_rate': [],
            'bearish_acc': [],
            'bullish_acc': []
        }
        
        # Best model tracking - NOW BASED ON MCC
        self.best_val_acc = 0.0  # Still track for reference
        self.best_val_mcc = -1.0  # PRIMARY metric (MCC ranges -1 to 1)
        self.best_epoch = 0
        self.epochs_no_improve = 0
        
        print(f"\nTrainer Configuration:")
        print(f"  Device: {device}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning Rate: {config.LEARNING_RATE}")
    
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
            
            # For binary, model outputs 3 classes but we only use first 2
            outputs_binary = outputs[:, :2]
            
            loss = self.criterion(outputs_binary, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = outputs_binary.argmax(dim=1)
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
        
        # Per-class tracking
        bearish_correct = 0
        bearish_total = 0
        bullish_correct = 0
        bullish_total = 0
        
        # For MCC calculation
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                outputs_binary = outputs[:, :2]
                
                loss = self.criterion(outputs_binary, y)
                
                total_loss += loss.item()
                pred = outputs_binary.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                # Collect for MCC
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
                # Per-class accuracy
                bearish_mask = y == 0
                bearish_total += bearish_mask.sum().item()
                bearish_correct += (pred[bearish_mask] == y[bearish_mask]).sum().item()
                
                bullish_mask = y == 1
                bullish_total += bullish_mask.sum().item()
                bullish_correct += (pred[bullish_mask] == y[bullish_mask]).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        bearish_acc = 100. * bearish_correct / bearish_total if bearish_total > 0 else 0
        bullish_acc = 100. * bullish_correct / bullish_total if bullish_total > 0 else 0
        
        # Calculate MCC
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_mcc = matthews_corrcoef(all_labels, all_preds)
        
        return avg_loss, accuracy, bearish_acc, bullish_acc, val_mcc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = config.CHECKPOINT_DIR / f"{self.model_name}_binary_focal"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': self.history['val_acc'][-1],
            'val_mcc': self.history['val_mcc'][-1],  # Save MCC
            'best_val_mcc': self.best_val_mcc,  # Save best MCC
            'history': self.history,
            'model_name': self.model_name,
            'num_classes': 2,
            'focal_loss_config': {
                'use_focal_loss': config.USE_FOCAL_LOSS,
                'focal_loss_type': config.FOCAL_LOSS_TYPE,
                'alpha': config.FOCAL_ALPHA,
                'gamma': config.FOCAL_GAMMA,
                'cost_fn': config.COST_FN,
                'cost_fp': config.COST_FP
            }
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            best_path = checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"    ✓ Saved best model (MCC: {self.best_val_mcc:.3f}): {best_path}")
    
    def train(self):
        """Complete training loop"""
        print("\n" + "="*70)
        print("BINARY CLASSIFICATION TRAINING WITH FOCAL LOSS")
        print("="*70)
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate - now returns MCC too
            val_loss, val_acc, bearish_acc, bullish_acc, val_mcc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_mcc'].append(val_mcc)  # Store MCC
            self.history['learning_rate'].append(current_lr)
            self.history['bearish_acc'].append(bearish_acc)
            self.history['bullish_acc'].append(bullish_acc)
            
            # Print metrics - NOW INCLUDES MCC
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | MCC: {val_mcc:.3f}")
            print(f"  Bearish: {bearish_acc:.2f}% | Bullish: {bullish_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")
            
            # Learning rate scheduling - STILL BASED ON ACCURACY (for stability)
            if self.scheduler:
                self.scheduler.step(val_acc)
            
            # Check if best model - NOW BASED ON MCC
            is_best = val_mcc > self.best_val_mcc
            if is_best:
                self.best_val_mcc = val_mcc
                self.best_val_acc = val_acc  # Update for reference
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                print(f"*** New best MCC: {val_mcc:.3f} (Acc: {val_acc:.2f}%) ***")
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            if is_best:
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping - BASED ON MCC
            if self.is_gpu and self.epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"No improvement in MCC for {config.EARLY_STOPPING_PATIENCE} epochs")
                break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best MCC:     {self.best_val_mcc:.3f} (Epoch {self.best_epoch})")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return self.history


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=['Bearish', 'Bullish'],
               yticklabels=['Bearish', 'Bullish'],
               cbar_kws={'label': 'Proportion'}, ax=ax, square=True)
    ax.set_title(f'{model_name} - Confusion Matrix (Focal Loss)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix: {save_path}")
        if IN_COLAB:
            display(Image(save_path))
    
    if not IN_COLAB:
        plt.show()
    plt.close()


def main():
    """Main training pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train binary classifier with Focal Loss')
    parser.add_argument('--model_id', type=str, default='model_2_large_capacity',
                       help='Model to train (default: model_2_large_capacity)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BINARY CLASSIFICATION WITH FOCAL LOSS: BEARISH VS BULLISH")
    print("="*70)
    print(f"\nModel: {args.model_id}")
    
    # Load binary data
    print("\nLoading binary data...")
    try:
        train_data = pd.read_csv(
            config.REGIME_DATA_DIR / "train_labeled_engineered_binary.csv",
            index_col=0, parse_dates=True
        )
        val_data = pd.read_csv(
            config.REGIME_DATA_DIR / "val_labeled_engineered_binary.csv",
            index_col=0, parse_dates=True
        )
        test_data = pd.read_csv(
            config.REGIME_DATA_DIR / "test_labeled_engineered_binary.csv",
            index_col=0, parse_dates=True
        )
    except FileNotFoundError:
        print("\nERROR: Binary labeled data not found!")
        print("Please run: python scripts/create_binary_labels.py")
        return
    
    print(f"\nData loaded:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    
    # Get model config
    try:
        model_config = config.get_model_config(args.model_id)
    except ValueError as e:
        print(f"\nERROR: {e}")
        print(f"Available models: {config.list_available_models()}")
        return
    
    feature_list = model_config['features']
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MarketRegimeDataset(train_data, feature_list)
    val_dataset = MarketRegimeDataset(val_data, feature_list)
    test_dataset = MarketRegimeDataset(test_data, feature_list)
    
    # Create dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.BATCH_SIZE_GPU if device.type == 'cuda' else config.BATCH_SIZE_CPU
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nInitializing {model_config['name']}...")
    model, device = create_model(args.model_id)
    print(f"  Parameters: {model.get_num_parameters():,}")
    
    # Train
    trainer = BinaryTrainerFocal(model, train_loader, val_loader, device, model_name=args.model_id)
    history = trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            outputs_binary = outputs[:, :2]
            pred = outputs_binary.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    test_acc = (all_preds == all_labels).mean() * 100
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Bearish', 'Bullish'], digits=3))
    
    # Plot confusion matrix
    cm_path = config.DATA_DIR / f"{args.model_id}_binary_focal_confusion.png"
    plot_confusion_matrix(all_labels, all_preds, model_config['name'], save_path=cm_path)
    
    print("\n" + "="*70)
    print("FOCAL LOSS TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved to: checkpoints/{args.model_id}_binary_focal/")
    print("\n")


if __name__ == "__main__":
    main()