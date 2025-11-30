"""Train Binary Classification: Bearish vs Bullish (drop Neutral)"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import seaborn as sns

import config
from src.dataset import MarketRegimeDataset
from src.model import create_model
from torch.utils.data import DataLoader

# Try to import Colab display
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False


class BinaryTrainer:
    """Trainer for binary classification (Bearish vs Bullish)"""
    
    def __init__(self, model, train_loader, val_loader, device, model_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        # Training configuration
        self.is_gpu = device.type == 'cuda'
        self.num_epochs = config.NUM_EPOCHS_GPU if self.is_gpu else config.NUM_EPOCHS_CPU
        
        # Binary classification - 2 classes
        self.num_classes = 2
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
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
        
        print(f"\nBinary Trainer Configuration:")
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
        checkpoint_dir = config.CHECKPOINT_DIR / f"{self.model_name}_binary"
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
            'num_classes': 2
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
        print("BINARY CLASSIFICATION TRAINING")
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
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return self.history


def plot_training_history(history, model_name, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Overall Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Per-Class Accuracy
    axes[1, 0].plot(epochs, history['bearish_acc'], 'red', label='Bearish', linewidth=2, marker='o')
    axes[1, 0].plot(epochs, history['bullish_acc'], 'green', label='Bullish', linewidth=2, marker='o')
    axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Per-Class Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Binary Classification Training', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved training curves: {save_path}")
        if IN_COLAB:
            display(Image(save_path))
    
    if not IN_COLAB:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=['Bearish', 'Bullish'],
               yticklabels=['Bearish', 'Bullish'],
               cbar_kws={'label': 'Proportion'}, ax=ax, square=True)
    ax.set_title('Binary Classification - Confusion Matrix', fontsize=13, fontweight='bold', pad=15)
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


def main(args):
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("BINARY CLASSIFICATION: BEARISH VS BULLISH")
    print("="*70)
    print(f"\nModel: {args.model_id}")
    
    # Load binary data
    print("\nLoading binary data...")
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
    
    print(f"\nData loaded:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"\nInitializing {model_config['name']}...")
    model, device = create_model(args.model_id)
    
    # Train
    trainer = BinaryTrainer(model, train_loader, val_loader, device, model_name=args.model_id)
    history = trainer.train()
    
    # Plot training history
    plot_path = config.DATA_DIR / f"{args.model_id}_binary_history.png"
    plot_training_history(history, model_config['name'], save_path=plot_path)
    
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
    cm_path = config.DATA_DIR / f"{args.model_id}_binary_confusion.png"
    plot_confusion_matrix(all_labels, all_preds, save_path=cm_path)
    
    print("\n" + "="*70)
    print("BINARY CLASSIFICATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train binary classifier (Bearish vs Bullish)')
    
    parser.add_argument('--model_id', type=str, default='model_2_large_capacity',
                       help='Model to train (default: model_2_large_capacity)')
    
    args = parser.parse_args()
    
    main(args)
