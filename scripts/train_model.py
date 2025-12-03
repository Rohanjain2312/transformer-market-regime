"""Train Binary Classification with Focal Loss: Bearish vs Bullish"""

import sys
from pathlib import Path
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
            'val_mcc': [],
            'learning_rate': [],
            'bearish_acc': [],
            'bullish_acc': []
        }
        
        # Best model tracking - based on MCC
        self.best_val_acc = 0.0
        self.best_val_mcc = -1.0
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
        checkpoint_dir = config.CHECKPOINT_DIR / self.model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': self.history['val_acc'][-1],
            'val_mcc': self.history['val_mcc'][-1],
            'best_val_mcc': self.best_val_mcc,
            'history': self.history,
            'model_name': self.model_name,
            'num_classes': 2
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            best_path = checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Complete training loop"""
        print("\n" + "="*70)
        print(f"TRAINING: {self.model_name}")
        print("="*70)
        
        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, bearish_acc, bullish_acc, val_mcc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_mcc'].append(val_mcc)
            self.history['learning_rate'].append(current_lr)
            self.history['bearish_acc'].append(bearish_acc)
            self.history['bullish_acc'].append(bullish_acc)
            
            # Print every 10 epochs or last epoch
            if epoch % 10 == 0 or epoch == self.num_epochs:
                print(f"Epoch {epoch}/{self.num_epochs}: "
                      f"Val Acc={val_acc:.1f}% | MCC={val_mcc:.3f} | "
                      f"Bearish={bearish_acc:.1f}% | Bullish={bullish_acc:.1f}%")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_acc)
            
            # Check if best model - based on MCC
            is_best = val_mcc > self.best_val_mcc
            if is_best:
                self.best_val_mcc = val_mcc
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                print(f"  ✓ New best MCC: {val_mcc:.3f}")
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            if is_best:
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping
            if self.is_gpu and self.epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\n✓ Training complete: Best MCC={self.best_val_mcc:.3f} (Epoch {self.best_epoch})")
        
        return self.history


def main():
    """Main training pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train binary classifier')
    parser.add_argument('--model_id', type=str, required=True,
                       help='Model to train (e.g., model_0_baseline)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BINARY REGIME CLASSIFICATION")
    print("="*70)
    print(f"Model: {args.model_id}")
    
    # Get model config
    model_config = config.get_model_config(args.model_id)
    feature_list = model_config['features']
    feature_set = 'engineered' if feature_list == config.ENGINEERED_FEATURES else 'baseline'
    
    print(f"Features: {feature_set.upper()} ({len(feature_list)} features)")
    
    # Load data
    print("\nLoading data...", end=" ", flush=True)
    train_data = pd.read_csv(
        config.REGIME_DATA_DIR / f"train_labeled_{feature_set}.csv",
        index_col=0, parse_dates=True
    )
    val_data = pd.read_csv(
        config.REGIME_DATA_DIR / f"val_labeled_{feature_set}.csv",
        index_col=0, parse_dates=True
    )
    test_data = pd.read_csv(
        config.REGIME_DATA_DIR / f"test_labeled_{feature_set}.csv",
        index_col=0, parse_dates=True
    )
    print(f"✓ (Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)})")
    
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
    print(f"Creating model...", end=" ", flush=True)
    model, device = create_model(args.model_id)
    print(f"✓ ({model.get_num_parameters():,} parameters, {device})")
    
    # Train
    trainer = BinaryTrainerFocal(model, train_loader, val_loader, device, model_name=args.model_id)
    history = trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("TEST EVALUATION")
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
    test_mcc = matthews_corrcoef(all_labels, all_preds)
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test MCC: {test_mcc:.3f}")
    print("\nPer-Class Performance:")
    print(classification_report(all_labels, all_preds, target_names=['Bearish', 'Bullish'], digits=3))
    
    print("\n" + "="*70)
    print("✓ COMPLETE")
    print("="*70)
    print(f"Checkpoint: checkpoints/{args.model_id}/best.pth\n")


if __name__ == "__main__":
    main()