"""Train Binary Models with SMOTE-Augmented Data"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import torch
import numpy as np

import config
from src.dataset import MarketRegimeDataset
from src.model import create_model
from src.focal_loss import get_focal_loss
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef


class BinaryTrainerSMOTE:
    """Trainer for binary classification with SMOTE-augmented data"""
    
    def __init__(self, model, train_loader, val_loader, device, model_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        self.is_gpu = device.type == 'cuda'
        self.num_epochs = config.NUM_EPOCHS_GPU if self.is_gpu else config.NUM_EPOCHS_CPU
        self.num_classes = 2
        
        # Focal Loss
        if config.USE_FOCAL_LOSS:
            self.criterion = get_focal_loss(
                loss_type=config.FOCAL_LOSS_TYPE,
                alpha=config.FOCAL_ALPHA,
                gamma=config.FOCAL_GAMMA,
                cost_fn=config.COST_FN,
                cost_fp=config.COST_FP
            )
            print(f"Using {config.FOCAL_LOSS_TYPE.upper()} Focal Loss with SMOTE data")
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss with SMOTE data")
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # LR Scheduler
        self.scheduler = None
        if config.USE_LR_SCHEDULER and self.is_gpu:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max',
                factor=config.LR_SCHEDULER_FACTOR,
                patience=config.LR_SCHEDULER_PATIENCE,
                min_lr=config.LR_SCHEDULER_MIN_LR
            )
        
        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_mcc': [],
            'learning_rate': [], 'bearish_acc': [], 'bullish_acc': []
        }
        
        # Best model tracking (MCC-based)
        self.best_val_acc = 0.0
        self.best_val_mcc = -1.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
    
    def train_epoch(self):
        """Train one epoch"""
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
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Validate with MCC"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        bearish_correct = bearish_total = 0
        bullish_correct = bullish_total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                outputs_binary = outputs[:, :2]
                loss = self.criterion(outputs_binary, y)
                
                pred = outputs_binary.argmax(dim=1)
                total_loss += loss.item()
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
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
        
        val_mcc = matthews_corrcoef(np.array(all_labels), np.array(all_preds))
        
        return avg_loss, accuracy, bearish_acc, bullish_acc, val_mcc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint_dir = config.CHECKPOINT_DIR / f"{self.model_name}_binary_smote"
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
            'num_classes': 2,
            'smote_augmented': True
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            best_path = checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"    ✓ Saved best model (MCC: {self.best_val_mcc:.3f}): {best_path}")
    
    def train(self):
        """Training loop"""
        print("\n" + "="*70)
        print("TRAINING WITH SMOTE-AUGMENTED DATA")
        print("="*70)
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, bearish_acc, bullish_acc, val_mcc = self.validate()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_mcc'].append(val_mcc)
            self.history['learning_rate'].append(current_lr)
            self.history['bearish_acc'].append(bearish_acc)
            self.history['bullish_acc'].append(bullish_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | MCC: {val_mcc:.3f}")
            print(f"  Bearish: {bearish_acc:.2f}% | Bullish: {bullish_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")
            
            if self.scheduler:
                self.scheduler.step(val_acc)
            
            is_best = val_mcc > self.best_val_mcc
            if is_best:
                self.best_val_mcc = val_mcc
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                print(f"*** New best MCC: {val_mcc:.3f} (Acc: {val_acc:.2f}%) ***")
            else:
                self.epochs_no_improve += 1
            
            if is_best:
                self.save_checkpoint(epoch, is_best=True)
            
            if self.is_gpu and self.epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping after {epoch} epochs")
                break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE (SMOTE)")
        print("="*70)
        print(f"Best MCC:     {self.best_val_mcc:.3f} (Epoch {self.best_epoch})")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return self.history


def main():
    parser = argparse.ArgumentParser(description='Train with SMOTE data')
    parser.add_argument('--model_id', type=str, default='model_5_multiscale',
                       help='Model to train')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BINARY CLASSIFICATION WITH SMOTE-AUGMENTED DATA")
    print("="*70)
    print(f"\nModel: {args.model_id}")
    
    # Load SMOTE-augmented training data
    print("\nLoading SMOTE-augmented training data...")
    
    model_config = config.get_model_config(args.model_id)
    feature_list = model_config['features']
    feature_set = 'engineered' if feature_list == config.ENGINEERED_FEATURES else 'baseline'
    
    train_smote_file = config.REGIME_DATA_DIR / f"train_labeled_{feature_set}_binary_smote.csv"
    
    if not train_smote_file.exists():
        print(f"\nERROR: SMOTE data not found: {train_smote_file}")
        print("Please run: python scripts/create_smote_data.py")
        return
    
    train_data = pd.read_csv(train_smote_file, index_col=0, parse_dates=True)
    
    # Load regular validation data (NOT augmented)
    val_data = pd.read_csv(
        config.REGIME_DATA_DIR / f"val_labeled_{feature_set}_binary.csv",
        index_col=0, parse_dates=True
    )
    
    # Load test data
    test_data = pd.read_csv(
        config.REGIME_DATA_DIR / f"test_labeled_{feature_set}_binary.csv",
        index_col=0, parse_dates=True
    )
    
    print(f"\nData loaded:")
    print(f"  Train (SMOTE): {len(train_data)} samples")
    print(f"  Val:           {len(val_data)} samples")
    print(f"  Test:          {len(test_data)} samples")
    
    # Create datasets
    train_dataset = MarketRegimeDataset(train_data, feature_list)
    val_dataset = MarketRegimeDataset(val_data, feature_list)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.BATCH_SIZE_GPU if device.type == 'cuda' else config.BATCH_SIZE_CPU
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\nInitializing {model_config['name']}...")
    model, device = create_model(args.model_id)
    print(f"  Parameters: {model.get_num_parameters():,}")
    
    # Train
    trainer = BinaryTrainerSMOTE(model, train_loader, val_loader, device, args.model_id)
    history = trainer.train()
    
    print("\n" + "="*70)
    print("SMOTE TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved to: checkpoints/{args.model_id}_binary_smote/")
    print("\nNext: Run compare_smote_results.py to compare with non-SMOTE models")
    print()


if __name__ == "__main__":
    main()
