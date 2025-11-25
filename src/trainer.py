"""Training Module: Training loop with GPU/CPU auto-detection and LR scheduling"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

class Trainer:
    """
    Handles model training, validation, and checkpointing.
    Auto-detects GPU availability and adjusts training accordingly.
    """
    
    def __init__(self, model, train_loader, val_loader, device, model_name='model_0_baseline'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        # Training configuration based on device
        self.is_gpu = device.type == 'cuda'
        self.batch_size = config.BATCH_SIZE_GPU if self.is_gpu else config.BATCH_SIZE_CPU
        self.num_epochs = config.NUM_EPOCHS_GPU if self.is_gpu else config.NUM_EPOCHS_CPU
        
        # Calculate class weights for imbalanced data
        class_counts = torch.zeros(config.NUM_CLASSES)
        for _, y in train_loader:
            for label in y:
                class_counts[label] += 1
        
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES
        class_weights = class_weights.to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
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
                mode='max',  # Maximize validation accuracy
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
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        
        print(f"\nTraining Configuration:")
        print(f"    Device: {device}")
        print(f"    Mode: {'GPU (Full)' if self.is_gpu else 'CPU (Quick Test)'}")
        print(f"    Epochs: {self.num_epochs}")
        print(f"    Batch Size: {self.batch_size}")
        print(f"    Learning Rate: {config.LEARNING_RATE}")
        print(f"    LR Scheduler: {'Enabled' if self.scheduler else 'Disabled'}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
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
        
        latest_path = checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"    Saved best model: {best_path}")
    
    def train(self):
        """Complete training loop"""
        print("\n" + "="*70)
        print("TRAINING START")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"LR:         {current_lr:.6f}")
            
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
            if config.SAVE_BEST_ONLY and is_best:
                self.save_checkpoint(epoch, is_best=True)
            elif not config.SAVE_BEST_ONLY:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.is_gpu and self.epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total Time:        {elapsed_time:.2f}s ({elapsed_time/60:.1f}m)")
        print(f"Best Val Acc:      {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"Final Train Acc:   {self.history['train_acc'][-1]:.2f}%")
        print(f"Final Val Acc:     {self.history['val_acc'][-1]:.2f}%")
        
        return self.history
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


def reduce_dataset_for_cpu(train_loader, val_loader, ratio=0.25):
    """Reduce dataset size for CPU training"""
    from torch.utils.data import Subset, DataLoader
    
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    train_size = int(len(train_dataset) * ratio)
    val_size = int(len(val_dataset) * ratio)
    
    train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
    val_indices = np.random.choice(len(val_dataset), val_size, replace=False)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    train_loader_reduced = DataLoader(
        train_subset, 
        batch_size=config.BATCH_SIZE_CPU,
        shuffle=True,
        drop_last=True
    )
    val_loader_reduced = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE_CPU,
        shuffle=False
    )
    
    print(f"\nDataset reduced to {ratio*100}% for CPU mode")
    print(f"    Train batches: {len(train_loader)} -> {len(train_loader_reduced)}")
    print(f"    Val batches:   {len(val_loader)} -> {len(val_loader_reduced)}")
    
    return train_loader_reduced, val_loader_reduced


if __name__ == "__main__":
    import pandas as pd
    from dataset import create_dataloaders
    from model import create_model
    
    print("\n" + "="*70)
    print("TRAINER TEST")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train = pd.read_csv(config.REGIME_DATA_DIR / "train_labeled_baseline.csv",
                        index_col=0, parse_dates=True)
    val = pd.read_csv(config.REGIME_DATA_DIR / "val_labeled_baseline.csv",
                      index_col=0, parse_dates=True)
    test = pd.read_csv(config.REGIME_DATA_DIR / "test_labeled_baseline.csv",
                       index_col=0, parse_dates=True)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, n_features = create_dataloaders(
        train, val, test, config.BASELINE_FEATURES
    )
    
    # Create model
    print("\nInitializing model...")
    model, device = create_model('model_0_baseline')
    
    # Reduce dataset if CPU
    if device.type == 'cpu':
        train_loader, val_loader = reduce_dataset_for_cpu(train_loader, val_loader)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, device, model_name='model_0_baseline')
    history = trainer.train()
    
    print("\n" + "="*70)
    print("TRAINER TEST COMPLETE")
    print("="*70 + "\n")