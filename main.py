"""Main Orchestrator: End-to-end pipeline for Transformer Market Regime Classification"""

import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import config
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.regime_labeling import RegimeLabeler
from src.dataset import create_dataloaders
from src.model import create_model
from src.trainer import Trainer, reduce_dataset_for_cpu

def run_data_pipeline():
    """Execute data acquisition and preprocessing"""
    print("\n" + "="*70)
    print("PHASE 1: DATA PIPELINE")
    print("="*70)
    
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()
    
    return data

def run_feature_engineering(train, val, test, feature_set='baseline'):
    """Execute feature engineering"""
    print("\n" + "="*70)
    print(f"PHASE 2: FEATURE ENGINEERING ({feature_set.upper()})")
    print("="*70)
    
    engineer = FeatureEngineer()
    train_feat, val_feat, test_feat, feature_list = engineer.process_dataset(
        train, val, test, feature_set
    )
    engineer.save_features(train_feat, val_feat, test_feat, feature_set)
    
    return train_feat, val_feat, test_feat, feature_list

def run_regime_labeling(train, val, test, feature_set='baseline'):
    """Execute HMM regime labeling"""
    print("\n" + "="*70)
    print("PHASE 3: REGIME LABELING")
    print("="*70)
    
    labeler = RegimeLabeler()
    train_labeled, val_labeled, test_labeled = labeler.label_dataset(train, val, test)
    labeler.save_labeled_data(train_labeled, val_labeled, test_labeled, feature_set)
    
    return train_labeled, val_labeled, test_labeled

def run_training(train, val, test, feature_list, model_name='baseline'):
    """Execute model training"""
    print("\n" + "="*70)
    print(f"PHASE 4: MODEL TRAINING ({model_name.upper()})")
    print("="*70)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, n_features = create_dataloaders(
        train, val, test, feature_list
    )
    
    # Create model
    print("Initializing model...")
    model, device = create_model(n_features)
    
    # Reduce dataset if CPU
    if device.type == 'cpu':
        train_loader, val_loader = reduce_dataset_for_cpu(train_loader, val_loader)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, device, model_name=model_name)
    history = trainer.train()
    
    return trainer, history, test_loader

def evaluate_model(trainer, test_loader):
    """Evaluate model on test set"""
    print("\n" + "="*70)
    print("PHASE 5: FINAL EVALUATION")
    print("="*70)
    
    trainer.model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(trainer.device), y.to(trainer.device)
            outputs = trainer.model(x)
            pred = outputs.argmax(dim=1)
            
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = 100. * correct / total
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Regime-wise accuracy
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=config.REGIME_NAMES,
                                digits=3))
    
    return test_acc

def plot_training_history(history, save_path=None):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved training curves: {save_path}")
    plt.show()

def main(args):
    """Main execution pipeline"""
    
    print("\n" + "="*70)
    print("TRANSFORMER MARKET REGIME CLASSIFICATION")
    print("="*70)
    print(f"\nMode: {args.mode}")
    print(f"Feature Set: {args.feature_set}")
    
    # Phase 1: Data Pipeline
    if args.mode in ['all', 'data']:
        data = run_data_pipeline()
        train = data['train']
        val = data['val']
        test = data['test']
        if args.mode == 'data':
            return
    
    # Load data if starting from features
    if args.mode in ['features', 'train', 'evaluate']:
        print("\nLoading existing data...")
        train = pd.read_csv(config.PROCESSED_DATA_DIR / "train_data.csv",
                           index_col=0, parse_dates=True)
        val = pd.read_csv(config.PROCESSED_DATA_DIR / "val_data.csv",
                         index_col=0, parse_dates=True)
        test = pd.read_csv(config.PROCESSED_DATA_DIR / "test_data.csv",
                          index_col=0, parse_dates=True)
    
    # Phase 2: Feature Engineering
    if args.mode in ['all', 'features']:
        train_feat, val_feat, test_feat, feature_list = run_feature_engineering(
            train, val, test, args.feature_set
        )
        if args.mode == 'features':
            return
    
    # Load features if starting from labeling
    if args.mode in ['label', 'train', 'evaluate']:
        print("\nLoading existing features...")
        suffix = f"_{args.feature_set}"
        train_feat = pd.read_csv(config.PROCESSED_DATA_DIR / f"train_features{suffix}.csv",
                                index_col=0, parse_dates=True)
        val_feat = pd.read_csv(config.PROCESSED_DATA_DIR / f"val_features{suffix}.csv",
                              index_col=0, parse_dates=True)
        test_feat = pd.read_csv(config.PROCESSED_DATA_DIR / f"test_features{suffix}.csv",
                               index_col=0, parse_dates=True)
        feature_list = config.get_active_features()
    
    # Phase 3: Regime Labeling
    if args.mode in ['all', 'label']:
        train_labeled, val_labeled, test_labeled = run_regime_labeling(
            train_feat, val_feat, test_feat, args.feature_set
        )
        if args.mode == 'label':
            return
    
    # Load labeled data if starting from training
    if args.mode in ['train', 'evaluate']:
        print("\nLoading labeled data...")
        suffix = f"_{args.feature_set}"
        train_labeled = pd.read_csv(config.REGIME_DATA_DIR / f"train_labeled{suffix}.csv",
                                    index_col=0, parse_dates=True)
        val_labeled = pd.read_csv(config.REGIME_DATA_DIR / f"val_labeled{suffix}.csv",
                                  index_col=0, parse_dates=True)
        test_labeled = pd.read_csv(config.REGIME_DATA_DIR / f"test_labeled{suffix}.csv",
                                   index_col=0, parse_dates=True)
        feature_list = config.get_active_features()
    
    # Phase 4: Training
    if args.mode in ['all', 'train']:
        model_name = f"{args.feature_set}_transformer"
        trainer, history, test_loader = run_training(
            train_labeled, val_labeled, test_labeled, 
            feature_list, model_name
        )
        
        # Plot training curves
        plot_training_history(history, save_path=config.DATA_DIR / f"{model_name}_history.png")
        
        # Phase 5: Evaluation
        test_acc = evaluate_model(trainer, test_loader)
        
        # Save final summary
        summary = {
            'model_name': model_name,
            'feature_set': args.feature_set,
            'n_features': len(feature_list),
            'best_val_acc': trainer.best_val_acc,
            'test_acc': test_acc,
            'device': str(trainer.device)
        }
        
        summary_path = config.CHECKPOINT_DIR / model_name / 'summary.txt'
        with open(summary_path, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\nSaved summary: {summary_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Market Regime Classification')
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'data', 'features', 'label', 'train', 'evaluate'],
                       help='Execution mode: all (full pipeline) or individual phases')
    
    parser.add_argument('--feature_set', type=str, default='baseline',
                       choices=['baseline', 'engineered'],
                       help='Feature set to use: baseline or engineered')
    
    args = parser.parse_args()
    
    main(args)