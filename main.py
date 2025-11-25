"""Main Orchestrator: Consolidated pipeline for training multiple models"""

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

# Try to import Colab display (will fail if not in Colab)
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False

def display_image_if_colab(image_path):
    """Display image in Colab if available"""
    if IN_COLAB and Path(image_path).exists():
        display(Image(image_path))

def run_data_pipeline():
    """Phase 1: Data acquisition and preprocessing"""
    print("\n" + "="*70)
    print("PHASE 1: DATA PIPELINE")
    print("="*70)
    
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()
    
    return data['train'], data['val'], data['test']

def run_feature_engineering(train, val, test):
    """Phase 2: Create BOTH baseline and engineered features"""
    print("\n" + "="*70)
    print("PHASE 2: FEATURE ENGINEERING (BOTH SETS)")
    print("="*70)
    
    engineer = FeatureEngineer()
    
    # Create baseline features
    print("\nCreating BASELINE features...")
    train_base, val_base, test_base, _ = engineer.process_dataset(
        train, val, test, 'baseline'
    )
    engineer.save_features(train_base, val_base, test_base, 'baseline')
    
    # Create engineered features
    print("\nCreating ENGINEERED features...")
    train_eng, val_eng, test_eng, _ = engineer.process_dataset(
        train, val, test, 'engineered'
    )
    engineer.save_features(train_eng, val_eng, test_eng, 'engineered')
    
    # Display sample data
    print("\n" + "="*70)
    print("SAMPLE DATA (First 50 rows of ENGINEERED features)")
    print("="*70)
    print(train_eng.head(50))
    
    return {
        'baseline': (train_base, val_base, test_base),
        'engineered': (train_eng, val_eng, test_eng)
    }

def run_regime_labeling(feature_data):
    """Phase 3: Label regimes for BOTH feature sets"""
    print("\n" + "="*70)
    print("PHASE 3: REGIME LABELING (BOTH SETS)")
    print("="*70)
    
    labeler = RegimeLabeler()
    labeled_data = {}
    
    for feature_set, (train, val, test) in feature_data.items():
        print(f"\nLabeling {feature_set.upper()} data...")
        train_labeled, val_labeled, test_labeled = labeler.label_dataset(train, val, test)
        labeler.save_labeled_data(train_labeled, val_labeled, test_labeled, feature_set)
        labeled_data[feature_set] = (train_labeled, val_labeled, test_labeled)
    
    return labeled_data

def train_single_model(model_id, labeled_data):
    """Phase 4: Train a single model"""
    print("\n" + "="*70)
    print(f"PHASE 4: TRAINING {model_id.upper()}")
    print("="*70)
    
    # Get model config
    model_config = config.get_model_config(model_id)
    feature_list = model_config['features']
    
    # Determine which feature set to use
    feature_set = 'engineered' if feature_list == config.ENGINEERED_FEATURES else 'baseline'
    train, val, test = labeled_data[feature_set]
    
    print(f"\nUsing {feature_set.upper()} features ({len(feature_list)} features)")
    
    # Create dataloaders
    print("Creating dataloaders...")
    batch_size = config.BATCH_SIZE_GPU if torch.cuda.is_available() else config.BATCH_SIZE_CPU
    train_loader, val_loader, test_loader, n_features = create_dataloaders(
        train, val, test, feature_list, batch_size
    )
    
    # Create model
    print(f"Initializing {model_config['name']}...")
    model, device = create_model(model_id)
    
    # Reduce dataset if CPU
    if device.type == 'cpu':
        train_loader, val_loader = reduce_dataset_for_cpu(train_loader, val_loader)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, device, model_name=model_id)
    history = trainer.train()
    
    # Plot and save training curves
    plot_path = config.DATA_DIR / f"{model_id}_history.png"
    plot_training_history(history, model_config['name'], save_path=plot_path)
    display_image_if_colab(plot_path)
    
    # Evaluate on test set
    test_acc = evaluate_model(trainer, test_loader)
    
    # Save summary
    save_model_summary(model_id, model_config, trainer, test_acc)
    
    return trainer, history, test_acc

def evaluate_model(trainer, test_loader):
    """Evaluate model on test set"""
    print("\n" + "="*70)
    print("FINAL EVALUATION")
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
    
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=config.REGIME_NAMES,
                                digits=3))
    
    return test_acc

def plot_training_history(history, model_name, save_path=None):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved training curves: {save_path}")
    
    if not IN_COLAB:
        plt.show()
    plt.close()

def save_model_summary(model_id, model_config, trainer, test_acc):
    """Save model summary to file"""
    summary_dir = config.CHECKPOINT_DIR / model_id
    summary_path = summary_dir / 'summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write(f"Model: {model_config['name']}\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Architecture: {model_config['architecture']}\n")
        f.write(f"Features: {len(model_config['features'])}\n")
        f.write(f"Parameters: {trainer.model.get_num_parameters():,}\n")
        f.write(f"Best Val Acc: {trainer.best_val_acc:.2f}%\n")
        f.write(f"Test Acc: {test_acc:.2f}%\n")
        f.write(f"Device: {trainer.device}\n")
    
    print(f"\nSaved summary: {summary_path}")

def main(args):
    """Main execution pipeline"""
    
    print("\n" + "="*70)
    print("TRANSFORMER MARKET REGIME CLASSIFICATION")
    print("="*70)
    print(f"\nMode: {args.mode}")
    print(f"Models to train: {args.models if args.models != 'all' else 'ALL'}")
    
    # Phase 1-3: Run once regardless of number of models
    if args.mode in ['all', 'prepare']:
        # Phase 1: Data Pipeline
        train, val, test = run_data_pipeline()
        
        # Phase 2: Feature Engineering (both sets)
        feature_data = run_feature_engineering(train, val, test)
        
        # Phase 3: Regime Labeling (both sets)
        labeled_data = run_regime_labeling(feature_data)
        
        if args.mode == 'prepare':
            print("\n" + "="*70)
            print("DATA PREPARATION COMPLETE")
            print("="*70 + "\n")
            return
    
    # Load labeled data if starting from training
    if args.mode == 'train':
        print("\nLoading labeled data...")
        labeled_data = {}
        for feature_set in ['baseline', 'engineered']:
            suffix = f"_{feature_set}"
            train = pd.read_csv(config.REGIME_DATA_DIR / f"train_labeled{suffix}.csv",
                               index_col=0, parse_dates=True)
            val = pd.read_csv(config.REGIME_DATA_DIR / f"val_labeled{suffix}.csv",
                             index_col=0, parse_dates=True)
            test = pd.read_csv(config.REGIME_DATA_DIR / f"test_labeled{suffix}.csv",
                              index_col=0, parse_dates=True)
            labeled_data[feature_set] = (train, val, test)
    
    # Phase 4: Training
    if args.mode in ['all', 'train']:
        # Determine which models to train
        if args.models == 'all':
            models_to_train = config.list_available_models()
        else:
            models_to_train = [m.strip() for m in args.models.split(',')]
        
        print(f"\nTraining {len(models_to_train)} model(s)...")
        
        results = {}
        for model_id in models_to_train:
            if model_id not in config.list_available_models():
                print(f"\nWarning: Unknown model '{model_id}', skipping...")
                continue
            
            try:
                trainer, history, test_acc = train_single_model(model_id, labeled_data)
                results[model_id] = {
                    'trainer': trainer,
                    'history': history,
                    'test_acc': test_acc
                }
            except Exception as e:
                print(f"\nError training {model_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary of all trained models
        if len(results) > 1:
            print("\n" + "="*70)
            print("TRAINING SUMMARY - ALL MODELS")
            print("="*70)
            for model_id, result in results.items():
                model_config = config.get_model_config(model_id)
                print(f"\n{model_config['name']} ({model_id}):")
                print(f"    Best Val Acc: {result['trainer'].best_val_acc:.2f}%")
                print(f"    Test Acc:     {result['test_acc']:.2f}%")
            
            # Find best model
            best_model = max(results.items(), key=lambda x: x[1]['test_acc'])
            print(f"\nBest Model: {config.get_model_config(best_model[0])['name']}")
            print(f"Test Accuracy: {best_model[1]['test_acc']:.2f}%")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Market Regime Classification')
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'prepare', 'train'],
                       help='Execution mode: all (full pipeline), prepare (data only), train (training only)')
    
    parser.add_argument('--models', type=str, default='all',
                       help='Models to train: "all" or comma-separated list (e.g., "model_0_baseline,model_2_large_capacity")')
    
    args = parser.parse_args()
    
    main(args)