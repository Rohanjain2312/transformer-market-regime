"""Ensemble Models: Combine top models with tuned thresholds for better predictions"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import config
from src.dataset import create_dataloaders
from src.model import create_model

# Try to import Colab display
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False


def get_model_predictions(model_id, data_loader, device):
    """Get probability predictions from a single model"""
    
    # Load model
    model, _ = create_model(model_id, device)
    checkpoint_path = config.CHECKPOINT_DIR / f"{model_id}_binary_focal" / "best.pth"
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found for {model_id}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            outputs = model(x)
            outputs_binary = outputs[:, :2]
            probs = torch.softmax(outputs_binary, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    
    return all_probs, all_labels


def ensemble_predictions(prob_list, weights=None, method='weighted_average'):
    """
    Combine predictions from multiple models
    
    Args:
        prob_list: List of probability arrays from each model
        weights: List of weights for each model (must sum to 1)
        method: 'weighted_average', 'voting', or 'max'
    
    Returns:
        ensemble_probs: Combined probability predictions
    """
    n_models = len(prob_list)
    
    if weights is None:
        weights = [1.0 / n_models] * n_models
    
    if method == 'weighted_average':
        # Weighted average of probabilities
        ensemble_probs = np.zeros_like(prob_list[0])
        for probs, weight in zip(prob_list, weights):
            ensemble_probs += probs * weight
    
    elif method == 'voting':
        # Hard voting (predict class then vote)
        votes = np.zeros((len(prob_list[0]), 2))
        for probs, weight in zip(prob_list, weights):
            preds = probs.argmax(axis=1)
            for i, pred in enumerate(preds):
                votes[i, pred] += weight
        
        # Convert votes back to probabilities
        ensemble_probs = votes / votes.sum(axis=1, keepdims=True)
    
    elif method == 'max':
        # Take maximum probability for each class
        ensemble_probs = np.maximum.reduce(prob_list)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_probs


def evaluate_ensemble(probs, labels, threshold=0.5):
    """Evaluate ensemble predictions"""
    
    # Predict Bearish (0) if prob_bearish > threshold, else Bullish (1)
    predictions = (~(probs[:, 0] > threshold)).astype(int)
    
    mcc = matthews_corrcoef(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    balanced_acc = balanced_accuracy_score(labels, predictions)
    
    f1_bearish = f1_score(labels, predictions, pos_label=0, zero_division=0)
    f1_bullish = f1_score(labels, predictions, pos_label=1, zero_division=0)
    
    conf_matrix = confusion_matrix(labels, predictions)
    
    return {
        'mcc': mcc,
        'f1_macro': f1_macro,
        'balanced_accuracy': balanced_acc,
        'f1_bearish': f1_bearish,
        'f1_bullish': f1_bullish,
        'confusion_matrix': conf_matrix,
        'predictions': predictions
    }


def find_optimal_ensemble_threshold(probs, labels):
    """Find optimal threshold for ensemble"""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_mcc = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (~(probs[:, 0] > threshold)).astype(int)
        mcc = matthews_corrcoef(labels, predictions)
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    
    return best_threshold, best_mcc


def plot_ensemble_comparison(results_dict, save_path=None):
    """Plot comparison of individual models vs ensemble"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MCC comparison
    ax1 = axes[0]
    models = list(results_dict.keys())
    mccs = [results_dict[m]['mcc'] for m in models]
    colors = ['steelblue'] * (len(models) - 1) + ['darkgreen']
    
    bars = ax1.bar(range(len(models)), mccs, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('MCC', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison - Matthews Correlation Coefficient', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, mcc in zip(bars, mccs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mcc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score comparison
    ax2 = axes[1]
    f1_bearish = [results_dict[m]['f1_bearish'] for m in models]
    f1_bullish = [results_dict[m]['f1_bullish'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, f1_bearish, width, label='Bearish', color='red', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, f1_bullish, width, label='Bullish', color='green', alpha=0.7, edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class F1-Score Comparison', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {save_path}")
        if IN_COLAB:
            display(Image(save_path))
    
    if not IN_COLAB:
        plt.show()
    plt.close()


def main():
    """Main ensemble workflow"""
    
    print("\n" + "="*70)
    print("ENSEMBLE MODEL PREDICTIONS")
    print("="*70)
    
    # Define ensemble configuration - weighted toward best model
    ensemble_config = {
        'model_1_engineered': {
            'weight': 0.70,
            'threshold': 0.670,
            'name': 'Engineered'
        },
        'model_2_large_capacity': {
            'weight': 0.20,
            'threshold': 0.600,
            'name': 'Large Capacity'
        },
        'model_5_multiscale': {
            'weight': 0.10,
            'threshold': 0.620,
            'name': 'Multi-Scale'
        }
    }
    
    model_ids = list(ensemble_config.keys())
    weights = [ensemble_config[m]['weight'] for m in model_ids]
    
    print("\nEnsemble Configuration:")
    for model_id, config_data in ensemble_config.items():
        print(f"  {config_data['name']}: weight={config_data['weight']}, threshold={config_data['threshold']}")
    
    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load validation and test data (use engineered features)
    val_data = pd.read_csv(
        config.REGIME_DATA_DIR / "val_labeled_engineered_binary.csv",
        index_col=0, parse_dates=True
    )
    test_data = pd.read_csv(
        config.REGIME_DATA_DIR / "test_labeled_engineered_binary.csv",
        index_col=0, parse_dates=True
    )
    
    # Create dataloaders
    batch_size = config.BATCH_SIZE_GPU if device.type == 'cuda' else config.BATCH_SIZE_CPU
    _, val_loader, test_loader, _ = create_dataloaders(
        val_data, val_data, test_data, config.ENGINEERED_FEATURES, batch_size
    )
    
    print(f"\nData loaded:")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Get predictions from each model
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS FROM INDIVIDUAL MODELS")
    print("="*70)
    
    val_prob_list = []
    test_prob_list = []
    
    for model_id in model_ids:
        print(f"\nLoading {ensemble_config[model_id]['name']}...")
        
        val_probs, val_labels = get_model_predictions(model_id, val_loader, device)
        test_probs, test_labels = get_model_predictions(model_id, test_loader, device)
        
        if val_probs is None or test_probs is None:
            print(f"Skipping {model_id}")
            continue
        
        val_prob_list.append(val_probs)
        test_prob_list.append(test_probs)
    
    # Create ensemble predictions
    print("\n" + "="*70)
    print("CREATING ENSEMBLE PREDICTIONS")
    print("="*70)
    
    val_ensemble_probs = ensemble_predictions(val_prob_list, weights=weights, method='weighted_average')
    test_ensemble_probs = ensemble_predictions(test_prob_list, weights=weights, method='weighted_average')
    
    # Find optimal threshold on validation set
    print("\nFinding optimal ensemble threshold on validation set...")
    optimal_threshold, best_val_mcc = find_optimal_ensemble_threshold(val_ensemble_probs, val_labels)
    
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    print(f"  Validation MCC: {best_val_mcc:.3f}")
    
    # Evaluate ensemble on test set
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    
    # Evaluate individual models with their tuned thresholds
    results = {}
    
    for i, model_id in enumerate(model_ids):
        model_name = ensemble_config[model_id]['name']
        threshold = ensemble_config[model_id]['threshold']
        
        result = evaluate_ensemble(test_prob_list[i], test_labels, threshold)
        results[model_name] = result
        
        print(f"\n{model_name} (threshold={threshold:.3f}):")
        print(f"  MCC: {result['mcc']:.3f}")
        print(f"  F1-Score: {result['f1_macro']:.3f}")
        print(f"  Bearish F1: {result['f1_bearish']:.3f}")
        print(f"  Bullish F1: {result['f1_bullish']:.3f}")
    
    # Evaluate ensemble
    ensemble_result = evaluate_ensemble(test_ensemble_probs, test_labels, optimal_threshold)
    results['Ensemble'] = ensemble_result
    
    print(f"\nEnsemble (threshold={optimal_threshold:.3f}):")
    print(f"  MCC: {ensemble_result['mcc']:.3f}")
    print(f"  F1-Score: {ensemble_result['f1_macro']:.3f}")
    print(f"  Balanced Acc: {ensemble_result['balanced_accuracy']:.3f}")
    print(f"  Bearish F1: {ensemble_result['f1_bearish']:.3f}")
    print(f"  Bullish F1: {ensemble_result['f1_bullish']:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    best_individual_mcc = max([results[ensemble_config[m]['name']]['mcc'] for m in model_ids])
    improvement = ensemble_result['mcc'] - best_individual_mcc
    
    print(f"\nBest Individual Model MCC: {best_individual_mcc:.3f}")
    print(f"Ensemble MCC: {ensemble_result['mcc']:.3f}")
    print(f"Improvement: {improvement:+.3f} ({improvement/best_individual_mcc*100:+.1f}%)")
    
    # Plot comparison
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOT")
    print("="*70)
    
    plot_ensemble_comparison(
        results,
        save_path=config.DATA_DIR / "ensemble_comparison.png"
    )
    
    # Save results
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'MCC': result['mcc'],
            'F1-Score': result['f1_macro'],
            'Balanced Acc': result['balanced_accuracy'],
            'Bearish F1': result['f1_bearish'],
            'Bullish F1': result['f1_bullish']
        })
    
    results_df = pd.DataFrame(summary_data)
    results_df.to_csv(config.DATA_DIR / "ensemble_results.csv", index=False)
    print(f"\nResults saved to: {config.DATA_DIR / 'ensemble_results.csv'}")
    
    print("\n" + "="*70)
    print("ENSEMBLE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()