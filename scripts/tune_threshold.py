"""Threshold Tuning: Optimize decision threshold for better MCC"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, f1_score
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


def get_predictions_and_probabilities(model, data_loader, device):
    """Get predictions and probabilities from model"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            outputs = model(x)
            outputs_binary = outputs[:, :2]
            probs = torch.softmax(outputs_binary, dim=1)
            
            all_probs.extend(probs[:, 0].cpu().numpy())  # Probability of Bearish (class 0)
            all_labels.extend(y.numpy())
    
    return np.array(all_probs), np.array(all_labels)


def find_optimal_threshold(probs_bearish, labels, metric='mcc'):
    """
    Find optimal threshold that maximizes given metric
    
    Args:
        probs_bearish: Probability of Bearish class
        labels: True labels (0=Bearish, 1=Bullish)
        metric: 'mcc' or 'f1'
    
    Returns:
        optimal_threshold, best_score, threshold_scores
    """
    thresholds = np.linspace(0.1, 0.9, 81)  # Test 81 thresholds
    scores = []
    
    for threshold in thresholds:
        # Predict Bearish if prob > threshold
        predictions = (probs_bearish > threshold).astype(int)
        
        if metric == 'mcc':
            score = matthews_corrcoef(labels, predictions)
        elif metric == 'f1':
            score = f1_score(labels, predictions, average='macro')
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    scores = np.array(scores)
    best_idx = scores.argmax()
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return optimal_threshold, best_score, thresholds, scores


def evaluate_with_threshold(probs_bearish, labels, threshold):
    """Evaluate predictions with given threshold"""
    from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    predictions = (probs_bearish > threshold).astype(int)
    
    mcc = matthews_corrcoef(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    balanced_acc = balanced_accuracy_score(labels, predictions)
    
    precision_bearish = precision_score(labels, predictions, pos_label=0, zero_division=0)
    recall_bearish = recall_score(labels, predictions, pos_label=0, zero_division=0)
    f1_bearish = f1_score(labels, predictions, pos_label=0, zero_division=0)
    
    precision_bullish = precision_score(labels, predictions, pos_label=1, zero_division=0)
    recall_bullish = recall_score(labels, predictions, pos_label=1, zero_division=0)
    f1_bullish = f1_score(labels, predictions, pos_label=1, zero_division=0)
    
    conf_matrix = confusion_matrix(labels, predictions)
    
    return {
        'mcc': mcc,
        'f1_macro': f1_macro,
        'balanced_accuracy': balanced_acc,
        'precision_bearish': precision_bearish,
        'recall_bearish': recall_bearish,
        'f1_bearish': f1_bearish,
        'precision_bullish': precision_bullish,
        'recall_bullish': recall_bullish,
        'f1_bullish': f1_bullish,
        'confusion_matrix': conf_matrix,
        'predictions': predictions
    }


def plot_threshold_analysis(thresholds, scores, optimal_threshold, best_score, save_path=None):
    """Plot threshold vs MCC curve"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(thresholds, scores, linewidth=2.5, color='steelblue', label='MCC Score')
    ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
              label=f'Optimal Threshold: {optimal_threshold:.3f}')
    ax.axhline(best_score, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
              label=f'Best MCC: {best_score:.3f}')
    
    ax.scatter([optimal_threshold], [best_score], color='red', s=200, zorder=5,
              edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Matthews Correlation Coefficient (MCC)', fontsize=12, fontweight='bold')
    ax.set_title('Threshold Optimization - MCC vs Decision Threshold', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    ax.set_xlim(0.1, 0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved threshold plot: {save_path}")
        if IN_COLAB:
            display(Image(save_path))
    
    if not IN_COLAB:
        plt.show()
    plt.close()


def main():
    """Main threshold tuning workflow"""
    
    print("\n" + "="*70)
    print("THRESHOLD TUNING FOR BINARY CLASSIFICATION")
    print("="*70)
    
    # Load best focal loss model
    best_model_id = 'model_6_lstm_transformer'  # Change this based on your best model
    
    print(f"\nModel: {best_model_id}")
    print("Finding optimal decision threshold...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = create_model(best_model_id, device)
    
    checkpoint_path = config.CHECKPOINT_DIR / f"{best_model_id}_binary_focal" / "best.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint with val MCC: {checkpoint.get('val_mcc', 'N/A')}")
    
    # Load data
    model_config = config.get_model_config(best_model_id)
    feature_list = model_config['features']
    feature_set = 'engineered' if feature_list == config.ENGINEERED_FEATURES else 'baseline'
    
    val_data = pd.read_csv(
        config.REGIME_DATA_DIR / f"val_labeled_{feature_set}_binary.csv",
        index_col=0, parse_dates=True
    )
    test_data = pd.read_csv(
        config.REGIME_DATA_DIR / f"test_labeled_{feature_set}_binary.csv",
        index_col=0, parse_dates=True
    )
    
    # Create dataloaders
    batch_size = config.BATCH_SIZE_GPU if device.type == 'cuda' else config.BATCH_SIZE_CPU
    _, val_loader, test_loader, _ = create_dataloaders(
        val_data, val_data, test_data, feature_list, batch_size
    )
    
    print("\nData loaded:")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Get predictions
    print("\nGenerating predictions...")
    val_probs, val_labels = get_predictions_and_probabilities(model, val_loader, device)
    test_probs, test_labels = get_predictions_and_probabilities(model, test_loader, device)
    
    # Find optimal threshold on validation set
    print("\n" + "="*70)
    print("TUNING THRESHOLD ON VALIDATION SET")
    print("="*70)
    
    optimal_threshold, best_val_mcc, thresholds, scores = find_optimal_threshold(
        val_probs, val_labels, metric='mcc'
    )
    
    print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
    print(f"Validation MCC at optimal threshold: {best_val_mcc:.3f}")
    
    # Evaluate on validation with default and optimal thresholds
    print("\n" + "="*70)
    print("VALIDATION SET PERFORMANCE")
    print("="*70)
    
    print("\nDefault Threshold (0.500):")
    val_results_default = evaluate_with_threshold(val_probs, val_labels, 0.5)
    print(f"  MCC: {val_results_default['mcc']:.3f}")
    print(f"  F1-Score: {val_results_default['f1_macro']:.3f}")
    print(f"  Balanced Acc: {val_results_default['balanced_accuracy']:.3f}")
    
    print(f"\nOptimal Threshold ({optimal_threshold:.3f}):")
    val_results_optimal = evaluate_with_threshold(val_probs, val_labels, optimal_threshold)
    print(f"  MCC: {val_results_optimal['mcc']:.3f}")
    print(f"  F1-Score: {val_results_optimal['f1_macro']:.3f}")
    print(f"  Balanced Acc: {val_results_optimal['balanced_accuracy']:.3f}")
    
    print(f"\nImprovement: {val_results_optimal['mcc'] - val_results_default['mcc']:+.3f} MCC")
    
    # Apply to test set
    print("\n" + "="*70)
    print("TEST SET PERFORMANCE")
    print("="*70)
    
    print("\nDefault Threshold (0.500):")
    test_results_default = evaluate_with_threshold(test_probs, test_labels, 0.5)
    print(f"  MCC: {test_results_default['mcc']:.3f}")
    print(f"  F1-Score: {test_results_default['f1_macro']:.3f}")
    print(f"  Balanced Acc: {test_results_default['balanced_accuracy']:.3f}")
    print(f"  Bearish F1: {test_results_default['f1_bearish']:.3f}")
    print(f"  Bullish F1: {test_results_default['f1_bullish']:.3f}")
    
    print(f"\nOptimal Threshold ({optimal_threshold:.3f}):")
    test_results_optimal = evaluate_with_threshold(test_probs, test_labels, optimal_threshold)
    print(f"  MCC: {test_results_optimal['mcc']:.3f}")
    print(f"  F1-Score: {test_results_optimal['f1_macro']:.3f}")
    print(f"  Balanced Acc: {test_results_optimal['balanced_accuracy']:.3f}")
    print(f"  Bearish F1: {test_results_optimal['f1_bearish']:.3f}")
    print(f"  Bullish F1: {test_results_optimal['f1_bullish']:.3f}")
    
    print(f"\n{'='*70}")
    print(f"TEST IMPROVEMENT: {test_results_optimal['mcc'] - test_results_default['mcc']:+.3f} MCC")
    print(f"{'='*70}")
    
    # Plot threshold analysis
    print("\nGenerating threshold analysis plot...")
    plot_threshold_analysis(
        thresholds, scores, optimal_threshold, best_val_mcc,
        save_path=config.DATA_DIR / "threshold_optimization.png"
    )
    
    # Save results
    results_summary = {
        'model_id': best_model_id,
        'optimal_threshold': optimal_threshold,
        'val_mcc_default': val_results_default['mcc'],
        'val_mcc_optimal': val_results_optimal['mcc'],
        'test_mcc_default': test_results_default['mcc'],
        'test_mcc_optimal': test_results_optimal['mcc'],
        'improvement': test_results_optimal['mcc'] - test_results_default['mcc']
    }
    
    results_df = pd.DataFrame([results_summary])
    results_df.to_csv(config.DATA_DIR / "threshold_tuning_results.csv", index=False)
    print(f"\nResults saved to: threshold_tuning_results.csv")
    
    print("\n" + "="*70)
    print("THRESHOLD TUNING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
