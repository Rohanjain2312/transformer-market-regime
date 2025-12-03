"""Tune classification thresholds for all trained models"""

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import config
from src.dataset import create_dataloaders
from src.model import create_model

def get_predictions_and_probabilities(model, dataloader, device):
    """Get model predictions and probabilities"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs = model(x)
            outputs_binary = outputs[:, :2]
            probs = torch.softmax(outputs_binary, dim=1)[:, 1]  # Probability of Bullish
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    return np.array(all_probs), np.array(all_labels)

def find_optimal_threshold(probs, labels):
    """Find threshold that maximizes MCC on validation set"""
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_threshold = 0.5
    best_mcc = -1
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        try:
            mcc = matthews_corrcoef(labels, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
        except:
            continue
    
    return best_threshold, best_mcc

def evaluate_with_threshold(probs, labels, threshold):
    """Evaluate predictions with given threshold"""
    preds = (probs >= threshold).astype(int)
    mcc = matthews_corrcoef(labels, preds)
    accuracy = (preds == labels).mean() * 100
    return mcc, accuracy

def tune_model_threshold(model_id):
    """Tune threshold for a single model"""
    checkpoint_path = config.CHECKPOINT_DIR / model_id / "best.pth"
    
    if not checkpoint_path.exists():
        return None
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = create_model(model_id, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    model_config = config.get_model_config(model_id)
    feature_list = model_config['features']
    feature_set = 'engineered' if feature_list == config.ENGINEERED_FEATURES else 'baseline'
    
    val_data = pd.read_csv(
        config.REGIME_DATA_DIR / f"val_labeled_{feature_set}.csv",
        index_col=0, parse_dates=True
    )
    test_data = pd.read_csv(
        config.REGIME_DATA_DIR / f"test_labeled_{feature_set}.csv",
        index_col=0, parse_dates=True
    )
    
    _, val_loader, test_loader, _ = create_dataloaders(
        val_data, val_data, test_data, feature_list
    )
    
    # Get predictions
    val_probs, val_labels = get_predictions_and_probabilities(model, val_loader, device)
    test_probs, test_labels = get_predictions_and_probabilities(model, test_loader, device)
    
    # Find optimal threshold on validation set
    optimal_threshold, val_mcc = find_optimal_threshold(val_probs, val_labels)
    
    # Evaluate on test set with both thresholds
    test_mcc_default, test_acc_default = evaluate_with_threshold(test_probs, test_labels, 0.5)
    test_mcc_tuned, test_acc_tuned = evaluate_with_threshold(test_probs, test_labels, optimal_threshold)
    
    improvement = test_mcc_tuned - test_mcc_default
    
    return {
        'model_id': model_id,
        'model_name': model_config['name'],
        'optimal_threshold': optimal_threshold,
        'val_mcc': val_mcc,
        'test_mcc_default': test_mcc_default,
        'test_acc_default': test_acc_default,
        'test_mcc_tuned': test_mcc_tuned,
        'test_acc_tuned': test_acc_tuned,
        'improvement': improvement
    }

def main():
    print("\n" + "="*70)
    print("TUNING THRESHOLDS FOR ALL MODELS")
    print("="*70)
    
    results = []
    models = config.list_available_models()
    
    for i, model_id in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_id}...", end=" ", flush=True)
        
        result = tune_model_threshold(model_id)
        
        if result is None:
            print("✗ Checkpoint not found")
            continue
        
        results.append(result)
        print(f"✓ Threshold={result['optimal_threshold']:.3f}, MCC={result['test_mcc_tuned']:.3f}")
    
    if not results:
        print("\n✗ No trained models found. Train models first.")
        return
    
    # Create DataFrame and sort by tuned test MCC
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_mcc_tuned', ascending=False)
    
    # Save results
    output_path = config.DATA_DIR / "threshold_tuning_all_models.csv"
    df_results.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("THRESHOLD TUNING SUMMARY")
    print("="*70)
    print("\n" + df_results[['model_id', 'optimal_threshold', 'test_mcc_default', 
                             'test_mcc_tuned', 'improvement']].to_string(index=False))
    
    # Best model
    best = df_results.iloc[0]
    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    print(f"Model:      {best['model_name']}")
    print(f"Threshold:  {best['optimal_threshold']:.3f}")
    print(f"Test MCC:   {best['test_mcc_tuned']:.3f}")
    print(f"Test Acc:   {best['test_acc_tuned']:.2f}%")
    
    print("\n" + "="*70)
    print("✓ THRESHOLD TUNING COMPLETE")
    print("="*70)
    print(f"Results saved: {output_path}")
    print(f"\nNext step: python scripts/compare_models.py")

if __name__ == "__main__":
    main()