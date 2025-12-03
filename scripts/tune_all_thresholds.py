"""Tune Thresholds: Optimize decision threshold for all trained models"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

import config
from src.dataset import create_dataloaders
from src.model import create_model


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
            
            all_probs.extend(probs[:, 0].cpu().numpy())
            all_labels.extend(y.numpy())
    
    return np.array(all_probs), np.array(all_labels)


def find_optimal_threshold(probs_bearish, labels):
    """Find optimal threshold that maximizes MCC"""
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = []
    
    for threshold in thresholds:
        predictions = (~(probs_bearish > threshold)).astype(int)
        score = matthews_corrcoef(labels, predictions)
        scores.append(score)
    
    scores = np.array(scores)
    best_idx = scores.argmax()
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return optimal_threshold, best_score


def main():
    print("\n" + "="*70)
    print("TUNING THRESHOLDS FOR ALL MODELS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    available_models = config.list_available_models()
    
    results = []
    
    for i, model_id in enumerate(available_models, 1):
        print(f"\n[{i}/{len(available_models)}] Processing: {model_id}")
        print("-"*70)
        
        checkpoint_path = config.CHECKPOINT_DIR / f"{model_id}_binary_focal" / "best.pth"
        if not checkpoint_path.exists():
            print(f"  ✗ Checkpoint not found, skipping...")
            continue
        
        try:
            model_config = config.get_model_config(model_id)
            model, _ = create_model(model_id, device)
            
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
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
            
            batch_size = config.BATCH_SIZE_GPU if device.type == 'cuda' else config.BATCH_SIZE_CPU
            _, val_loader, test_loader, _ = create_dataloaders(
                val_data, val_data, test_data, feature_list, batch_size
            )
            
            val_probs, val_labels = get_predictions_and_probabilities(model, val_loader, device)
            test_probs, test_labels = get_predictions_and_probabilities(model, test_loader, device)
            
            optimal_threshold, val_mcc = find_optimal_threshold(val_probs, val_labels)
            
            test_predictions = (~(test_probs > optimal_threshold)).astype(int)
            test_mcc_tuned = matthews_corrcoef(test_labels, test_predictions)
            
            test_predictions_default = (~(test_probs > 0.5)).astype(int)
            test_mcc_default = matthews_corrcoef(test_labels, test_predictions_default)
            
            print(f"  Optimal Threshold: {optimal_threshold:.3f}")
            print(f"  Val MCC: {val_mcc:.4f}")
            print(f"  Test MCC (default 0.5): {test_mcc_default:.4f}")
            print(f"  Test MCC (tuned {optimal_threshold:.3f}): {test_mcc_tuned:.4f}")
            print(f"  Improvement: {test_mcc_tuned - test_mcc_default:+.4f}")
            
            results.append({
                'model_id': model_id,
                'model_name': model_config['name'],
                'optimal_threshold': optimal_threshold,
                'val_mcc': val_mcc,
                'test_mcc_default': test_mcc_default,
                'test_mcc_tuned': test_mcc_tuned,
                'improvement': test_mcc_tuned - test_mcc_default
            })
            
            threshold_df = pd.DataFrame([{
                'model_id': model_id,
                'optimal_threshold': optimal_threshold,
                'val_mcc': val_mcc,
                'test_mcc_tuned': test_mcc_tuned
            }])
            threshold_df.to_csv(
                config.DATA_DIR / f"threshold_tuning_{model_id}.csv",
                index=False
            )
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('test_mcc_tuned', ascending=False)
        results_df.to_csv(config.DATA_DIR / "threshold_tuning_all_models.csv", index=False)
        
        print("\n" + "="*70)
        print("THRESHOLD TUNING SUMMARY")
        print("="*70)
        print("\n" + results_df.to_string(index=False))
        
        best = results_df.iloc[0]
        print("\n" + "="*70)
        print("BEST MODEL (After Threshold Tuning)")
        print("="*70)
        print(f"Model: {best['model_name']}")
        print(f"Threshold: {best['optimal_threshold']:.3f}")
        print(f"Test MCC: {best['test_mcc_tuned']:.4f}")
        print(f"Improvement: {best['improvement']:+.4f}")
    
    print("\n" + "="*70)
    print("THRESHOLD TUNING COMPLETE")
    print("="*70)
    print("\nNext step: python scripts/compare_binary_models.py")
    print("\n")


if __name__ == "__main__":
    main()