"""Compare SMOTE vs Non-SMOTE Model Performance"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix

import config
from src.dataset import create_dataloaders
from src.model import create_model

# Try to import Colab display
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False


def evaluate_model(model_path, model_id, test_loader, device):
    """Evaluate a single model"""
    if not model_path.exists():
        return None
    
    model, _ = create_model(model_id, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)[:, :2]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
    
    mcc = matthews_corrcoef(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    f1_bearish = f1_score(all_labels, all_preds, pos_label=0, zero_division=0)
    f1_bullish = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    
    precision_bearish = precision_score(all_labels, all_preds, pos_label=0, zero_division=0)
    recall_bearish = recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    return {
        'mcc': mcc,
        'f1_macro': f1_macro,
        'balanced_accuracy': balanced_acc,
        'f1_bearish': f1_bearish,
        'f1_bullish': f1_bullish,
        'precision_bearish': precision_bearish,
        'recall_bearish': recall_bearish,
        'confusion_matrix': conf_mat,
        'val_mcc': checkpoint.get('val_mcc', 'N/A')
    }


def main():
    print("\n" + "="*70)
    print("SMOTE VS NON-SMOTE COMPARISON")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Models to compare
    models_to_compare = config.list_available_models()
    
    print(f"\nComparing {len(models_to_compare)} models...")
    
    results = []
    
    for model_id in models_to_compare:
        model_config = config.get_model_config(model_id)
        feature_list = model_config['features']
        feature_set = 'engineered' if feature_list == config.ENGINEERED_FEATURES else 'baseline'
        
        # Load test data
        test_data = pd.read_csv(
            config.REGIME_DATA_DIR / f"test_labeled_{feature_set}_binary.csv",
            index_col=0, parse_dates=True
        )
        
        _, _, test_loader, _ = create_dataloaders(
            test_data, test_data, test_data, feature_list,
            batch_size=config.BATCH_SIZE_GPU if device.type == 'cuda' else config.BATCH_SIZE_CPU
        )
        
        # Evaluate non-SMOTE model
        focal_path = config.CHECKPOINT_DIR / f"{model_id}_binary_focal" / "best.pth"
        focal_results = evaluate_model(focal_path, model_id, test_loader, device)
        
        # Evaluate SMOTE model
        smote_path = config.CHECKPOINT_DIR / f"{model_id}_binary_smote" / "best.pth"
        smote_results = evaluate_model(smote_path, model_id, test_loader, device)
        
        if focal_results is not None or smote_results is not None:
            results.append({
                'model_id': model_id,
                'model_name': model_config['name'],
                'focal': focal_results,
                'smote': smote_results
            })
    
    # Create comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    table_data = []
    
    for result in results:
        focal = result['focal']
        smote = result['smote']
        
        row = {
            'Model': result['model_name'],
        }
        
        if focal:
            row['Focal MCC'] = focal['mcc']
            row['Focal F1'] = focal['f1_macro']
            row['Focal Bearish F1'] = focal['f1_bearish']
        else:
            row['Focal MCC'] = np.nan
            row['Focal F1'] = np.nan
            row['Focal Bearish F1'] = np.nan
        
        if smote:
            row['SMOTE MCC'] = smote['mcc']
            row['SMOTE F1'] = smote['f1_macro']
            row['SMOTE Bearish F1'] = smote['f1_bearish']
        else:
            row['SMOTE MCC'] = np.nan
            row['SMOTE F1'] = np.nan
            row['SMOTE Bearish F1'] = np.nan
        
        if focal and smote:
            row['MCC Δ'] = smote['mcc'] - focal['mcc']
            row['F1 Δ'] = smote['f1_macro'] - focal['f1_macro']
        else:
            row['MCC Δ'] = np.nan
            row['F1 Δ'] = np.nan
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    df = df.sort_values('SMOTE MCC', ascending=False)
    
    print("\n" + df.to_string(index=False, float_format='%.3f'))
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    valid_comparisons = df.dropna(subset=['MCC Δ'])
    
    if len(valid_comparisons) > 0:
        avg_mcc_improvement = valid_comparisons['MCC Δ'].mean()
        avg_f1_improvement = valid_comparisons['F1 Δ'].mean()
        
        improved_count = (valid_comparisons['MCC Δ'] > 0).sum()
        worse_count = (valid_comparisons['MCC Δ'] < 0).sum()
        
        print(f"\nModels compared: {len(valid_comparisons)}")
        print(f"  Improved with SMOTE: {improved_count}")
        print(f"  Worse with SMOTE:    {worse_count}")
        print(f"\nAverage MCC improvement: {avg_mcc_improvement:+.3f}")
        print(f"Average F1 improvement:  {avg_f1_improvement:+.3f}")
        
        if avg_mcc_improvement > 0.05:
            print("\n✅ SMOTE shows significant improvement! Consider merging.")
        elif avg_mcc_improvement > 0.0:
            print("\n⚠️  SMOTE shows modest improvement. Review case-by-case.")
        else:
            print("\n❌ SMOTE does not improve performance. Do not merge.")
    
    # Save results
    df.to_csv(config.DATA_DIR / "smote_comparison_results.csv", index=False)
    print(f"\nResults saved to: {config.DATA_DIR / 'smote_comparison_results.csv'}")
    
    # Plot comparison
    if len(valid_comparisons) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # MCC comparison
        ax1 = axes[0]
        x = np.arange(len(valid_comparisons))
        width = 0.35
        
        ax1.bar(x - width/2, valid_comparisons['Focal MCC'], width, label='Focal Loss', color='steelblue')
        ax1.bar(x + width/2, valid_comparisons['SMOTE MCC'], width, label='SMOTE', color='coral')
        
        ax1.set_xlabel('Model', fontweight='bold')
        ax1.set_ylabel('MCC', fontweight='bold')
        ax1.set_title('MCC Comparison: Focal vs SMOTE', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_comparisons['Model'], rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Improvement plot
        ax2 = axes[1]
        colors = ['green' if d > 0 else 'red' for d in valid_comparisons['MCC Δ']]
        ax2.barh(range(len(valid_comparisons)), valid_comparisons['MCC Δ'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(valid_comparisons)))
        ax2.set_yticklabels(valid_comparisons['Model'], fontsize=9)
        ax2.set_xlabel('MCC Improvement (SMOTE - Focal)', fontweight='bold')
        ax2.set_title('MCC Improvement with SMOTE', fontweight='bold')
        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = config.DATA_DIR / "smote_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {plot_path}")
        
        if IN_COLAB:
            display(Image(filename=str(plot_path)))
        else:
            plt.show()
        plt.close()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
