"""Compare All Binary Models: Evaluate and visualize performance"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, 
    balanced_accuracy_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score
)

import config
from src.dataset import create_dataloaders
from src.model import create_model

# Check if running in Colab
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False


class BinaryModelComparator:
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_id):
        """Load trained binary model from checkpoint"""
        checkpoint_path = config.CHECKPOINT_DIR / model_id / "best.pth"
        
        if not checkpoint_path.exists():
            return False
        
        model_config = config.get_model_config(model_id)
        model, _ = create_model(model_id, self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.models[model_id] = {
            'model': model,
            'config': model_config,
            'checkpoint': checkpoint
        }
        
        return True
    
    def evaluate_model(self, model_id, test_loader):
        """Evaluate binary model on test set with comprehensive metrics"""
        model = self.models[model_id]['model']
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                outputs = model(x)
                
                # Binary: use first 2 outputs
                outputs_binary = outputs[:, :2]
                probs = torch.softmax(outputs_binary, dim=1)
                preds = outputs_binary.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute comprehensive metrics
        accuracy = (all_preds == all_labels).mean() * 100
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Per-class metrics
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        # Overall metrics
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        # Per-class accuracy
        bearish_mask = all_labels == 0
        bullish_mask = all_labels == 1
        
        bearish_acc = (all_preds[bearish_mask] == all_labels[bearish_mask]).mean() * 100 if bearish_mask.sum() > 0 else 0
        bullish_acc = (all_preds[bullish_mask] == all_labels[bullish_mask]).mean() * 100 if bullish_mask.sum() > 0 else 0
        
        self.results[model_id] = {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'bearish_acc': bearish_acc,
            'bullish_acc': bullish_acc,
            'precision_bearish': precision_per_class[0] * 100,
            'precision_bullish': precision_per_class[1] * 100,
            'recall_bearish': recall_per_class[0] * 100,
            'recall_bullish': recall_per_class[1] * 100,
            'f1_bearish': f1_per_class[0] * 100,
            'f1_bullish': f1_per_class[1] * 100,
            'f1_macro': f1_macro * 100,
            'balanced_accuracy': balanced_acc * 100,
            'mcc': mcc
        }
        
        return accuracy
    
    def create_results_table(self):
        """Create comprehensive results table with all metrics"""
        table_data = []
        
        for model_id in self.results.keys():
            model_info = self.models[model_id]
            config_data = model_info['config']
            checkpoint = model_info['checkpoint']
            result = self.results[model_id]
            
            table_data.append({
                'Model ID': model_id,
                'Name': config_data['name'],
                'Architecture': config_data['architecture'],
                'Parameters': model_info['model'].get_num_parameters(),
                'Val Acc (%)': checkpoint.get('val_acc', 0),
                'Test Acc (%)': result['accuracy'],
                'F1-Score (%)': result['f1_macro'],
                'MCC': result['mcc'],
                'Balanced Acc (%)': result['balanced_accuracy'],
                'F1 Bearish (%)': result['f1_bearish'],
                'F1 Bullish (%)': result['f1_bullish'],
                'Precision Bearish (%)': result['precision_bearish'],
                'Precision Bullish (%)': result['precision_bullish'],
                'Recall Bearish (%)': result['recall_bearish'],
                'Recall Bullish (%)': result['recall_bullish'],
                'Bearish Acc (%)': result['bearish_acc'],
                'Bullish Acc (%)': result['bullish_acc']
            })
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('MCC', ascending=False)
        
        return df
    
    def plot_comparison(self, results_df, save_dir=None):
        """Generate clean comparison visualizations"""
        if save_dir is None:
            save_dir = config.DATA_DIR
        
        n_models = len(results_df)
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        # ============================================================
        # PLOT 1: MCC Comparison (Primary Metric)
        # ============================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(range(n_models), results_df['MCC'], 
                      color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(results_df['Name'], fontsize=10)
        ax.set_xlabel('Matthews Correlation Coefficient (MCC)', fontsize=12, fontweight='bold')
        ax.set_title('Binary Classification - MCC Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(-1, 1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        for i, (bar, mcc) in enumerate(zip(bars, results_df['MCC'])):
            ax.text(mcc + 0.02, i, f'{mcc:.3f}', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plot1_path = save_dir / "plot1_mcc.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {plot1_path.name}")
        
        if IN_COLAB:
            display(Image(plot1_path))
        else:
            plt.show()
        plt.close()
        
        # ============================================================
        # PLOT 2: F1-Score Comparison
        # ============================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(range(n_models), results_df['F1-Score (%)'], 
                      color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(results_df['Name'], fontsize=10)
        ax.set_xlabel('F1-Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Binary Classification - F1-Score Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 100)
        
        for i, (bar, f1) in enumerate(zip(bars, results_df['F1-Score (%)'])):
            ax.text(f1 + 1, i, f'{f1:.1f}%', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plot2_path = save_dir / "plot2_f1_score.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {plot2_path.name}")
        
        if IN_COLAB:
            display(Image(plot2_path))
        else:
            plt.show()
        plt.close()
        
        # ============================================================
        # PLOT 3: Per-Class F1-Score Heatmap
        # ============================================================
        fig, ax = plt.subplots(figsize=(8, 10))
        
        f1_scores = results_df[['F1 Bearish (%)', 'F1 Bullish (%)']].values
        
        sns.heatmap(f1_scores, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=['Bearish', 'Bullish'], 
                   yticklabels=results_df['Name'],
                   cbar_kws={'label': 'F1-Score (%)'}, ax=ax, 
                   vmin=0, vmax=100, linewidths=0.5, linecolor='gray')
        
        ax.set_title('Per-Class F1-Score Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot3_path = save_dir / "plot3_per_class_f1.png"
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {plot3_path.name}")
        
        if IN_COLAB:
            display(Image(plot3_path))
        else:
            plt.show()
        plt.close()
        
        # ============================================================
        # PLOT 4: Best Model Confusion Matrix
        # ============================================================
        fig, ax = plt.subplots(figsize=(8, 7))
        
        best_model_id = results_df.iloc[0]['Model ID']
        conf_mat = self.results[best_model_id]['confusion_matrix']
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=['Bearish', 'Bullish'], 
                   yticklabels=['Bearish', 'Bullish'],
                   cbar_kws={'label': 'Proportion'}, ax=ax, square=True,
                   vmin=0, vmax=1, linewidths=2, linecolor='black')
        
        best_name = results_df.iloc[0]['Name']
        best_mcc = results_df.iloc[0]['MCC']
        
        ax.set_title(f'Best Model: {best_name}\nMCC: {best_mcc:.3f}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot4_path = save_dir / "plot4_best_model_confusion.png"
        plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {plot4_path.name}")
        
        if IN_COLAB:
            display(Image(plot4_path))
        else:
            plt.show()
        plt.close()
    
    def print_detailed_report(self, results_df):
        """Print clean comparison report"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Essential metrics only
        essential_cols = ['Name', 'MCC', 'F1-Score (%)', 'Test Acc (%)']
        print("\n" + results_df[essential_cols].to_string(index=False))
        
        # Best model
        best = results_df.iloc[0]
        print("\n" + "="*80)
        print("BEST MODEL")
        print("="*80)
        print(f"Name:        {best['Name']}")
        print(f"MCC:         {best['MCC']:.3f}")
        print(f"F1-Score:    {best['F1-Score (%)']:.1f}%")
        print(f"Test Acc:    {best['Test Acc (%)']:.1f}%")
        print(f"Bearish F1:  {best['F1 Bearish (%)']:.1f}%")
        print(f"Bullish F1:  {best['F1 Bullish (%)']:.1f}%")
        print(f"Parameters:  {best['Parameters']:,}")


def main():
    print("\n" + "="*70)
    print("BINARY MODEL COMPARISON")
    print("="*70)
    
    comparator = BinaryModelComparator()
    
    # Load all trained models
    available_models = config.list_available_models()
    loaded_models = []
    
    print("\nLoading trained models...", end=" ", flush=True)
    for model_id in available_models:
        if comparator.load_model(model_id):
            loaded_models.append(model_id)
    
    print(f"✓ ({len(loaded_models)} models)")
    
    if len(loaded_models) == 0:
        print("\n✗ No trained models found. Train models first.")
        return
    
    # Evaluate all models
    print("Evaluating models...", end=" ", flush=True)
    for model_id in loaded_models:
        model_config = config.get_model_config(model_id)
        feature_list = model_config['features']
        feature_set = 'engineered' if feature_list == config.ENGINEERED_FEATURES else 'baseline'
        
        test_data = pd.read_csv(
            config.REGIME_DATA_DIR / f"test_labeled_{feature_set}.csv",
            index_col=0, parse_dates=True
        )
        
        _, _, test_loader, _ = create_dataloaders(
            test_data, test_data, test_data, feature_list
        )
        
        comparator.evaluate_model(model_id, test_loader)
    
    print("✓")
    
    # Create results table
    results_df = comparator.create_results_table()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    comparator.plot_comparison(results_df, save_dir=config.DATA_DIR)
    
    # Print report
    comparator.print_detailed_report(results_df)
    
    # Save results
    results_df.to_csv(config.DATA_DIR / "model_results.csv", index=False)
    print(f"\n✓ Results saved: {config.DATA_DIR / 'model_results.csv'}")
    
    print("\n" + "="*70)
    print("✓ COMPARISON COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()