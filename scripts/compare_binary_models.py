"""Compare All Binary Models: Evaluate and visualize performance"""

import sys
from pathlib import Path

# Add project root to path
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

# Try to import Colab display
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False

# Try to import Colab display
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
        checkpoint_path = config.CHECKPOINT_DIR / f"{model_id}_binary" / "best.pth"
        
        if not checkpoint_path.exists():
            print(f"  Warning: Binary checkpoint not found for {model_id}")
            return False
        
        model_config = config.get_model_config(model_id)
        model, _ = create_model(model_id, self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.models[model_id] = {
            'model': model,
            'config': model_config,
            'checkpoint': checkpoint
        }
        
        val_acc = checkpoint['val_acc']
        print(f"  ✓ {model_config['name']}: {val_acc:.2f}% val acc")
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
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        # Per-class accuracy (for backward compatibility)
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
            # New comprehensive metrics
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
                'Val Acc (%)': checkpoint['val_acc'],
                'Test Acc (%)': result['accuracy'],
                # Primary Metrics
                'F1-Score (%)': result['f1_macro'],
                'MCC': result['mcc'],
                'Balanced Acc (%)': result['balanced_accuracy'],
                # Per-class F1
                'F1 Bearish (%)': result['f1_bearish'],
                'F1 Bullish (%)': result['f1_bullish'],
                # Per-class Precision
                'Precision Bearish (%)': result['precision_bearish'],
                'Precision Bullish (%)': result['precision_bullish'],
                # Per-class Recall
                'Recall Bearish (%)': result['recall_bearish'],
                'Recall Bullish (%)': result['recall_bullish'],
                # Legacy (for compatibility)
                'Bearish Acc (%)': result['bearish_acc'],
                'Bullish Acc (%)': result['bullish_acc']
            })
        
        df = pd.DataFrame(table_data)
        
        # Sort by MCC (primary metric for model selection)
        df = df.sort_values('MCC', ascending=False)
        
        return df
    
    def plot_comparison(self, results_df, save_dir=None):
        """Plot separate clean comparison graphs"""
        if save_dir is None:
            save_dir = config.DATA_DIR
        
        n_models = len(results_df)
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        # ==================================================================
        # PLOT 1: Test Accuracy Bar Chart
        # ==================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(range(n_models), results_df['Test Acc (%)'], 
                      color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(results_df['Name'], fontsize=10)
        ax.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Binary Classification - Test Accuracy Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 100)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, results_df['Test Acc (%)'])):
            ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plot1_path = save_dir / "plot1_test_accuracy.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot1_path}")
        
        if IN_COLAB:
            display(Image(plot1_path))
        else:
            plt.show()
        plt.close()
        
        # ==================================================================
        # PLOT 2: Per-Class Accuracy Heatmap
        # ==================================================================
        fig, ax = plt.subplots(figsize=(8, 10))
        
        class_accs = results_df[['Bearish Acc (%)', 'Bullish Acc (%)']].values
        
        sns.heatmap(class_accs, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=['Bearish', 'Bullish'], 
                   yticklabels=results_df['Name'],
                   cbar_kws={'label': 'Accuracy (%)'}, ax=ax, 
                   vmin=0, vmax=100, linewidths=0.5, linecolor='gray')
        
        ax.set_title('Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot2_path = save_dir / "plot2_per_class_accuracy.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot2_path}")
        
        if IN_COLAB:
            display(Image(plot2_path))
        else:
            plt.show()
        plt.close()
        
        # ==================================================================
        # PLOT 3: Model Complexity vs Performance Scatter
        # ==================================================================
        fig, ax = plt.subplots(figsize=(12, 7))
        
        scatter = ax.scatter(results_df['Parameters'], results_df['Test Acc (%)'],
                           c=range(n_models), cmap='Set3', s=300, alpha=0.7,
                           edgecolor='black', linewidth=2)
        
        # Add model names as annotations
        for i, row in results_df.iterrows():
            ax.annotate(row['Name'], 
                       (row['Parameters'], row['Test Acc (%)']),
                       fontsize=9, ha='right', va='bottom',
                       xytext=(-5, 5), textcoords='offset points')
        
        ax.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Complexity vs Performance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(50, 90)
        
        plt.tight_layout()
        plot3_path = save_dir / "plot3_complexity_vs_performance.png"
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot3_path}")
        
        if IN_COLAB:
            display(Image(plot3_path))
        else:
            plt.show()
        plt.close()
        
        # ==================================================================
        # PLOT 4: Best Model Confusion Matrix
        # ==================================================================
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
        best_acc = results_df.iloc[0]['Test Acc (%)']
        
        ax.set_title(f'Best Model: {best_name}\nTest Accuracy: {best_acc:.2f}%',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot4_path = save_dir / "plot4_best_model_confusion.png"
        plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot4_path}")
        
        if IN_COLAB:
            display(Image(plot4_path))
        else:
            plt.show()
        plt.close()
        
    def plot_comparison(self, results_df, save_dir=None):
        """Plot separate clean comparison graphs with comprehensive metrics"""
        if save_dir is None:
            save_dir = config.DATA_DIR
        
        n_models = len(results_df)
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        # ==================================================================
        # PLOT 1: F1-Score Comparison (PRIMARY METRIC)
        # ==================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(range(n_models), results_df['F1-Score (%)'], 
                      color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(results_df['Name'], fontsize=10)
        ax.set_xlabel('F1-Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Binary Classification - F1-Score Comparison (Primary Metric)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 100)
        
        for i, (bar, f1) in enumerate(zip(bars, results_df['F1-Score (%)'])):
            ax.text(f1 + 1, i, f'{f1:.1f}%', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plot1_path = save_dir / "plot1_f1_score.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot1_path}")
        
        if IN_COLAB:
            display(Image(filename=str(plot1_path)))
        else:
            plt.show()
        plt.close()
        
        # ==================================================================
        # PLOT 2: MCC Comparison (MODEL SELECTION METRIC)
        # ==================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(range(n_models), results_df['MCC'], 
                      color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(results_df['Name'], fontsize=10)
        ax.set_xlabel('Matthews Correlation Coefficient (MCC)', fontsize=12, fontweight='bold')
        ax.set_title('Binary Classification - MCC Comparison (Model Selection Metric)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(-1, 1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        for i, (bar, mcc) in enumerate(zip(bars, results_df['MCC'])):
            ax.text(mcc + 0.02, i, f'{mcc:.3f}', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plot2_path = save_dir / "plot2_mcc.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot2_path}")
        
        if IN_COLAB:
            display(Image(filename=str(plot2_path)))
        else:
            plt.show()
        plt.close()
        
        # ==================================================================
        # PLOT 3: Per-Class F1-Score Heatmap
        # ==================================================================
        fig, ax = plt.subplots(figsize=(8, 10))
        
        f1_scores = results_df[['F1 Bearish (%)', 'F1 Bullish (%)']].values
        
        sns.heatmap(f1_scores, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=['Bearish F1', 'Bullish F1'], 
                   yticklabels=results_df['Name'],
                   cbar_kws={'label': 'F1-Score (%)'}, ax=ax, 
                   vmin=0, vmax=100, linewidths=0.5, linecolor='gray')
        
        ax.set_title('Per-Class F1-Score Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot3_path = save_dir / "plot3_per_class_f1.png"
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot3_path}")
        
        if IN_COLAB:
            display(Image(filename=str(plot3_path)))
        else:
            plt.show()
        plt.close()
        
        # ==================================================================
        # PLOT 4: Precision vs Recall Scatter
        # ==================================================================
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Bearish class
        ax.scatter(results_df['Recall Bearish (%)'], results_df['Precision Bearish (%)'],
                  c='red', s=200, alpha=0.7, edgecolor='black', linewidth=2,
                  label='Bearish', marker='o')
        
        # Bullish class
        ax.scatter(results_df['Recall Bullish (%)'], results_df['Precision Bullish (%)'],
                  c='green', s=200, alpha=0.7, edgecolor='black', linewidth=2,
                  label='Bullish', marker='s')
        
        # Add model names
        for i, row in results_df.iterrows():
            ax.annotate(row['Name'], 
                       (row['Recall Bearish (%)'], row['Precision Bearish (%)']),
                       fontsize=8, ha='right', va='bottom', xytext=(-5, 5),
                       textcoords='offset points')
        
        ax.set_xlabel('Recall (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
        ax.set_title('Precision vs Recall - Per Class', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plot4_path = save_dir / "plot4_precision_recall_scatter.png"
        plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot4_path}")
        
        if IN_COLAB:
            display(Image(filename=str(plot4_path)))
        else:
            plt.show()
        plt.close()
        
        # ==================================================================
        # PLOT 5: Best Model Confusion Matrix
        # ==================================================================
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
        best_f1 = results_df.iloc[0]['F1-Score (%)']
        best_mcc = results_df.iloc[0]['MCC']
        
        ax.set_title(f'Best Model: {best_name}\nF1-Score: {best_f1:.2f}% | MCC: {best_mcc:.3f}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot5_path = save_dir / "plot5_best_model_confusion.png"
        plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot5_path}")
        
        if IN_COLAB:
            display(Image(filename=str(plot5_path)))
        else:
            plt.show()
        plt.close()
        
        # ==================================================================
        # PLOT 6: Precision-Recall Curves for Best Model
        # ==================================================================
        fig, ax = plt.subplots(figsize=(10, 8))
        
        best_model_id = results_df.iloc[0]['Model ID']
        y_true = self.results[best_model_id]['labels']
        y_probs = self.results[best_model_id]['probabilities']
        
        # Bearish class (class 0)
        precision_bearish, recall_bearish, _ = precision_recall_curve(
            (y_true == 0).astype(int), y_probs[:, 0]
        )
        ap_bearish = average_precision_score((y_true == 0).astype(int), y_probs[:, 0])
        
        # Bullish class (class 1)
        precision_bullish, recall_bullish, _ = precision_recall_curve(
            (y_true == 1).astype(int), y_probs[:, 1]
        )
        ap_bullish = average_precision_score((y_true == 1).astype(int), y_probs[:, 1])
        
        ax.plot(recall_bearish, precision_bearish, linewidth=2.5, 
               label=f'Bearish (AP = {ap_bearish:.3f})', color='red')
        ax.plot(recall_bullish, precision_bullish, linewidth=2.5,
               label=f'Bullish (AP = {ap_bullish:.3f})', color='green')
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title(f'Precision-Recall Curves - {best_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plot6_path = save_dir / "plot6_precision_recall_curves.png"
        plt.savefig(plot6_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot6_path}")
        
        if IN_COLAB:
            display(Image(filename=str(plot6_path)))
        else:
            plt.show()
        plt.close()
        
        # ==================================================================
        # PLOT 7: Balanced Accuracy Comparison
        # ==================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(range(n_models), results_df['Balanced Acc (%)'], 
                      color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(results_df['Name'], fontsize=10)
        ax.set_xlabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Binary Classification - Balanced Accuracy Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 100)
        
        for i, (bar, bal_acc) in enumerate(zip(bars, results_df['Balanced Acc (%)'])):
            ax.text(bal_acc + 1, i, f'{bal_acc:.1f}%', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plot7_path = save_dir / "plot7_balanced_accuracy.png"
        plt.savefig(plot7_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot7_path}")
        
        if IN_COLAB:
            display(Image(filename=str(plot7_path)))
        else:
            plt.show()
        plt.close()
        
        print(f"\nAll plots saved to: {save_dir}")
        print("Total: 7 visualizations generated")
    
    def print_detailed_report(self, results_df):
        """Print clean, essential comparison report"""
        print("\n" + "="*80)
        print("MODEL COMPARISON - PRIMARY METRICS")
        print("="*80)
        
        # Show only essential columns in clean format
        essential_cols = ['Name', 'F1-Score (%)', 'MCC', 'Balanced Acc (%)']
        print("\n" + results_df[essential_cols].to_string(index=False))
        
        # Best model summary (compact)
        best = results_df.iloc[0]
        print("\n" + "="*80)
        print(f"BEST MODEL: {best['Name']}")
        print("="*80)
        print(f"F1-Score: {best['F1-Score (%)']:.1f}% | MCC: {best['MCC']:.3f} | Balanced Acc: {best['Balanced Acc (%)']:.1f}%")
        print(f"Bearish F1: {best['F1 Bearish (%)']:.1f}% | Bullish F1: {best['F1 Bullish (%)']:.1f}%")
        print(f"Parameters: {best['Parameters']:,} | Test Acc: {best['Test Acc (%)']:.1f}%")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BINARY MODEL COMPARISON")
    print("="*70)
    
    comparator = BinaryModelComparator()
    
    # Find all trained binary models
    available_models = config.list_available_models()
    loaded_models = []
    
    print("\nScanning for trained binary models...")
    for model_id in available_models:
        if comparator.load_model(model_id):
            loaded_models.append(model_id)
    
    if len(loaded_models) == 0:
        print("\nNo trained binary models found.")
        print("Run: python train_all_binary.py")
        exit(1)
    
    print(f"\nFound {len(loaded_models)} trained binary model(s)")
    
    # Evaluate all models (each with their own features)
    print("\nEvaluating models...")
    for model_id in loaded_models:
        # Get model config to know which features to use
        model_config = config.get_model_config(model_id)
        feature_list = model_config['features']
        
        # Determine which feature set
        if feature_list == config.BASELINE_FEATURES:
            feature_set = 'baseline'
        else:
            feature_set = 'engineered'
        
        # Load appropriate test data
        test_data = pd.read_csv(
            config.REGIME_DATA_DIR / f"test_labeled_{feature_set}_binary.csv",
            index_col=0, parse_dates=True
        )
        
        # Create test loader with correct features
        _, _, test_loader, _ = create_dataloaders(
            test_data, test_data, test_data,
            feature_list,
            batch_size=config.BATCH_SIZE_GPU if torch.cuda.is_available() else config.BATCH_SIZE_CPU
        )
        
        comparator.evaluate_model(model_id, test_loader)
    
    # Create results table
    results_df = comparator.create_results_table()
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)
    
    comparator.plot_comparison(results_df, save_dir=config.DATA_DIR)
    
    # Print report
    comparator.print_detailed_report(results_df)
    
    # Save results table
    results_df.to_csv(config.DATA_DIR / "binary_model_results.csv", index=False)
    print(f"\nSaved results table: {config.DATA_DIR / 'binary_model_results.csv'}")
    
    # ==================================================================
    # FINAL DISPLAY FOR COLAB
    # ==================================================================
    print("\n" + "="*70)
    print("VISUALIZATIONS")
    print("="*70)
    
    # Display all 7 plots in Colab
    if IN_COLAB:
        from IPython.display import display as ipython_display, Image as IPythonImage
        
        plot_files = [
            ("F1-Score Comparison", config.DATA_DIR / "plot1_f1_score.png"),
            ("MCC Comparison", config.DATA_DIR / "plot2_mcc.png"),
            ("Per-Class F1 Heatmap", config.DATA_DIR / "plot3_per_class_f1.png"),
            ("Precision vs Recall", config.DATA_DIR / "plot4_precision_recall_scatter.png"),
            ("Confusion Matrix", config.DATA_DIR / "plot5_best_model_confusion.png"),
            ("Precision-Recall Curves", config.DATA_DIR / "plot6_precision_recall_curves.png"),
            ("Balanced Accuracy", config.DATA_DIR / "plot7_balanced_accuracy.png")
        ]
        
        for title, plot_path in plot_files:
            if plot_path.exists():
                print(f"\n{title}")
                print("-" * 70)
                ipython_display(IPythonImage(filename=str(plot_path)))
    else:
        print("\nPlots saved to data/ folder")
    
    print("\n" + "="*70)
    print("COMPLETE - Results saved to binary_model_results.csv")
    print("="*70 + "\n")
