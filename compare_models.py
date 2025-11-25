"""Model Comparison: Compare baseline vs engineered models with visualizations"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

import config
from src.dataset import create_dataloaders
from src.model import TransformerRegimeClassifier

class ModelComparator:
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_name, checkpoint_path, n_features):
        """Load trained model from checkpoint"""
        model = TransformerRegimeClassifier(n_features=n_features)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.models[model_name] = {
            'model': model,
            'checkpoint': checkpoint,
            'n_features': n_features
        }
        
        print(f"Loaded {model_name}: Val Acc = {checkpoint['val_acc']:.2f}%")
        return model
    
    def evaluate_model(self, model_name, test_loader):
        """Evaluate model on test set"""
        model = self.models[model_name]['model']
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Metrics
        accuracy = (all_preds == all_labels).mean() * 100
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        self.results[model_name] = {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
        
        return accuracy
    
    def plot_comparison_summary(self, save_path=None):
        """Create comprehensive comparison visualization across multiple pages"""
        model_names = list(self.results.keys())
        colors = ['#3498db', '#e74c3c']
        n_models = len(model_names)
        
        # PAGE 1: Overview and Accuracy Metrics
        fig1 = plt.figure(figsize=(16, 10))
        gs1 = fig1.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
        
        # 1. Accuracy Comparison Bar Chart
        ax1 = fig1.add_subplot(gs1[0, 0])
        accuracies = [self.results[m]['accuracy'] for m in model_names]
        bars = ax1.bar(model_names, accuracies, color=colors[:n_models], alpha=0.7, edgecolor='black', width=0.6)
        ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 2. Model Summary Table
        ax2 = fig1.add_subplot(gs1[0, 1])
        ax2.axis('off')
        
        table_data = []
        for model_name in model_names:
            n_feat = self.models[model_name]['n_features']
            val_acc = self.models[model_name]['checkpoint']['val_acc']
            test_acc = self.results[model_name]['accuracy']
            n_params = self.models[model_name]['model'].get_num_parameters()
            
            table_data.append([
                model_name,
                n_feat,
                f"{val_acc:.1f}%",
                f"{test_acc:.1f}%",
                f"{n_params:,}"
            ])
        
        table = ax2.table(cellText=table_data,
                         colLabels=['Model', 'Features', 'Val Acc', 'Test Acc', 'Parameters'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0.2, 1, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('Model Summary', fontsize=14, fontweight='bold', pad=20)
        
        # 3. Per-Class Accuracy Comparison
        ax3 = fig1.add_subplot(gs1[1, 0])
        x = np.arange(len(config.REGIME_NAMES))
        width = 0.35
        
        for i, model_name in enumerate(model_names):
            conf_mat = self.results[model_name]['confusion_matrix']
            per_class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1) * 100
            offset = (i - (n_models-1)/2) * width
            ax3.bar(x + offset, per_class_acc, width, label=model_name, 
                   color=colors[i], alpha=0.7, edgecolor='black')
        
        ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Per-Regime Accuracy', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xticks(x)
        ax3.set_xticklabels(config.REGIME_NAMES, fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Prediction Distribution
        ax4 = fig1.add_subplot(gs1[1, 1])
        
        for i, model_name in enumerate(model_names):
            preds = self.results[model_name]['predictions']
            pred_counts = [(preds == j).sum() / len(preds) * 100 for j in range(3)]
            offset = (i - (n_models-1)/2) * width * 0.8
            ax4.bar(x + offset, pred_counts, width*0.8, label=model_name,
                   color=colors[i], alpha=0.7, edgecolor='black')
        
        labels = self.results[model_names[0]]['labels']
        true_counts = [(labels == j).sum() / len(labels) * 100 for j in range(3)]
        ax4.plot(x, true_counts, 'ko-', linewidth=2.5, markersize=10, label='Ground Truth')
        
        ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Prediction Distribution', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xticks(x)
        ax4.set_xticklabels(config.REGIME_NAMES, fontsize=11)
        ax4.legend(fontsize=10)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Comparison - Overview', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            path1 = Path(save_path).parent / (Path(save_path).stem + "_page1.png")
            plt.savefig(path1, dpi=300, bbox_inches='tight')
            print(f"\nSaved page 1: {path1}")
        plt.show()
        
        # PAGE 2: Confusion Matrices
        fig2 = plt.figure(figsize=(16, 8))
        
        for i, model_name in enumerate(model_names):
            ax = fig2.add_subplot(1, 2, i+1)
            conf_mat = self.results[model_name]['confusion_matrix']
            conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=config.REGIME_NAMES,
                       yticklabels=config.REGIME_NAMES,
                       cbar_kws={'label': 'Proportion'},
                       ax=ax, square=True, annot_kws={'size': 12})
            ax.set_title(f'{model_name} Model\nConfusion Matrix', fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel('True Regime', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Regime', fontsize=12, fontweight='bold')
            ax.tick_params(labelsize=11)
        
        plt.suptitle('Model Comparison - Confusion Matrices', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            path2 = Path(save_path).parent / (Path(save_path).stem + "_page2.png")
            plt.savefig(path2, dpi=300, bbox_inches='tight')
            print(f"Saved page 2: {path2}")
        plt.show()
        
        # PAGE 3: Training History and Confidence
        fig3 = plt.figure(figsize=(16, 10))
        gs3 = fig3.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
        
        # Training History
        for i, model_name in enumerate(model_names):
            ax = fig3.add_subplot(gs3[0, i])
            history = self.models[model_name]['checkpoint']['history']
            epochs = range(1, len(history['train_acc']) + 1)
            
            ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2.5)
            ax.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2.5)
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name} Model\nTraining History', fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)
        
        # Model Confidence
        ax_conf = fig3.add_subplot(gs3[1, :])
        
        for i, model_name in enumerate(model_names):
            probs = self.results[model_name]['probabilities']
            labels = self.results[model_name]['labels']
            
            regime_confidence = []
            for regime_idx in range(3):
                regime_mask = labels == regime_idx
                if regime_mask.sum() > 0:
                    regime_probs = probs[regime_mask][:, regime_idx]
                    regime_confidence.append(regime_probs.mean() * 100)
                else:
                    regime_confidence.append(0)
            
            x = np.arange(len(config.REGIME_NAMES))
            offset = (i - (n_models-1)/2) * width
            ax_conf.bar(x + offset, regime_confidence, width, label=model_name,
                       color=colors[i], alpha=0.7, edgecolor='black')
        
        ax_conf.set_ylabel('Avg Confidence (%)', fontsize=12, fontweight='bold')
        ax_conf.set_title('Model Confidence by Regime', fontsize=14, fontweight='bold', pad=15)
        ax_conf.set_xticks(x)
        ax_conf.set_xticklabels(config.REGIME_NAMES, fontsize=11)
        ax_conf.legend(fontsize=11)
        ax_conf.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Comparison - Training & Confidence', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            path3 = Path(save_path).parent / (Path(save_path).stem + "_page3.png")
            plt.savefig(path3, dpi=300, bbox_inches='tight')
            print(f"Saved page 3: {path3}")
        plt.show()
    
    def print_detailed_report(self):
        """Print detailed text report"""
        print("\n" + "="*70)
        print("DETAILED MODEL COMPARISON REPORT")
        print("="*70)
        
        for model_name in self.results.keys():
            print(f"\n{model_name.upper()}")
            print("-"*70)
            
            labels = self.results[model_name]['labels']
            preds = self.results[model_name]['predictions']
            
            print(classification_report(labels, preds, 
                                       target_names=config.REGIME_NAMES,
                                       digits=3))
            
            print(f"Overall Accuracy: {self.results[model_name]['accuracy']:.2f}%")
        
        # Winner
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_model[0].upper()}")
        print(f"Test Accuracy: {best_model[1]['accuracy']:.2f}%")
        print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    comparator = ModelComparator()
    
    # Load test data for both feature sets
    test_baseline = pd.read_csv(config.REGIME_DATA_DIR / "test_labeled_baseline.csv",
                                index_col=0, parse_dates=True)
    test_engineered = pd.read_csv(config.REGIME_DATA_DIR / "test_labeled_engineered.csv",
                                  index_col=0, parse_dates=True)
    
    # Check which models exist
    baseline_checkpoint = config.CHECKPOINT_DIR / "baseline_transformer" / "best.pth"
    engineered_checkpoint = config.CHECKPOINT_DIR / "engineered_transformer" / "best.pth"
    
    models_to_compare = []
    
    if baseline_checkpoint.exists():
        print("\nLoading Baseline Model...")
        _, _, test_loader_baseline, n_feat_baseline = create_dataloaders(
            test_baseline, test_baseline, test_baseline, config.BASELINE_FEATURES
        )
        comparator.load_model('Baseline', baseline_checkpoint, n_feat_baseline)
        comparator.evaluate_model('Baseline', test_loader_baseline)
        models_to_compare.append('Baseline')
    else:
        print("\nBaseline model not found. Train it first:")
        print("  python main.py --mode all --feature_set baseline")
    
    if engineered_checkpoint.exists():
        print("\nLoading Engineered Model...")
        _, _, test_loader_eng, n_feat_eng = create_dataloaders(
            test_engineered, test_engineered, test_engineered, config.ENGINEERED_FEATURES
        )
        comparator.load_model('Engineered', engineered_checkpoint, n_feat_eng)
        comparator.evaluate_model('Engineered', test_loader_eng)
        models_to_compare.append('Engineered')
    else:
        print("\nEngineered model not found. Train it first:")
        print("  python main.py --mode all --feature_set engineered")
    
    if len(models_to_compare) >= 2:
        # Generate comparison visualization
        print("\n" + "="*70)
        print("GENERATING COMPARISON VISUALIZATION")
        print("="*70)
        
        comparator.plot_comparison_summary(
            save_path=config.DATA_DIR / "model_comparison.png"
        )
        
        # Print detailed report
        comparator.print_detailed_report()
    
    elif len(models_to_compare) == 1:
        print(f"\nOnly {models_to_compare[0]} model available.")
        print("Train the other model to enable comparison.")
    
    else:
        print("\nNo trained models found. Train models first:")
        print("  python main.py --mode all --feature_set baseline")
        print("  python main.py --mode all --feature_set engineered")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70 + "\n")