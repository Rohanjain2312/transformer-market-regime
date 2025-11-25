"""Model Comparison: Compare all trained models with comprehensive visualizations"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

import config
from src.dataset import create_dataloaders
from src.model import create_model

# Try to import Colab display
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False

class ModelComparator:
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_id):
        """Load trained model from checkpoint"""
        checkpoint_path = config.CHECKPOINT_DIR / model_id / "best.pth"
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found for {model_id}")
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
        
        print(f"Loaded {model_config['name']}: Val Acc = {checkpoint['val_acc']:.2f}%")
        return True
    
    def evaluate_model(self, model_id, test_loader):
        """Evaluate model on test set"""
        model = self.models[model_id]['model']
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
        
        accuracy = (all_preds == all_labels).mean() * 100
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        self.results[model_id] = {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
        
        return accuracy
    
    def create_results_table(self):
        """Create comprehensive results table"""
        table_data = []
        
        for model_id in self.results.keys():
            model_info = self.models[model_id]
            config_data = model_info['config']
            checkpoint = model_info['checkpoint']
            result = self.results[model_id]
            
            # Per-class accuracy
            conf_mat = result['confusion_matrix']
            per_class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1) * 100
            
            table_data.append({
                'Model ID': model_id,
                'Name': config_data['name'],
                'Architecture': config_data['architecture'],
                'Features': config_data['n_features'],
                'd_model': config_data['d_model'],
                'Layers': config_data['n_layers'],
                'Heads': config_data['n_heads'],
                'Parameters': model_info['model'].get_num_parameters(),
                'Val Acc (%)': checkpoint['val_acc'],
                'Test Acc (%)': result['accuracy'],
                'Bearish Acc (%)': per_class_acc[0],
                'Neutral Acc (%)': per_class_acc[1],
                'Bullish Acc (%)': per_class_acc[2]
            })
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('Test Acc (%)', ascending=False)
        
        return df
    
    def plot_comparison_overview(self, results_df, save_path=None):
        """Page 1: Overview comparison"""
        n_models = len(results_df)
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
        
        model_ids = results_df['Model ID'].tolist()
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        # 1. Test Accuracy Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.barh(range(n_models), results_df['Test Acc (%)'], color=colors, edgecolor='black')
        ax1.set_yticks(range(n_models))
        ax1.set_yticklabels(results_df['Name'], fontsize=9)
        ax1.set_xlabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3)
        
        for i, (bar, acc) in enumerate(zip(bars, results_df['Test Acc (%)'])):
            ax1.text(acc + 1, i, f'{acc:.1f}%', va='center', fontweight='bold', fontsize=9)
        
        # 2. Per-Regime Accuracy Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        regime_accs = results_df[['Bearish Acc (%)', 'Neutral Acc (%)', 'Bullish Acc (%)']].values
        
        sns.heatmap(regime_accs, annot=True, fmt='.1f', cmap='RdYlGn', 
                   xticklabels=config.REGIME_NAMES, yticklabels=results_df['Name'],
                   cbar_kws={'label': 'Accuracy (%)'}, ax=ax2, vmin=0, vmax=100)
        ax2.set_title('Per-Regime Accuracy', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        
        # 3. Model Complexity vs Performance
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(results_df['Parameters'], results_df['Test Acc (%)'], 
                             c=range(n_models), cmap='Set3', s=200, alpha=0.7, edgecolor='black', linewidth=2)
        
        for i, row in results_df.iterrows():
            ax3.annotate(row['Name'], (row['Parameters'], row['Test Acc (%)']),
                        fontsize=8, ha='right', va='bottom')
        
        ax3.set_xlabel('Number of Parameters', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Model Complexity vs Performance', fontsize=13, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3)
        
        # 4. Architecture Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        arch_counts = results_df['Architecture'].value_counts()
        ax4.pie(arch_counts.values, labels=arch_counts.index, autopct='%1.0f%%',
               colors=colors[:len(arch_counts)], startangle=90)
        ax4.set_title('Architecture Distribution', fontsize=13, fontweight='bold', pad=15)
        
        plt.suptitle('Model Comparison - Overview', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            if IN_COLAB:
                display(Image(save_path))
        
        if not IN_COLAB:
            plt.show()
        plt.close()
    
    def plot_confusion_matrices(self, results_df, save_path=None):
        """Page 2: Confusion matrices for all models"""
        n_models = len(results_df)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        
        for idx, model_id in enumerate(results_df['Model ID']):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            
            conf_mat = self.results[model_id]['confusion_matrix']
            conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=config.REGIME_NAMES, yticklabels=config.REGIME_NAMES,
                       cbar_kws={'label': 'Proportion'}, ax=ax, square=True)
            
            model_name = self.models[model_id]['config']['name']
            test_acc = self.results[model_id]['accuracy']
            ax.set_title(f'{model_name}\nTest Acc: {test_acc:.1f}%', 
                        fontsize=11, fontweight='bold', pad=10)
            ax.set_ylabel('True Regime', fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted Regime', fontsize=10, fontweight='bold')
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            if IN_COLAB:
                display(Image(save_path))
        
        if not IN_COLAB:
            plt.show()
        plt.close()
    
    def plot_training_comparison(self, results_df, save_path=None):
        """Page 3: Training history comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
        
        # Plot training curves for each model
        for idx, model_id in enumerate(results_df['Model ID']):
            history = self.models[model_id]['checkpoint']['history']
            epochs = range(1, len(history['train_acc']) + 1)
            color = colors[idx]
            model_name = self.models[model_id]['config']['name']
            
            # Train accuracy
            axes[0].plot(epochs, history['train_acc'], color=color, 
                        linewidth=2, label=model_name, alpha=0.8)
            
            # Val accuracy
            axes[1].plot(epochs, history['val_acc'], color=color,
                        linewidth=2, label=model_name, alpha=0.8)
            
            # Train loss
            axes[2].plot(epochs, history['train_loss'], color=color,
                        linewidth=2, label=model_name, alpha=0.8)
            
            # Val loss
            axes[3].plot(epochs, history['val_loss'], color=color,
                        linewidth=2, label=model_name, alpha=0.8)
        
        # Formatting
        axes[0].set_title('Training Accuracy', fontsize=13, fontweight='bold', pad=10)
        axes[0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)
        
        axes[1].set_title('Validation Accuracy', fontsize=13, fontweight='bold', pad=10)
        axes[1].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)
        
        axes[2].set_title('Training Loss', fontsize=13, fontweight='bold', pad=10)
        axes[2].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Loss', fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=9)
        
        axes[3].set_title('Validation Loss', fontsize=13, fontweight='bold', pad=10)
        axes[3].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[3].set_ylabel('Loss', fontsize=11, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(fontsize=9)
        
        plt.suptitle('Training History - All Models', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            if IN_COLAB:
                display(Image(save_path))
        
        if not IN_COLAB:
            plt.show()
        plt.close()
    
    def print_detailed_report(self, results_df):
        """Print comprehensive comparison report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*80)
        
        # Display results table
        print("\n" + results_df.to_string(index=False))
        
        # Best model
        best_row = results_df.iloc[0]
        print("\n" + "="*80)
        print("BEST MODEL")
        print("="*80)
        print(f"Name:         {best_row['Name']}")
        print(f"Model ID:     {best_row['Model ID']}")
        print(f"Test Acc:     {best_row['Test Acc (%)']:.2f}%")
        print(f"Val Acc:      {best_row['Val Acc (%)']:.2f}%")
        print(f"Parameters:   {best_row['Parameters']:,}")
        print(f"Architecture: {best_row['Architecture']}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-MODEL COMPARISON")
    print("="*70)
    
    comparator = ModelComparator()
    
    # Find all trained models
    available_models = config.list_available_models()
    loaded_models = []
    
    print("\nScanning for trained models...")
    for model_id in available_models:
        if comparator.load_model(model_id):
            loaded_models.append(model_id)
    
    if len(loaded_models) == 0:
        print("\nNo trained models found. Train models first using main.py")
        exit(1)
    
    print(f"\nFound {len(loaded_models)} trained model(s)")
    
    # Load test data and evaluate all models
    print("\nEvaluating models...")
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
    
    # Create results table
    results_df = comparator.create_results_table()
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)
    
    comparator.plot_comparison_overview(
        results_df, 
        save_path=config.DATA_DIR / "comparison_overview.png"
    )
    
    comparator.plot_confusion_matrices(
        results_df,
        save_path=config.DATA_DIR / "comparison_confusion.png"
    )
    
    comparator.plot_training_comparison(
        results_df,
        save_path=config.DATA_DIR / "comparison_training.png"
    )
    
    # Print report
    comparator.print_detailed_report(results_df)
    
    # Save results table
    results_df.to_csv(config.DATA_DIR / "model_comparison_results.csv", index=False)
    print(f"\nSaved results table: {config.DATA_DIR / 'model_comparison_results.csv'}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70 + "\n")