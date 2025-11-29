"""Compare All Binary Models: Evaluate and visualize performance"""

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
        """Evaluate binary model on test set"""
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
        
        # Compute metrics
        accuracy = (all_preds == all_labels).mean() * 100
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
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
            'bullish_acc': bullish_acc
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
            
            table_data.append({
                'Model ID': model_id,
                'Name': config_data['name'],
                'Architecture': config_data['architecture'],
                'Parameters': model_info['model'].get_num_parameters(),
                'Val Acc (%)': checkpoint['val_acc'],
                'Test Acc (%)': result['accuracy'],
                'Bearish Acc (%)': result['bearish_acc'],
                'Bullish Acc (%)': result['bullish_acc']
            })
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('Test Acc (%)', ascending=False)
        
        return df
    
    def plot_comparison(self, results_df, save_path=None):
        """Plot comprehensive comparison"""
        n_models = len(results_df)
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        model_ids = results_df['Model ID'].tolist()
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        # 1. Test Accuracy Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.barh(range(n_models), results_df['Test Acc (%)'], color=colors, edgecolor='black')
        ax1.set_yticks(range(n_models))
        ax1.set_yticklabels(results_df['Name'], fontsize=9)
        ax1.set_xlabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Binary Classification - Test Accuracy', fontsize=13, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3)
        
        for i, (bar, acc) in enumerate(zip(bars, results_df['Test Acc (%)'])):
            ax1.text(acc + 0.5, i, f'{acc:.1f}%', va='center', fontweight='bold', fontsize=9)
        
        # 2. Per-Class Accuracy Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        class_accs = results_df[['Bearish Acc (%)', 'Bullish Acc (%)']].values
        
        sns.heatmap(class_accs, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=['Bearish', 'Bullish'], yticklabels=results_df['Name'],
                   cbar_kws={'label': 'Accuracy (%)'}, ax=ax2, vmin=0, vmax=100)
        ax2.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        
        # 3. Model Complexity vs Performance
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(results_df['Parameters'], results_df['Test Acc (%)'],
                             c=range(n_models), cmap='Set3', s=200, alpha=0.7, 
                             edgecolor='black', linewidth=2)
        
        for i, row in results_df.iterrows():
            ax3.annotate(row['Name'], (row['Parameters'], row['Test Acc (%)']),
                        fontsize=8, ha='right', va='bottom')
        
        ax3.set_xlabel('Number of Parameters', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Model Complexity vs Performance', fontsize=13, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3)
        
        # 4. Best Model Confusion Matrix
        ax4 = fig.add_subplot(gs[1, 1])
        best_model_id = results_df.iloc[0]['Model ID']
        conf_mat = self.results[best_model_id]['confusion_matrix']
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=['Bearish', 'Bullish'], yticklabels=['Bearish', 'Bullish'],
                   cbar_kws={'label': 'Proportion'}, ax=ax4, square=True)
        
        best_name = results_df.iloc[0]['Name']
        best_acc = results_df.iloc[0]['Test Acc (%)']
        ax4.set_title(f'Best Model: {best_name}\nTest Acc: {best_acc:.1f}%',
                     fontsize=12, fontweight='bold', pad=10)
        ax4.set_ylabel('True Label', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
        
        plt.suptitle('Binary Classification - Model Comparison', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved comparison plot: {save_path}")
            if IN_COLAB:
                display(Image(save_path))
        
        if not IN_COLAB:
            plt.show()
        plt.close()
    
    def print_detailed_report(self, results_df):
        """Print comprehensive comparison report"""
        print("\n" + "="*80)
        print("BINARY CLASSIFICATION - COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        # Display results table
        print("\n" + results_df.to_string(index=False))
        
        # Best model
        best_row = results_df.iloc[0]
        print("\n" + "="*80)
        print("BEST MODEL")
        print("="*80)
        print(f"Name:           {best_row['Name']}")
        print(f"Model ID:       {best_row['Model ID']}")
        print(f"Test Acc:       {best_row['Test Acc (%)']:.2f}%")
        print(f"Val Acc:        {best_row['Val Acc (%)']:.2f}%")
        print(f"Bearish Acc:    {best_row['Bearish Acc (%)']:.2f}%")
        print(f"Bullish Acc:    {best_row['Bullish Acc (%)']:.2f}%")
        print(f"Parameters:     {best_row['Parameters']:,}")
        
        # Overall statistics
        print("\n" + "="*80)
        print("OVERALL STATISTICS")
        print("="*80)
        print(f"Average Test Acc:    {results_df['Test Acc (%)'].mean():.2f}%")
        print(f"Best Test Acc:       {results_df['Test Acc (%)'].max():.2f}%")
        print(f"Worst Test Acc:      {results_df['Test Acc (%)'].min():.2f}%")
        print(f"Std Dev:             {results_df['Test Acc (%)'].std():.2f}%")


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
    
    # Generate visualization
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)
    
    comparator.plot_comparison(
        results_df,
        save_path=config.DATA_DIR / "binary_model_comparison.png"
    )
    
    # Print report
    comparator.print_detailed_report(results_df)
    
    # Save results table
    results_df.to_csv(config.DATA_DIR / "binary_model_results.csv", index=False)
    print(f"\nSaved results table: {config.DATA_DIR / 'binary_model_results.csv'}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70 + "\n")