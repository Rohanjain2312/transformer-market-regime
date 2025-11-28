"""Evaluate Ensemble: Compare ensemble strategies against individual models"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config
from src.ensemble import RegimeEnsemble
from src.dataset import create_dataloaders

# Try to import Colab display
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False


def load_individual_results():
    """Load results from individual model comparison"""
    results_path = config.DATA_DIR / "model_comparison_results.csv"
    
    if results_path.exists():
        df = pd.read_csv(results_path)
        return df
    else:
        print("Warning: model_comparison_results.csv not found")
        return None


def plot_ensemble_comparison(individual_df, ensemble_results, save_path=None):
    """
    Plot comparison between individual models and ensemble
    
    Args:
        individual_df: DataFrame with individual model results
        ensemble_results: Dict with ensemble results for both strategies
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== Plot 1: Overall Accuracy Comparison =====
    ax1 = axes[0, 0]
    
    # Get top 5 models used in ensemble
    top_5 = individual_df[individual_df['Model ID'].isin(ensemble_results['model_ids'])]
    top_5 = top_5.sort_values('Test Acc (%)', ascending=False)
    
    # Prepare data
    model_names = top_5['Name'].tolist()
    test_accs = top_5['Test Acc (%)'].tolist()
    
    # Add ensemble results
    model_names.append('Ensemble\n(Weighted)')
    test_accs.append(ensemble_results['weighted']['accuracy'])
    
    model_names.append('Ensemble\n(Best-Per-Regime)')
    test_accs.append(ensemble_results['best_per_regime']['accuracy'])
    
    # Plot
    colors = ['skyblue'] * len(top_5) + ['green', 'darkgreen']
    bars = ax1.barh(range(len(model_names)), test_accs, color=colors, edgecolor='black')
    
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names, fontsize=9)
    ax1.set_xlabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Test Accuracy Comparison', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, test_accs)):
        ax1.text(acc + 0.5, i, f'{acc:.1f}%', va='center', fontweight='bold', fontsize=9)
    
    # ===== Plot 2: Per-Regime Accuracy Heatmap =====
    ax2 = axes[0, 1]
    
    # Prepare data
    regime_data = []
    labels = []
    
    for _, row in top_5.iterrows():
        regime_data.append([
            row['Bearish Acc (%)'],
            row['Neutral Acc (%)'],
            row['Bullish Acc (%)']
        ])
        labels.append(row['Name'])
    
    # Add ensemble results
    regime_data.append([
        ensemble_results['weighted']['per_regime_accuracy']['Bearish'],
        ensemble_results['weighted']['per_regime_accuracy']['Neutral'],
        ensemble_results['weighted']['per_regime_accuracy']['Bullish']
    ])
    labels.append('Ensemble (Weighted)')
    
    regime_data.append([
        ensemble_results['best_per_regime']['per_regime_accuracy']['Bearish'],
        ensemble_results['best_per_regime']['per_regime_accuracy']['Neutral'],
        ensemble_results['best_per_regime']['per_regime_accuracy']['Bullish']
    ])
    labels.append('Ensemble (Best-Per-Regime)')
    
    regime_data = np.array(regime_data)
    
    sns.heatmap(regime_data, annot=True, fmt='.1f', cmap='RdYlGn',
               xticklabels=config.REGIME_NAMES, yticklabels=labels,
               cbar_kws={'label': 'Accuracy (%)'}, ax=ax2, vmin=0, vmax=100)
    ax2.set_title('Per-Regime Accuracy Comparison', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    # ===== Plot 3: Ensemble Weighted - Confusion Matrix =====
    ax3 = axes[1, 0]
    
    conf_mat = ensemble_results['weighted']['confusion_matrix']
    conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=config.REGIME_NAMES, yticklabels=config.REGIME_NAMES,
               cbar_kws={'label': 'Proportion'}, ax=ax3, square=True)
    ax3.set_title(f'Ensemble (Weighted) Confusion Matrix\nTest Acc: {ensemble_results["weighted"]["accuracy"]:.1f}%',
                 fontsize=12, fontweight='bold', pad=10)
    ax3.set_ylabel('True Regime', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Predicted Regime', fontsize=10, fontweight='bold')
    
    # ===== Plot 4: Ensemble Best-Per-Regime - Confusion Matrix =====
    ax4 = axes[1, 1]
    
    conf_mat = ensemble_results['best_per_regime']['confusion_matrix']
    conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=config.REGIME_NAMES, yticklabels=config.REGIME_NAMES,
               cbar_kws={'label': 'Proportion'}, ax=ax4, square=True)
    ax4.set_title(f'Ensemble (Best-Per-Regime) Confusion Matrix\nTest Acc: {ensemble_results["best_per_regime"]["accuracy"]:.1f}%',
                 fontsize=12, fontweight='bold', pad=10)
    ax4.set_ylabel('True Regime', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Predicted Regime', fontsize=10, fontweight='bold')
    
    plt.suptitle('Ensemble vs Individual Models - Comprehensive Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plot: {save_path}")
        if IN_COLAB:
            display(Image(save_path))
    
    if not IN_COLAB:
        plt.show()
    plt.close()


def save_ensemble_results(ensemble_results, individual_df):
    """Save ensemble results to CSV"""
    
    # Create results dataframe
    results_data = []
    
    # Add individual models (top 5 only)
    top_5 = individual_df[individual_df['Model ID'].isin(ensemble_results['model_ids'])]
    for _, row in top_5.iterrows():
        results_data.append({
            'Model': row['Name'],
            'Type': 'Individual',
            'Test Acc (%)': row['Test Acc (%)'],
            'Bearish Acc (%)': row['Bearish Acc (%)'],
            'Neutral Acc (%)': row['Neutral Acc (%)'],
            'Bullish Acc (%)': row['Bullish Acc (%)']
        })
    
    # Add ensemble results
    for strategy_name, strategy_key in [('Weighted', 'weighted'), ('Best-Per-Regime', 'best_per_regime')]:
        results_data.append({
            'Model': f'Ensemble ({strategy_name})',
            'Type': 'Ensemble',
            'Test Acc (%)': ensemble_results[strategy_key]['accuracy'],
            'Bearish Acc (%)': ensemble_results[strategy_key]['per_regime_accuracy']['Bearish'],
            'Neutral Acc (%)': ensemble_results[strategy_key]['per_regime_accuracy']['Neutral'],
            'Bullish Acc (%)': ensemble_results[strategy_key]['per_regime_accuracy']['Bullish']
        })
    
    df = pd.DataFrame(results_data)
    df = df.sort_values('Test Acc (%)', ascending=False)
    
    save_path = config.DATA_DIR / "ensemble_results.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved results: {save_path}")
    
    return df


def main():
    """Main evaluation pipeline"""
    
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION")
    print("="*70)
    
    # Top 5 models with engineered features
    top_5_models = [
        'model_2_large_capacity',
        'model_3_attention_pooling',
        'model_4_cnn_transformer',
        'model_5_multiscale',
        'model_1_engineered'
    ]
    
    print(f"\nEnsembling top 5 models:")
    for model_id in top_5_models:
        model_config = config.get_model_config(model_id)
        print(f"  - {model_config['name']} ({model_id})")
    
    # Load data
    print("\nLoading test and validation data...")
    test_data = pd.read_csv(
        config.REGIME_DATA_DIR / "test_labeled_engineered.csv",
        index_col=0, parse_dates=True
    )
    val_data = pd.read_csv(
        config.REGIME_DATA_DIR / "val_labeled_engineered.csv",
        index_col=0, parse_dates=True
    )
    
    batch_size = config.BATCH_SIZE_GPU if torch.cuda.is_available() else config.BATCH_SIZE_CPU
    
    _, val_loader, test_loader, _ = create_dataloaders(
        test_data, val_data, test_data,
        config.ENGINEERED_FEATURES,
        batch_size=batch_size
    )
    
    # Create ensemble
    ensemble = RegimeEnsemble(top_5_models)
    
    # Compute per-regime accuracies on validation set
    ensemble.compute_regime_accuracies(val_loader)
    
    # Evaluate both strategies
    print("\n" + "="*70)
    print("EVALUATING ENSEMBLE STRATEGIES")
    print("="*70)
    
    results_weighted = ensemble.evaluate(test_loader, strategy='weighted')
    results_best = ensemble.evaluate(test_loader, strategy='best_per_regime')
    
    # Package results
    ensemble_results = {
        'model_ids': top_5_models,
        'weighted': results_weighted,
        'best_per_regime': results_best
    }
    
    # Load individual model results
    print("\n" + "="*70)
    print("GENERATING COMPARISON")
    print("="*70)
    
    individual_df = load_individual_results()
    
    if individual_df is not None:
        # Save results
        results_df = save_ensemble_results(ensemble_results, individual_df)
        
        # Print summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print("\n" + results_df.to_string(index=False))
        
        # Plot comparison
        plot_ensemble_comparison(
            individual_df,
            ensemble_results,
            save_path=config.DATA_DIR / "ensemble_comparison.png"
        )
        
        # Final summary
        best_individual = individual_df[individual_df['Model ID'].isin(top_5_models)]['Test Acc (%)'].max()
        weighted_acc = results_weighted['accuracy']
        best_per_regime_acc = results_best['accuracy']
        
        print("\n" + "="*70)
        print("IMPROVEMENT SUMMARY")
        print("="*70)
        print(f"\nBest Individual Model:           {best_individual:.2f}%")
        print(f"Ensemble (Weighted):             {weighted_acc:.2f}% ({weighted_acc - best_individual:+.2f}%)")
        print(f"Ensemble (Best-Per-Regime):      {best_per_regime_acc:.2f}% ({best_per_regime_acc - best_individual:+.2f}%)")
        
        if weighted_acc > best_individual or best_per_regime_acc > best_individual:
            print("\n✓ Ensemble improves over individual models!")
        else:
            print("\n✗ Ensemble does not improve over best individual model")
            print("  (This can happen if models make similar mistakes)")
    
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()