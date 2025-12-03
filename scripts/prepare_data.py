"""Complete Data Pipeline: Download → Features → Regimes (Binary HMM)"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import config
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.regime_labeling import RegimeLabeler

# Check if running in Colab
try:
    from IPython.display import Image, display
    from google.colab import files
    IN_COLAB = True
except:
    IN_COLAB = False

def plot_spy_with_regimes_and_splits(full_data, train, val, test, save_path):
    """Plot SPY price colored by regime with train/val/test split markers"""
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Plot regimes with colors
    colors = ['red', 'green']  # Bearish, Bullish
    regime_names = config.REGIME_NAMES
    
    for regime_idx in range(config.HMM_N_STATES):
        regime_data = full_data[full_data['regime'] == regime_idx]
        ax.scatter(regime_data.index, regime_data['Adj Close'], 
                  c=colors[regime_idx], label=regime_names[regime_idx],
                  alpha=0.6, s=8)
    
    # Add vertical lines for splits
    train_end = train.index[-1]
    val_end = val.index[-1]
    
    ax.axvline(train_end, color='blue', linestyle='--', linewidth=2, 
               alpha=0.7, label='Train/Val Split')
    ax.axvline(val_end, color='orange', linestyle='--', linewidth=2, 
               alpha=0.7, label='Val/Test Split')
    
    # Add shaded regions for splits
    ax.axvspan(train.index[0], train_end, alpha=0.05, color='blue', label='Train Period')
    ax.axvspan(train_end, val_end, alpha=0.05, color='orange', label='Val Period')
    ax.axvspan(val_end, test.index[-1], alpha=0.05, color='green', label='Test Period')
    
    ax.set_title('SPY Price with Binary Regimes & Train/Val/Test Splits', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path.name}")
    
    if IN_COLAB:
        display(Image(save_path))
    else:
        plt.show()
    
    plt.close()

def print_summary(train, val, test):
    """Print concise summary with split information"""
    print("\n" + "="*80)
    print("DATA PREPARATION SUMMARY")
    print("="*80)
    
    total_samples = len(train) + len(val) + len(test)
    
    # Create summary table
    summary_data = []
    for split_name, split_data in [('Train', train), ('Val', val), ('Test', test)]:
        start_date = split_data.index[0].strftime('%Y-%m-%d')
        end_date = split_data.index[-1].strftime('%Y-%m-%d')
        n_samples = len(split_data)
        pct = n_samples / total_samples * 100
        
        # Regime counts
        regime_counts = split_data['regime'].value_counts().sort_index()
        bearish_count = regime_counts.get(0, 0)
        bullish_count = regime_counts.get(1, 0)
        bearish_pct = bearish_count / n_samples * 100
        bullish_pct = bullish_count / n_samples * 100
        
        summary_data.append({
            'Split': split_name,
            'Start': start_date,
            'End': end_date,
            'Samples': n_samples,
            'Pct': f"{pct:.1f}%",
            'Bearish': f"{bearish_count} ({bearish_pct:.1f}%)",
            'Bullish': f"{bullish_count} ({bullish_pct:.1f}%)"
        })
    
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    
    print(f"\nTotal Samples: {total_samples:,} days")
    print(f"Date Range: {config.START_DATE} to {config.END_DATE}")
    print(f"Features: {len(train.columns)} columns")
    print(f"Classification: Binary (Bearish vs Bullish)")

def main():
    print("\n" + "="*80)
    print("BINARY REGIME CLASSIFICATION - DATA PIPELINE")
    print("="*80)
    
    # Suppress verbose output
    import warnings
    warnings.filterwarnings('ignore')
    
    # Step 1: Data Pipeline
    print("\n[1/4] Downloading SPY and FRED data...", end=" ", flush=True)
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()
    train, val, test = data['train'], data['val'], data['test']
    print("✓")
    
    # Step 2: Feature Engineering
    print("[2/4] Creating baseline & engineered features...", end=" ", flush=True)
    engineer = FeatureEngineer()
    
    train_base, val_base, test_base, _ = engineer.process_dataset(
        train, val, test, 'baseline'
    )
    engineer.save_features(train_base, val_base, test_base, 'baseline')
    
    train_eng, val_eng, test_eng, _ = engineer.process_dataset(
        train, val, test, 'engineered'
    )
    engineer.save_features(train_eng, val_eng, test_eng, 'engineered')
    print("✓")
    
    # Step 3: Regime Labeling (Binary HMM) - Only once, then copy regimes
    print("[3/4] Labeling regimes with binary HMM...", end=" ", flush=True)
    labeler = RegimeLabeler()
    
    # Label baseline data with HMM
    train_labeled_base, val_labeled_base, test_labeled_base = labeler.label_dataset(
        train_base, val_base, test_base
    )
    labeler.save_labeled_data(train_labeled_base, val_labeled_base, test_labeled_base, 'baseline')
    
    # Copy regime labels to engineered data (no need to re-run HMM)
    train_eng['regime'] = train_labeled_base.loc[train_eng.index, 'regime']
    val_eng['regime'] = val_labeled_base.loc[val_eng.index, 'regime']
    test_eng['regime'] = test_labeled_base.loc[test_eng.index, 'regime']
    
    # Drop NaN regimes (from missing indices in engineered data)
    train_eng = train_eng.dropna(subset=['regime'])
    val_eng = val_eng.dropna(subset=['regime'])
    test_eng = test_eng.dropna(subset=['regime'])
    
    # Convert regime to int
    train_eng['regime'] = train_eng['regime'].astype(int)
    val_eng['regime'] = val_eng['regime'].astype(int)
    test_eng['regime'] = test_eng['regime'].astype(int)
    
    labeler.save_labeled_data(train_eng, val_eng, test_eng, 'engineered')
    print("✓")
    
    # Step 4: Visualization (using baseline data - no gaps)
    print("[4/4] Generating visualization...", end=" ", flush=True)
    full_data = pd.concat([train_labeled_base, val_labeled_base, test_labeled_base])
    viz_path = config.DATA_DIR / "spy_regimes_binary.png"
    plot_spy_with_regimes_and_splits(full_data, train_labeled_base, 
                                     val_labeled_base, test_labeled_base, viz_path)
    
    # Print summary (using engineered data for feature count)
    print_summary(train_eng, val_eng, test_eng)
    
    print("\n" + "="*80)
    print("✓ DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\nNext step: python scripts/train_all_models.py")

if __name__ == "__main__":
    main()