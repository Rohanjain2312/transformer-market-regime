"""Create SMOTE-Augmented Training Data for Binary Classification"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

import config
from src.smote_timeseries import smote_dataframe


def create_smote_data(feature_set='engineered', target_ratio=0.40):
    """
    Create SMOTE-augmented training data.
    
    Args:
        feature_set: 'baseline' or 'engineered'
        target_ratio: Target ratio for minority class (default 0.40 = 40% Bearish)
    """
    print("\n" + "="*70)
    print("CREATING SMOTE-AUGMENTED TRAINING DATA")
    print("="*70)
    
    # Load original training data
    train_file = config.REGIME_DATA_DIR / f"train_labeled_{feature_set}_binary.csv"
    
    if not train_file.exists():
        print(f"\nERROR: Training file not found: {train_file}")
        print("Please run create_binary_labels.py first")
        return False
    
    print(f"\nLoading training data: {train_file}")
    train_data = pd.read_csv(train_file, index_col=0, parse_dates=True)
    
    print(f"  Shape: {train_data.shape}")
    print(f"  Date range: {train_data.index[0]} to {train_data.index[-1]}")
    
    # Get feature columns (all except 'regime')
    feature_columns = [col for col in train_data.columns if col != 'regime']
    
    print(f"\n  Features: {len(feature_columns)}")
    print(f"  Sample features: {feature_columns[:5]}")
    
    # Check original distribution
    print("\n" + "="*70)
    print("ORIGINAL TRAINING DATA")
    print("="*70)
    
    regime_counts = train_data['regime'].value_counts().sort_index()
    total = len(train_data)
    
    for regime, count in regime_counts.items():
        regime_name = 'Bearish' if regime == 0 else 'Bullish'
        print(f"  {regime_name} ({regime}): {count} samples ({count/total*100:.1f}%)")
    
    # Apply SMOTE
    print("\n" + "="*70)
    print(f"APPLYING SMOTE (Target: {target_ratio*100:.0f}% Bearish)")
    print("="*70)
    
    train_smote = smote_dataframe(
        train_data,
        feature_columns=feature_columns,
        label_column='regime',
        target_ratio=target_ratio,
        k_neighbors=5
    )
    
    # Preserve original index structure but with new range
    # (synthetic samples don't have real dates)
    train_smote.index = pd.date_range(
        start=train_data.index[0],
        periods=len(train_smote),
        freq='D'
    )
    
    # Save augmented data
    output_file = config.REGIME_DATA_DIR / f"train_labeled_{feature_set}_binary_smote.csv"
    train_smote.to_csv(output_file)
    
    print(f"\n" + "="*70)
    print("SMOTE DATA SAVED")
    print("="*70)
    print(f"  File: {output_file}")
    print(f"  Shape: {train_smote.shape}")
    print(f"  Original samples: {len(train_data)}")
    print(f"  New samples: {len(train_smote)}")
    print(f"  Synthetic added: {len(train_smote) - len(train_data)}")
    
    # Verify distribution
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    regime_counts_smote = train_smote['regime'].value_counts().sort_index()
    total_smote = len(train_smote)
    
    for regime, count in regime_counts_smote.items():
        regime_name = 'Bearish' if regime == 0 else 'Bullish'
        print(f"  {regime_name} ({regime}): {count} samples ({count/total_smote*100:.1f}%)")
    
    # Check for NaN values
    nan_count = train_smote.isna().sum().sum()
    if nan_count > 0:
        print(f"\n  WARNING: {nan_count} NaN values detected in augmented data!")
        print(train_smote.isna().sum()[train_smote.isna().sum() > 0])
    else:
        print(f"\n  ✓ No NaN values detected")
    
    return True


def create_all_smote_datasets():
    """Create SMOTE datasets for both baseline and engineered features"""
    
    print("\n" + "="*70)
    print("CREATING SMOTE DATASETS FOR ALL FEATURE SETS")
    print("="*70)
    
    success_count = 0
    
    for feature_set in ['baseline', 'engineered']:
        print(f"\n{'='*70}")
        print(f"Processing: {feature_set.upper()}")
        print(f"{'='*70}")
        
        success = create_smote_data(feature_set=feature_set, target_ratio=0.40)
        
        if success:
            success_count += 1
        else:
            print(f"\n  WARNING: Failed to create SMOTE data for {feature_set}")
    
    print("\n" + "="*70)
    print("SMOTE DATA CREATION COMPLETE")
    print("="*70)
    print(f"  Successfully created: {success_count}/2 datasets")
    
    if success_count == 2:
        print("\n  ✓ All SMOTE datasets ready for training!")
        print("\n  Next step: Run train_binary_smote.py to train models")
    else:
        print("\n  ✗ Some datasets failed. Check errors above.")
    
    print()


if __name__ == "__main__":
    create_all_smote_datasets()
