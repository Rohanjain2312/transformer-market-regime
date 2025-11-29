"""Binary Classification: Convert 3-class regimes to 2-class (Bearish vs Bullish)"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

import config


def convert_to_binary(data):
    """
    Convert 3-class regime labels to 2-class:
    - Keep Bearish (0) → 0
    - Drop Neutral (1) 
    - Keep Bullish (2) → 1
    
    Args:
        data: DataFrame with 'regime' column (0, 1, 2)
    
    Returns:
        DataFrame with only Bearish and Bullish samples, regime relabeled as 0/1
    """
    # Filter out Neutral (regime == 1)
    binary_data = data[data['regime'] != 1].copy()
    
    # Remap: Bearish (0) stays 0, Bullish (2) becomes 1
    binary_data.loc[binary_data['regime'] == 2, 'regime'] = 1
    
    print(f"  Original: {len(data)} samples")
    print(f"  Removed Neutral: {(data['regime'] == 1).sum()} samples")
    print(f"  Binary: {len(binary_data)} samples")
    print(f"    Bearish (0): {(binary_data['regime'] == 0).sum()}")
    print(f"    Bullish (1): {(binary_data['regime'] == 1).sum()}")
    
    return binary_data


def main():
    """Convert all splits to binary classification"""
    
    print("\n" + "="*70)
    print("CONVERTING TO BINARY CLASSIFICATION (BEARISH VS BULLISH)")
    print("="*70)
    
    # Process both feature sets
    for feature_set in ['baseline', 'engineered']:
        print(f"\n{'='*70}")
        print(f"Processing {feature_set.upper()} features")
        print(f"{'='*70}")
        
        # Load 3-class data
        train = pd.read_csv(
            config.REGIME_DATA_DIR / f"train_labeled_{feature_set}.csv",
            index_col=0, parse_dates=True
        )
        val = pd.read_csv(
            config.REGIME_DATA_DIR / f"val_labeled_{feature_set}.csv",
            index_col=0, parse_dates=True
        )
        test = pd.read_csv(
            config.REGIME_DATA_DIR / f"test_labeled_{feature_set}.csv",
            index_col=0, parse_dates=True
        )
        
        # Convert to binary
        print("\nTrain:")
        train_binary = convert_to_binary(train)
        
        print("\nVal:")
        val_binary = convert_to_binary(val)
        
        print("\nTest:")
        test_binary = convert_to_binary(test)
        
        # Save binary data
        train_binary.to_csv(config.REGIME_DATA_DIR / f"train_labeled_{feature_set}_binary.csv")
        val_binary.to_csv(config.REGIME_DATA_DIR / f"val_labeled_{feature_set}_binary.csv")
        test_binary.to_csv(config.REGIME_DATA_DIR / f"test_labeled_{feature_set}_binary.csv")
        
        print(f"\n✓ Saved binary data to:")
        print(f"  {config.REGIME_DATA_DIR / f'train_labeled_{feature_set}_binary.csv'}")
        print(f"  {config.REGIME_DATA_DIR / f'val_labeled_{feature_set}_binary.csv'}")
        print(f"  {config.REGIME_DATA_DIR / f'test_labeled_{feature_set}_binary.csv'}")
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python train_binary.py --model_id model_2_large_capacity")
    print("2. Expected accuracy: 75-85% (much easier than 3-class)")
    print("\n")


if __name__ == "__main__":
    main()