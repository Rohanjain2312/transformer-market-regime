"""Data Preparation: Complete pipeline from raw data to binary labels"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.regime_labeling import RegimeLabeler
import pandas as pd
import config


def main():
    print("\n" + "="*70)
    print("DATA PREPARATION PIPELINE")
    print("="*70)
    
    # Step 1: Download and process data
    print("\n[STEP 1/4] Data Pipeline (download SPY, FRED, etc.)")
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()
    
    # Step 2: Feature engineering (baseline + engineered)
    print("\n[STEP 2/4] Feature Engineering")
    engineer = FeatureEngineer()
    
    print("\n  Creating BASELINE features...")
    train_base, val_base, test_base, _ = engineer.process_dataset(
        data['train'], data['val'], data['test'], 'baseline'
    )
    engineer.save_features(train_base, val_base, test_base, 'baseline')
    
    print("\n  Creating ENGINEERED features...")
    train_eng, val_eng, test_eng, _ = engineer.process_dataset(
        data['train'], data['val'], data['test'], 'engineered'
    )
    engineer.save_features(train_eng, val_eng, test_eng, 'engineered')
    
    # Step 3: Regime labeling (HMM produces 3 classes)
    print("\n[STEP 3/4] Regime Labeling (HMM - 3 classes)")
    labeler = RegimeLabeler()
    
    # Label baseline
    train_l, val_l, test_l = labeler.label_dataset(train_base, val_base, test_base)
    labeler.save_labeled_data(train_l, val_l, test_l, 'baseline')
    
    # Label engineered (reuse same HMM)
    train_l, val_l, test_l = labeler.label_dataset(train_eng, val_eng, test_eng)
    labeler.save_labeled_data(train_l, val_l, test_l, 'engineered')
    
    # Step 4: Convert to binary (drop Neutral class)
    print("\n[STEP 4/4] Converting to Binary Labels (Bearish vs Bullish)")
    
    for feature_set in ['baseline', 'engineered']:
        print(f"\n  Processing {feature_set}...")
        
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
        
        # Convert: Keep Bearish (0), Drop Neutral (1), Remap Bullish (2→1)
        train_binary = train[train['regime'] != 1].copy()
        val_binary = val[val['regime'] != 1].copy()
        test_binary = test[test['regime'] != 1].copy()
        
        train_binary.loc[train_binary['regime'] == 2, 'regime'] = 1
        val_binary.loc[val_binary['regime'] == 2, 'regime'] = 1
        test_binary.loc[test_binary['regime'] == 2, 'regime'] = 1
        
        # Save binary data
        train_binary.to_csv(config.REGIME_DATA_DIR / f"train_labeled_{feature_set}_binary.csv")
        val_binary.to_csv(config.REGIME_DATA_DIR / f"val_labeled_{feature_set}_binary.csv")
        test_binary.to_csv(config.REGIME_DATA_DIR / f"test_labeled_{feature_set}_binary.csv")
        
        bearish_count = (train_binary['regime']==0).sum()
        bullish_count = (train_binary['regime']==1).sum()
        print(f"    Train: {len(train)} → {len(train_binary)} (Bearish: {bearish_count}, Bullish: {bullish_count})")
        print(f"    Val:   {len(val)} → {len(val_binary)}")
        print(f"    Test:  {len(test)} → {len(test_binary)}")
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Train models: python scripts/train_binary_focal.py --model_id model_5_multiscale")
    print("  2. Compare all:  python scripts/compare_binary_models.py")
    print("  3. Run demo:     python scripts/demo_scenario.py")
    print("\n")


if __name__ == "__main__":
    main()