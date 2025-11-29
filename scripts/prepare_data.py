"""Prepare Data Pipeline: Download data, create features, and label regimes"""

from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.regime_labeling import RegimeLabeler


def main():
    print("\n" + "="*70)
    print("DATA PREPARATION PIPELINE")
    print("="*70)
    
    # Step 1: Download and process data
    print("\n[1/3] Running data pipeline...")
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()
    
    # Step 2: Feature engineering
    print("\n[2/3] Creating features...")
    engineer = FeatureEngineer()
    
    # Baseline features
    train_base, val_base, test_base, _ = engineer.process_dataset(
        data['train'], data['val'], data['test'], 'baseline'
    )
    engineer.save_features(train_base, val_base, test_base, 'baseline')
    
    # Engineered features
    train_eng, val_eng, test_eng, _ = engineer.process_dataset(
        data['train'], data['val'], data['test'], 'engineered'
    )
    engineer.save_features(train_eng, val_eng, test_eng, 'engineered')
    
    # Step 3: Regime labeling
    print("\n[3/3] Labeling regimes...")
    labeler = RegimeLabeler()
    
    # Label baseline data
    train_l, val_l, test_l = labeler.label_dataset(train_base, val_base, test_base)
    labeler.save_labeled_data(train_l, val_l, test_l, 'baseline')
    
    # Label engineered data
    train_l, val_l, test_l = labeler.label_dataset(train_eng, val_eng, test_eng)
    labeler.save_labeled_data(train_l, val_l, test_l, 'engineered')
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    print("\nNext step: Run create_binary_labels.py")
    print("\n")


if __name__ == "__main__":
    main()
