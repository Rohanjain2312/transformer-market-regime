"""PyTorch Dataset: Time-series windowing for Transformer input"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

class MarketRegimeDataset(Dataset):
    """
    PyTorch Dataset for market regime classification.
    Creates sequences of length SEQUENCE_LENGTH for Transformer input.
    """
    
    def __init__(self, data, feature_list, sequence_length=None):
        """
        Args:
            data: DataFrame with features and 'regime' column
            feature_list: List of feature column names to use
            sequence_length: Number of timesteps to look back
        """
        if sequence_length is None:
            sequence_length = config.SEQUENCE_LENGTH
        
        self.data = data.copy()
        self.feature_list = feature_list
        self.sequence_length = sequence_length
        
        # Drop rows with NaN in features or regime
        initial_len = len(self.data)
        self.data = self.data[feature_list + ['regime']].dropna()
        dropped = initial_len - len(self.data)
        
        if dropped > 0:
            print(f"    Dropped {dropped} rows with NaN values")
        
        # Extract features and labels
        self.features = self.data[feature_list].values.astype(np.float32)
        self.labels = self.data['regime'].values.astype(np.int64)
        
        # Validate
        assert len(self.features) > sequence_length, "Data too short for sequence length"
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        """
        Returns:
            x: Tensor of shape (sequence_length, n_features)
            y: Scalar label for the last timestep
        """
        x = self.features[idx:idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length - 1]
        
        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()
    
    def get_feature_dim(self):
        """Return number of features"""
        return len(self.feature_list)


def create_dataloaders(train_data, val_data, test_data, feature_list, batch_size=None):
    """
    Create PyTorch DataLoaders for train/val/test splits.
    
    Args:
        train_data, val_data, test_data: DataFrames with features and regimes
        feature_list: List of feature column names
        batch_size: Batch size for DataLoader (if None, uses CPU batch size)
    
    Returns:
        train_loader, val_loader, test_loader, n_features
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE_CPU
    
    # Create datasets
    train_dataset = MarketRegimeDataset(train_data, feature_list)
    val_dataset = MarketRegimeDataset(val_data, feature_list)
    test_dataset = MarketRegimeDataset(test_data, feature_list)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.get_feature_dim()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PYTORCH DATASET TEST")
    print("="*70)
    
    # Test with baseline features
    print("\nLoading baseline data...")
    train = pd.read_csv(config.REGIME_DATA_DIR / "train_labeled_baseline.csv", 
                        index_col=0, parse_dates=True)
    val = pd.read_csv(config.REGIME_DATA_DIR / "val_labeled_baseline.csv", 
                      index_col=0, parse_dates=True)
    test = pd.read_csv(config.REGIME_DATA_DIR / "test_labeled_baseline.csv", 
                       index_col=0, parse_dates=True)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, n_features = create_dataloaders(
        train, val, test, config.BASELINE_FEATURES
    )
    
    # Summary
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Sequence Length:    {config.SEQUENCE_LENGTH}")
    print(f"Batch Size:         {config.BATCH_SIZE_CPU}")
    print(f"Features:           {n_features}")
    print(f"Number of Classes:  {config.NUM_CLASSES}")
    print(f"\nTrain Batches:      {len(train_loader)}")
    print(f"Val Batches:        {len(val_loader)}")
    print(f"Test Batches:       {len(test_loader)}")
    
    # Test batch
    print("\n" + "="*70)
    print("SAMPLE BATCH")
    print("="*70)
    x_batch, y_batch = next(iter(train_loader))
    print(f"Input shape:        {x_batch.shape}  (batch, seq_len, features)")
    print(f"Label shape:        {y_batch.shape}  (batch,)")
    print(f"Input dtype:        {x_batch.dtype}")
    print(f"Label dtype:        {y_batch.dtype}")
    print(f"Label values:       {torch.unique(y_batch).tolist()}")
    
    # Check for NaN
    assert not torch.isnan(x_batch).any(), "NaN detected in batch"
    print("\nValidation:         PASS (no NaN values)")
    
    print("\n" + "="*70)
    print("DATASET TEST COMPLETE")
    print("="*70 + "\n")