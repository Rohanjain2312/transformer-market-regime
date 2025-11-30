"""SMOTE for Time Series: Generate synthetic minority class samples"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class TimeSeriesSMOTE:
    """
    SMOTE (Synthetic Minority Over-sampling Technique) for time series data.
    
    Generates synthetic samples by interpolating between similar minority class sequences.
    Preserves temporal structure by treating each time window as a whole unit.
    """
    
    def __init__(self, k_neighbors=5, random_state=42):
        """
        Args:
            k_neighbors: Number of nearest neighbors to use for interpolation
            random_state: Random seed for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        np.random.seed(random_state)
    
    def fit_resample(self, X, y, target_ratio=0.5):
        """
        Generate synthetic samples to balance the dataset.
        
        Args:
            X: Feature array of shape (n_samples, n_features) or (n_samples, seq_len, n_features)
            y: Labels of shape (n_samples,)
            target_ratio: Desired ratio of minority class (0.5 = 50% minority)
        
        Returns:
            X_resampled, y_resampled: Balanced dataset
        """
        # Identify minority and majority classes
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        minority_count = np.min(counts)
        majority_count = np.max(counts)
        
        print(f"\nOriginal class distribution:")
        print(f"  Minority ({minority_class}): {minority_count} samples")
        print(f"  Majority ({majority_class}): {majority_count} samples")
        print(f"  Ratio: {minority_count / (minority_count + majority_count):.1%}")
        
        # Calculate how many synthetic samples to generate
        total_desired = majority_count / (1 - target_ratio)
        n_synthetic = int(total_desired * target_ratio - minority_count)
        
        if n_synthetic <= 0:
            print(f"  No synthetic samples needed (already at {target_ratio:.1%})")
            return X, y
        
        print(f"\nGenerating {n_synthetic} synthetic minority samples...")
        print(f"  Target ratio: {target_ratio:.1%}")
        
        # Get minority class samples
        minority_indices = np.where(y == minority_class)[0]
        X_minority = X[minority_indices]
        
        # Reshape if 3D (sequences)
        is_sequence = len(X.shape) == 3
        if is_sequence:
            original_shape = X.shape
            # Flatten sequences for nearest neighbor search
            X_minority_flat = X_minority.reshape(len(X_minority), -1)
        else:
            X_minority_flat = X_minority
        
        # Find k nearest neighbors for each minority sample
        knn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_minority)))
        knn.fit(X_minority_flat)
        
        # Generate synthetic samples
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # Randomly select a minority sample
            idx = np.random.randint(0, len(X_minority))
            sample = X_minority_flat[idx]
            
            # Find its k nearest neighbors
            _, neighbor_indices = knn.kneighbors([sample])
            
            # Remove the sample itself (first neighbor)
            neighbor_indices = neighbor_indices[0][1:]
            
            if len(neighbor_indices) == 0:
                # If no neighbors, just duplicate the sample with small noise
                if is_sequence:
                    synthetic = X_minority[idx] + np.random.normal(0, 0.01, X_minority[idx].shape)
                else:
                    synthetic = sample + np.random.normal(0, 0.01, sample.shape)
            else:
                # Randomly select one of the k neighbors
                neighbor_idx = neighbor_indices[np.random.randint(0, len(neighbor_indices))]
                neighbor = X_minority_flat[neighbor_idx]
                
                # Generate synthetic sample by linear interpolation
                alpha = np.random.random()  # Random weight between 0 and 1
                
                if is_sequence:
                    synthetic = alpha * X_minority[idx] + (1 - alpha) * X_minority[neighbor_idx]
                else:
                    synthetic = alpha * sample + (1 - alpha) * neighbor
            
            synthetic_samples.append(synthetic)
        
        # Convert to array
        synthetic_samples = np.array(synthetic_samples)
        
        # Reshape back if sequences
        if is_sequence:
            synthetic_samples = synthetic_samples.reshape(-1, original_shape[1], original_shape[2])
        
        # Combine original and synthetic samples
        X_resampled = np.vstack([X, synthetic_samples])
        y_synthetic = np.full(n_synthetic, minority_class)
        y_resampled = np.concatenate([y, y_synthetic])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled[shuffle_idx]
        y_resampled = y_resampled[shuffle_idx]
        
        # Print results
        new_minority_count = minority_count + n_synthetic
        new_total = len(y_resampled)
        print(f"\nNew class distribution:")
        print(f"  Minority ({minority_class}): {new_minority_count} samples")
        print(f"  Majority ({majority_class}): {majority_count} samples")
        print(f"  Ratio: {new_minority_count / new_total:.1%}")
        print(f"  Total samples: {len(y)} → {new_total} (+{n_synthetic})")
        
        return X_resampled, y_resampled


def smote_dataframe(df, feature_columns, label_column='regime', target_ratio=0.5, k_neighbors=5):
    """
    Apply SMOTE to a pandas DataFrame.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        label_column: Name of label column
        target_ratio: Desired minority class ratio
        k_neighbors: Number of neighbors for SMOTE
    
    Returns:
        Resampled DataFrame
    """
    # Extract features and labels
    X = df[feature_columns].values
    y = df[label_column].values
    
    # Apply SMOTE
    smote = TimeSeriesSMOTE(k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y, target_ratio=target_ratio)
    
    # Create new DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
    df_resampled[label_column] = y_resampled
    
    # Copy index metadata (use range index for synthetic samples)
    df_resampled.index = pd.RangeIndex(len(df_resampled))
    
    return df_resampled


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TIME-SERIES SMOTE TEST")
    print("="*70)
    
    # Test with synthetic data
    np.random.seed(42)
    
    # Generate imbalanced dataset
    n_majority = 100
    n_minority = 30
    n_features = 5
    
    X_majority = np.random.randn(n_majority, n_features)
    X_minority = np.random.randn(n_minority, n_features) + 2  # Different distribution
    
    X = np.vstack([X_majority, X_minority])
    y = np.array([1] * n_majority + [0] * n_minority)
    
    print(f"\nTest Data:")
    print(f"  Shape: {X.shape}")
    print(f"  Original ratio: {n_minority / (n_majority + n_minority):.1%} minority")
    
    # Apply SMOTE
    smote = TimeSeriesSMOTE(k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y, target_ratio=0.5)
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")
