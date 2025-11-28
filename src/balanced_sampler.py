"""Balanced Sampler: Ensures equal representation of regimes in each batch"""

import torch
from torch.utils.data import Sampler
import numpy as np


class RegimeBalancedBatchSampler(Sampler):
    """
    Samples batches such that each batch has approximately equal number
    of samples from each regime class.
    
    This helps address class imbalance during training by ensuring
    the model sees all regimes equally within each batch.
    
    Args:
        labels: Array of regime labels (0, 1, 2)
        batch_size: Total batch size
        drop_last: Whether to drop last incomplete batch
    """
    
    def __init__(self, labels, batch_size, drop_last=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Get unique classes
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)
        
        # Compute samples per class per batch
        self.samples_per_class = batch_size // self.num_classes
        self.remainder = batch_size % self.num_classes
        
        # Get indices for each class
        self.class_indices = {}
        for c in self.classes:
            self.class_indices[c] = np.where(self.labels == c)[0].tolist()
        
        # Compute number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_size // self.samples_per_class
        
        if not self.drop_last:
            self.num_batches += 1
        
        print(f"\nRegimeBalancedBatchSampler initialized:")
        print(f"  Total samples: {len(self.labels)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Samples per class per batch: {self.samples_per_class}")
        print(f"  Number of batches: {self.num_batches}")
        for c in self.classes:
            print(f"  Class {c}: {len(self.class_indices[c])} samples")
    
    def __iter__(self):
        # Shuffle indices for each class
        shuffled_indices = {}
        for c in self.classes:
            indices = self.class_indices[c].copy()
            np.random.shuffle(indices)
            shuffled_indices[c] = indices
        
        # Generate batches
        batches = []
        for batch_idx in range(self.num_batches):
            batch = []
            
            # Sample from each class
            for c in self.classes:
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                
                class_indices = shuffled_indices[c]
                
                # Handle wrapping if we run out of samples
                if end_idx <= len(class_indices):
                    batch.extend(class_indices[start_idx:end_idx])
                else:
                    # Wrap around
                    batch.extend(class_indices[start_idx:])
                    needed = end_idx - len(class_indices)
                    batch.extend(class_indices[:needed])
            
            # Add remainder samples from first classes if needed
            if self.remainder > 0 and len(batch) < self.batch_size:
                for c in self.classes[:self.remainder]:
                    start_idx = batch_idx * self.samples_per_class
                    if start_idx < len(shuffled_indices[c]):
                        batch.append(shuffled_indices[c][start_idx])
            
            # Shuffle batch to avoid systematic ordering
            np.random.shuffle(batch)
            
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        # Shuffle batch order
        np.random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        return self.num_batches


class WeightedRegimeSampler(Sampler):
    """
    Samples with probability proportional to inverse class frequency.
    Minority classes are sampled more frequently.
    
    Args:
        labels: Array of regime labels
        num_samples: Number of samples to draw per epoch
    """
    
    def __init__(self, labels, num_samples=None):
        self.labels = np.array(labels)
        
        if num_samples is None:
            num_samples = len(labels)
        self.num_samples = num_samples
        
        # Compute class weights (inverse frequency)
        classes, class_counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        
        self.weights = np.zeros(len(self.labels))
        for c, count in zip(classes, class_counts):
            class_weight = total / (len(classes) * count)
            self.weights[self.labels == c] = class_weight
        
        # Normalize
        self.weights = self.weights / self.weights.sum()
        
        print(f"\nWeightedRegimeSampler initialized:")
        print(f"  Total samples: {len(self.labels)}")
        print(f"  Samples per epoch: {self.num_samples}")
        for c, count in zip(classes, class_counts):
            weight = total / (len(classes) * count)
            print(f"  Class {c}: {count} samples, weight: {weight:.3f}")
    
    def __iter__(self):
        # Sample with replacement according to weights
        indices = np.random.choice(
            len(self.labels),
            size=self.num_samples,
            replace=True,
            p=self.weights
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    # Test samplers
    print("\n" + "="*70)
    print("SAMPLER TEST")
    print("="*70)
    
    # Create imbalanced dataset
    labels = np.concatenate([
        np.zeros(100),   # Bearish: 100 samples
        np.ones(50),     # Neutral: 50 samples (minority)
        np.full(120, 2)  # Bullish: 120 samples
    ])
    
    print(f"\nDataset: {len(labels)} samples")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {int(u)}: {c} samples ({c/len(labels)*100:.1f}%)")
    
    # Test balanced batch sampler
    print("\n" + "="*70)
    print("TEST 1: RegimeBalancedBatchSampler")
    print("="*70)
    
    batch_size = 30  # 10 per class
    sampler = RegimeBalancedBatchSampler(labels, batch_size, drop_last=True)
    
    # Sample a few batches
    print(f"\nSampling 3 batches:")
    for i, batch_indices in enumerate(sampler):
        if i >= 3:
            break
        batch_labels = labels[batch_indices]
        unique, counts = np.unique(batch_labels, return_counts=True)
        print(f"\n  Batch {i+1}:")
        print(f"    Size: {len(batch_indices)}")
        for u, c in zip(unique, counts):
            print(f"    Class {int(u)}: {c} samples")
    
    # Test weighted sampler
    print("\n" + "="*70)
    print("TEST 2: WeightedRegimeSampler")
    print("="*70)
    
    sampler = WeightedRegimeSampler(labels, num_samples=300)
    
    # Sample and check distribution
    sampled_indices = list(sampler)
    sampled_labels = labels[sampled_indices]
    unique, counts = np.unique(sampled_labels, return_counts=True)
    
    print(f"\nSampled {len(sampled_labels)} samples:")
    for u, c in zip(unique, counts):
        print(f"  Class {int(u)}: {c} samples ({c/len(sampled_labels)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("SAMPLER TEST COMPLETE")
    print("="*70 + "\n")