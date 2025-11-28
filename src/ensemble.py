"""Ensemble Module: Combine multiple trained models for improved regime classification"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.model import create_model


class RegimeEnsemble:
    """
    Ensemble classifier that combines predictions from multiple models.
    Two strategies:
    1. Weighted voting: Weight each model by its per-regime accuracy
    2. Best-per-regime: Route each prediction to the model best at that regime
    """
    
    def __init__(self, model_ids, device=None):
        """
        Args:
            model_ids: List of model IDs to ensemble (e.g., ['model_2_large_capacity', ...])
            device: Device to run models on
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model_ids = model_ids
        self.models = {}
        self.regime_accuracies = {}
        
        print(f"\n{'='*70}")
        print("INITIALIZING ENSEMBLE")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Models to ensemble: {len(model_ids)}")
        
        self._load_models()
        self._extract_regime_accuracies()
    
    def _load_models(self):
        """Load all trained models from checkpoints"""
        print("\nLoading models...")
        
        for model_id in self.model_ids:
            checkpoint_path = config.CHECKPOINT_DIR / model_id / "best.pth"
            
            if not checkpoint_path.exists():
                print(f"  Warning: Checkpoint not found for {model_id}, skipping...")
                continue
            
            # Create model
            model, _ = create_model(model_id, self.device)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models[model_id] = {
                'model': model,
                'config': config.get_model_config(model_id),
                'checkpoint': checkpoint
            }
            
            model_name = self.models[model_id]['config']['name']
            val_acc = checkpoint['val_acc']
            print(f"  ✓ {model_name}: {val_acc:.2f}% val acc")
        
        print(f"\nSuccessfully loaded {len(self.models)} models")
    
    def _extract_regime_accuracies(self):
        """Extract per-regime accuracies from validation results"""
        print("\nExtracting per-regime accuracies...")
        
        for model_id in self.models.keys():
            # We'll compute these during evaluation
            # For now, initialize with uniform weights
            self.regime_accuracies[model_id] = {
                'Bearish': 1.0 / config.NUM_CLASSES,
                'Neutral': 1.0 / config.NUM_CLASSES,
                'Bullish': 1.0 / config.NUM_CLASSES
            }
    
    def compute_regime_accuracies(self, val_loader):
        """
        Compute per-regime accuracy for each model on validation set.
        This should be called before using weighted voting.
        
        Args:
            val_loader: Validation data loader
        """
        print("\nComputing per-regime accuracies on validation set...")
        
        for model_id, model_dict in self.models.items():
            model = model_dict['model']
            model.eval()
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(self.device)
                    outputs = model(x)
                    preds = outputs.argmax(dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Compute per-regime accuracy
            for regime_idx, regime_name in enumerate(config.REGIME_NAMES):
                regime_mask = all_labels == regime_idx
                if regime_mask.sum() > 0:
                    regime_acc = (all_preds[regime_mask] == all_labels[regime_mask]).mean()
                    self.regime_accuracies[model_id][regime_name] = regime_acc
                else:
                    self.regime_accuracies[model_id][regime_name] = 0.0
            
            model_name = model_dict['config']['name']
            print(f"\n  {model_name}:")
            for regime_name in config.REGIME_NAMES:
                acc = self.regime_accuracies[model_id][regime_name] * 100
                print(f"    {regime_name}: {acc:.1f}%")
    
    def predict_weighted(self, x):
        """
        Weighted voting strategy: Weight each model by its per-regime accuracy
        
        Args:
            x: Input tensor (batch, seq_len, features)
        
        Returns:
            predictions: Predicted regime labels (batch,)
            probabilities: Class probabilities (batch, num_classes)
        """
        batch_size = x.size(0)
        weighted_probs = torch.zeros(batch_size, config.NUM_CLASSES).to(self.device)
        
        with torch.no_grad():
            for model_id, model_dict in self.models.items():
                model = model_dict['model']
                
                # Get model predictions
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)  # (batch, num_classes)
                
                # Apply regime-specific weights
                weights = torch.tensor([
                    self.regime_accuracies[model_id]['Bearish'],
                    self.regime_accuracies[model_id]['Neutral'],
                    self.regime_accuracies[model_id]['Bullish']
                ]).to(self.device)
                
                # Weight probabilities
                weighted_probs += probs * weights.unsqueeze(0)
        
        # Average across models
        weighted_probs /= len(self.models)
        
        # Get predictions
        predictions = weighted_probs.argmax(dim=1)
        
        return predictions, weighted_probs
    
    def predict_best_per_regime(self, x):
        """
        Best-per-regime strategy: For each sample, use prediction from model
        that's historically best at the predicted regime.
        
        Args:
            x: Input tensor (batch, seq_len, features)
        
        Returns:
            predictions: Predicted regime labels (batch,)
            probabilities: Class probabilities (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Get predictions from all models
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for model_id, model_dict in self.models.items():
                model = model_dict['model']
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_predictions.append(preds)
                all_probabilities.append(probs)
        
        # Stack: (num_models, batch_size)
        all_predictions = torch.stack(all_predictions)
        all_probabilities = torch.stack(all_probabilities)
        
        # For each sample, find the best model for each possible regime
        final_predictions = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        final_probabilities = torch.zeros(batch_size, config.NUM_CLASSES).to(self.device)
        
        for i in range(batch_size):
            # Get all model predictions for this sample
            sample_preds = all_predictions[:, i]  # (num_models,)
            sample_probs = all_probabilities[:, i, :]  # (num_models, num_classes)
            
            # For each regime, find which model is best at it
            regime_votes = torch.zeros(config.NUM_CLASSES).to(self.device)
            regime_probs = torch.zeros(config.NUM_CLASSES).to(self.device)
            
            for regime_idx, regime_name in enumerate(config.REGIME_NAMES):
                # Find model with highest accuracy for this regime
                best_acc = 0.0
                best_model_idx = 0
                
                for model_idx, model_id in enumerate(self.models.keys()):
                    acc = self.regime_accuracies[model_id][regime_name]
                    if acc > best_acc:
                        best_acc = acc
                        best_model_idx = model_idx
                
                # Use the best model's prediction for this regime
                best_model_pred = sample_preds[best_model_idx]
                best_model_prob = sample_probs[best_model_idx, regime_idx]
                
                # Vote weighted by the model's confidence and accuracy
                regime_votes[regime_idx] = best_model_prob * best_acc
                regime_probs[regime_idx] = best_model_prob
            
            # Final prediction is the regime with highest weighted vote
            final_predictions[i] = regime_votes.argmax()
            final_probabilities[i] = regime_probs
        
        return final_predictions, final_probabilities
    
    def evaluate(self, test_loader, strategy='weighted'):
        """
        Evaluate ensemble on test set
        
        Args:
            test_loader: Test data loader
            strategy: 'weighted' or 'best_per_regime'
        
        Returns:
            results: Dictionary with accuracy, confusion matrix, etc.
        """
        print(f"\nEvaluating ensemble with {strategy} strategy...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for x, y in test_loader:
            x = x.to(self.device)
            
            if strategy == 'weighted':
                preds, probs = self.predict_weighted(x)
            elif strategy == 'best_per_regime':
                preds, probs = self.predict_best_per_regime(x)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        accuracy = (all_preds == all_labels).mean() * 100
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Per-regime accuracy
        per_regime_acc = {}
        for regime_idx, regime_name in enumerate(config.REGIME_NAMES):
            regime_mask = all_labels == regime_idx
            if regime_mask.sum() > 0:
                regime_acc = (all_preds[regime_mask] == all_labels[regime_mask]).mean() * 100
                per_regime_acc[regime_name] = regime_acc
            else:
                per_regime_acc[regime_name] = 0.0
        
        results = {
            'strategy': strategy,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'per_regime_accuracy': per_regime_acc,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        # Print results
        print(f"\n{strategy.upper()} ENSEMBLE RESULTS:")
        print(f"  Test Accuracy: {accuracy:.2f}%")
        print(f"\n  Per-Regime Accuracy:")
        for regime_name, acc in per_regime_acc.items():
            print(f"    {regime_name}: {acc:.2f}%")
        
        return results


if __name__ == "__main__":
    # Test the ensemble module
    print("\n" + "="*70)
    print("ENSEMBLE MODULE TEST")
    print("="*70)
    
    from dataset import create_dataloaders
    
    # Top 5 models (engineered features)
    top_5_models = [
        'model_2_large_capacity',
        'model_3_attention_pooling',
        'model_4_cnn_transformer',
        'model_5_multiscale',
        'model_1_engineered'
    ]
    
    # Load test data
    print("\nLoading test data...")
    test_data = pd.read_csv(
        config.REGIME_DATA_DIR / "test_labeled_engineered.csv",
        index_col=0, parse_dates=True
    )
    val_data = pd.read_csv(
        config.REGIME_DATA_DIR / "val_labeled_engineered.csv",
        index_col=0, parse_dates=True
    )
    
    _, val_loader, test_loader, _ = create_dataloaders(
        test_data, val_data, test_data, 
        config.ENGINEERED_FEATURES,
        batch_size=config.BATCH_SIZE_GPU if torch.cuda.is_available() else config.BATCH_SIZE_CPU
    )
    
    # Create ensemble
    ensemble = RegimeEnsemble(top_5_models)
    
    # Compute regime accuracies
    ensemble.compute_regime_accuracies(val_loader)
    
    # Test both strategies
    results_weighted = ensemble.evaluate(test_loader, strategy='weighted')
    results_best = ensemble.evaluate(test_loader, strategy='best_per_regime')
    
    print("\n" + "="*70)
    print("ENSEMBLE MODULE TEST COMPLETE")
    print("="*70 + "\n")