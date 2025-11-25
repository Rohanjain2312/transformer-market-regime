"""Regime Labeling: HMM-based market regime classification"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

class RegimeLabeler:
    
    def __init__(self):
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.regime_mapping = {}
    
    def prepare_hmm_features(self, data):
        """Extract and scale features for HMM training"""
        features = data[config.HMM_FEATURES].copy()
        features = features.dropna()
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled, features.index
    
    def train_hmm(self, data, fit_on_train_only=None):
        """Train Gaussian HMM on returns and volatility"""
        if fit_on_train_only is None:
            fit_on_train_only = config.FIT_HMM_ON_TRAIN_ONLY
        
        print(f"\nTraining HMM...")
        print(f"    States: {config.HMM_N_STATES}")
        print(f"    Features: {config.HMM_FEATURES}")
        print(f"    Fit on train only: {fit_on_train_only}")
        
        # Prepare features
        features_scaled, feature_index = self.prepare_hmm_features(data)
        
        # Train HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=config.HMM_N_STATES,
            covariance_type="full",
            n_iter=config.HMM_N_ITER,
            random_state=config.HMM_RANDOM_STATE
        )
        
        self.hmm_model.fit(features_scaled)
        print(f"    HMM training complete ({len(features_scaled)} observations)")
        
        return self.hmm_model
    
    def predict_regimes(self, data):
        """Predict hidden states for given data"""
        features_scaled, feature_index = self.prepare_hmm_features(data)
        hidden_states = self.hmm_model.predict(features_scaled)
        
        # Create series with original index
        regime_series = pd.Series(hidden_states, index=feature_index, name='regime_state')
        return regime_series
    
    def map_states_to_regimes(self, data, states):
        """Map HMM states to regime names based on mean returns"""
        print("\nMapping states to regime names...")
        
        # Calculate mean return for each state
        state_returns = {}
        for state in range(config.HMM_N_STATES):
            state_mask = states == state
            state_data = data.loc[state_mask, 'log_return']
            mean_return = state_data.mean()
            state_returns[state] = mean_return
            print(f"    State {state}: mean return = {mean_return:.6f}")
        
        # Sort states by mean return (ascending)
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        
        # Map to regime names (Bearish, Neutral, Bullish)
        self.regime_mapping = {
            state: idx for idx, (state, _) in enumerate(sorted_states)
        }
        
        print("\nRegime mapping:")
        for state, regime_idx in self.regime_mapping.items():
            regime_name = config.REGIME_NAMES[regime_idx]
            print(f"    State {state} -> {regime_name} (label {regime_idx})")
        
        # Apply mapping
        mapped_regimes = states.map(self.regime_mapping)
        return mapped_regimes
    
    def label_dataset(self, train, val, test, fit_on_train_only=None):
        """Complete regime labeling pipeline"""
        if fit_on_train_only is None:
            fit_on_train_only = config.FIT_HMM_ON_TRAIN_ONLY
        
        print("\n" + "="*70)
        print("REGIME LABELING")
        print("="*70)
        
        # Train HMM
        if fit_on_train_only:
            print("\nStrategy: Fit HMM on training data only")
            self.train_hmm(train, fit_on_train_only=True)
        else:
            print("\nStrategy: Fit HMM on full dataset")
            full_data = pd.concat([train, val, test])
            self.train_hmm(full_data, fit_on_train_only=False)
        
        # Predict regimes for all splits
        print("\nPredicting regimes...")
        train_states = self.predict_regimes(train)
        val_states = self.predict_regimes(val)
        test_states = self.predict_regimes(test)
        
        # Map states to regime names
        all_states = pd.concat([train_states, val_states, test_states])
        all_data = pd.concat([train, val, test])
        mapped_regimes = self.map_states_to_regimes(all_data, all_states)
        
        # Split back into train/val/test
        train_regimes = mapped_regimes.loc[train.index]
        val_regimes = mapped_regimes.loc[val.index]
        test_regimes = mapped_regimes.loc[test.index]
        
        # Add regimes to dataframes
        train['regime'] = train_regimes
        val['regime'] = val_regimes
        test['regime'] = test_regimes
        
        # Summary statistics
        print("\n" + "="*70)
        print("REGIME DISTRIBUTION")
        print("="*70)
        
        for split_name, split_data in [('Train', train), ('Val', val), ('Test', test)]:
            print(f"\n{split_name}:")
            regime_counts = split_data['regime'].value_counts().sort_index()
            for regime_idx, count in regime_counts.items():
                regime_name = config.REGIME_NAMES[regime_idx]
                pct = count / len(split_data) * 100
                print(f"    {regime_name}: {count} ({pct:.1f}%)")
        
        return train, val, test
    
    def save_labeled_data(self, train, val, test, feature_set='baseline'):
        """Save data with regime labels"""
        suffix = f"_{feature_set}"
        train.to_csv(config.REGIME_DATA_DIR / f"train_labeled{suffix}.csv")
        val.to_csv(config.REGIME_DATA_DIR / f"val_labeled{suffix}.csv")
        test.to_csv(config.REGIME_DATA_DIR / f"test_labeled{suffix}.csv")
        print(f"\nSaved labeled data to {config.REGIME_DATA_DIR}")
    
    def plot_regimes(self, data, save_path=None):
        """Plot SPY price colored by regime"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Define colors for regimes
        colors = ['red', 'gray', 'green']
        regime_names = config.REGIME_NAMES
        
        # Plot each regime separately
        for regime_idx in range(config.HMM_N_STATES):
            regime_data = data[data['regime'] == regime_idx]
            ax.scatter(regime_data.index, regime_data['Adj Close'], 
                      c=colors[regime_idx], label=regime_names[regime_idx],
                      alpha=0.6, s=10)
        
        ax.set_title('Market Regimes - SPY Price', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {save_path}")
        plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REGIME LABELING TEST")
    print("="*70)
    
    # Load data (use baseline features for HMM)
    train = pd.read_csv(config.PROCESSED_DATA_DIR / "train_features_baseline.csv", 
                        index_col=0, parse_dates=True)
    val = pd.read_csv(config.PROCESSED_DATA_DIR / "val_features_baseline.csv", 
                      index_col=0, parse_dates=True)
    test = pd.read_csv(config.PROCESSED_DATA_DIR / "test_features_baseline.csv", 
                       index_col=0, parse_dates=True)
    
    # Label regimes
    labeler = RegimeLabeler()
    train_labeled, val_labeled, test_labeled = labeler.label_dataset(train, val, test)
    
    # Save labeled data for both feature sets
    for feature_set in ['baseline', 'engineered']:
        labeler.save_labeled_data(train_labeled, val_labeled, test_labeled, feature_set)
    
    # Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)
    
    full_data = pd.concat([train_labeled, val_labeled, test_labeled])
    print("\nRegime Visualization")
    labeler.plot_regimes(full_data, save_path=config.DATA_DIR / "market_regimes.png")
    
    print("\n" + "="*70)
    print("REGIME LABELING COMPLETE")
    print("="*70 + "\n")