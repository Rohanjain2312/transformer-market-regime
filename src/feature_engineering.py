"""Feature Engineering: Technical indicators and feature set creation"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

class FeatureEngineer:
    
    def __init__(self):
        self.feature_stats = {}
    
    def compute_rsi(self, data, period=None):
        if period is None:
            period = config.RSI_PERIOD
        rsi = RSIIndicator(close=data['Adj Close'], window=period)
        return rsi.rsi()
    
    def compute_macd(self, data):
        macd = MACD(
            close=data['Adj Close'],
            window_slow=config.MACD_SLOW,
            window_fast=config.MACD_FAST,
            window_sign=config.MACD_SIGNAL
        )
        return macd.macd_signal()
    
    def compute_bollinger_bands(self, data):
        bb = BollingerBands(
            close=data['Adj Close'],
            window=config.BOLLINGER_PERIOD,
            window_dev=config.BOLLINGER_STD
        )
        bb_high = bb.bollinger_hband()
        bb_low = bb.bollinger_lband()
        bb_width = (bb_high - bb_low) / data['Adj Close']
        return bb_width
    
    def compute_ma_distance(self, data, period=None):
        if period is None:
            period = config.MA_LONG_PERIOD
        ma = data['Adj Close'].rolling(window=period).mean()
        distance = (data['Adj Close'] - ma) / ma
        return distance
    
    def create_baseline_features(self, data):
        features = data.copy()
        required = ['log_return', 'volatility_change', 'UNRATE_change', 'CPIAUCSL_change', 
                   'FEDFUNDS_change', 'DGS10', 'T10Y2Y']
        missing = [col for col in required if col not in features.columns]
        if missing:
            print(f"    Warning: Missing baseline features: {missing}")
        return features
    
    def create_engineered_features(self, data):
        features = data.copy()
        features['rsi'] = self.compute_rsi(features)
        features['macd_signal'] = self.compute_macd(features)
        features['bollinger_width'] = self.compute_bollinger_bands(features)
        features['distance_from_ma200'] = self.compute_ma_distance(features)
        
        initial_len = len(features)
        features = features.dropna()
        dropped = initial_len - len(features)
        
        if dropped > 0:
            self.feature_stats['dropped_rows'] = dropped
        
        return features
    
    def process_dataset(self, train, val, test, feature_set='baseline'):
        if feature_set == 'baseline':
            train_feat = self.create_baseline_features(train)
            val_feat = self.create_baseline_features(val)
            test_feat = self.create_baseline_features(test)
            feature_list = config.BASELINE_FEATURES
        elif feature_set == 'engineered':
            train_feat = self.create_engineered_features(train)
            val_feat = self.create_engineered_features(val)
            test_feat = self.create_engineered_features(test)
            feature_list = config.ENGINEERED_FEATURES
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
        
        # Store stats
        self.feature_stats[feature_set] = {
            'train_shape': train_feat.shape,
            'val_shape': val_feat.shape,
            'test_shape': test_feat.shape,
            'n_features': len(feature_list)
        }
        
        return train_feat, val_feat, test_feat, feature_list
    
    def save_features(self, train, val, test, feature_set='baseline'):
        suffix = f"_{feature_set}"
        train.to_csv(config.PROCESSED_DATA_DIR / f"train_features{suffix}.csv")
        val.to_csv(config.PROCESSED_DATA_DIR / f"val_features{suffix}.csv")
        test.to_csv(config.PROCESSED_DATA_DIR / f"test_features{suffix}.csv")
    
    def plot_spy_price(self, data, save_path=None):
        """Simple plot of SPY price over entire time period"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(data.index, data['Adj Close'], color='black', linewidth=1.5)
        ax.set_title('SPY Price - Complete Time Series', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add start and end price annotations
        start_price = data['Adj Close'].iloc[0]
        end_price = data['Adj Close'].iloc[-1]
        total_return = (end_price / start_price - 1) * 100
        
        ax.text(0.02, 0.95, f'Start: ${start_price:.2f}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.88, f'End: ${end_price:.2f}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.81, f'Total Return: {total_return:.1f}%', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {save_path}")
        plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    # Load processed data
    train = pd.read_csv(config.PROCESSED_DATA_DIR / "train_data.csv", index_col=0, parse_dates=True)
    val = pd.read_csv(config.PROCESSED_DATA_DIR / "val_data.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(config.PROCESSED_DATA_DIR / "test_data.csv", index_col=0, parse_dates=True)
    
    engineer = FeatureEngineer()
    
    # Process both feature sets
    results = {}
    for feature_set in ['baseline', 'engineered']:
        print(f"\nProcessing {feature_set.upper()} features...")
        train_feat, val_feat, test_feat, feat_list = engineer.process_dataset(
            train, val, test, feature_set
        )
        engineer.save_features(train_feat, val_feat, test_feat, feature_set)
        results[feature_set] = (train_feat, val_feat, test_feat)
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Feature Set':<15} {'Train':<15} {'Val':<15} {'Test':<15} {'Features':<10}")
    print("-"*70)
    for fs in ['baseline', 'engineered']:
        stats = engineer.feature_stats[fs]
        print(f"{fs.capitalize():<15} {str(stats['train_shape']):<15} {str(stats['val_shape']):<15} "
              f"{str(stats['test_shape']):<15} {stats['n_features']:<10}")
    
    # Generate visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)
    
    # Combine all data for full time series
    full_data = pd.concat([train, val, test])
    
    print("\nSPY Price Chart")
    engineer.plot_spy_price(
        full_data,
        save_path=config.DATA_DIR / "spy_price.png"
    )
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70 + "\n")