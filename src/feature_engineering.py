"""Feature Engineering: Technical indicators and feature set creation - Feature Expansion"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

class FeatureEngineer:
    
    def __init__(self):
        self.feature_stats = {}
    
    # ========== ORIGINAL FEATURES ==========
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
    
    # ========== NEW: VOLATILITY INDICATORS ==========
    def compute_atr(self, data, period=None):
        """Average True Range"""
        if period is None:
            period = config.ATR_PERIOD
        atr = AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=data['Adj Close'],
            window=period
        )
        return atr.average_true_range()
    
    def compute_realized_volatility(self, data, window):
        """Realized volatility over specific window"""
        return data['log_return'].rolling(window=window).std()
    
    # ========== NEW: TECHNICAL INDICATORS ==========
    def compute_roc(self, data, period=None):
        """Rate of Change"""
        if period is None:
            period = config.ROC_PERIOD
        roc = ((data['Adj Close'] - data['Adj Close'].shift(period)) / data['Adj Close'].shift(period)) * 100
        return roc
    
    def compute_stochastic(self, data, period=None):
        """Stochastic Oscillator"""
        if period is None:
            period = config.STOCHASTIC_PERIOD
        stoch = StochasticOscillator(
            high=data['High'],
            low=data['Low'],
            close=data['Adj Close'],
            window=period
        )
        return stoch.stoch()
    
    def compute_williams_r(self, data, period=None):
        """Williams %R"""
        if period is None:
            period = config.WILLIAMS_R_PERIOD
        williams = WilliamsRIndicator(
            high=data['High'],
            low=data['Low'],
            close=data['Adj Close'],
            lbp=period
        )
        return williams.williams_r()
    
    def compute_mfi(self, data, period=None):
        """Money Flow Index"""
        if period is None:
            period = config.MFI_PERIOD
        mfi = MFIIndicator(
            high=data['High'],
            low=data['Low'],
            close=data['Adj Close'],
            volume=data['Volume'],
            window=period
        )
        return mfi.money_flow_index()
    
    # ========== NEW: SECTOR ROTATION ==========
    def compute_sector_rotation_tech_fin(self, data):
        """Tech vs Financials rotation (XLK/XLF)"""
        if 'XLK' in data.columns and 'XLF' in data.columns:
            return data['XLK'] / data['XLF']
        return pd.Series(index=data.index, dtype=float)
    
    def compute_sector_rotation_def_cyc(self, data):
        """Defensive vs Cyclical (XLV/XLE)"""
        if 'XLV' in data.columns and 'XLE' in data.columns:
            return data['XLV'] / data['XLE']
        return pd.Series(index=data.index, dtype=float)
    
    def compute_sector_breadth(self, data):
        """Average sector performance"""
        sector_cols = [col for col in data.columns if col in config.SECTOR_ETFS]
        if len(sector_cols) > 0:
            sector_returns = data[sector_cols].pct_change()
            return sector_returns.mean(axis=1)
        return pd.Series(index=data.index, dtype=float)
    
    # ========== NEW: INTER-MARKET RATIOS ==========
    def compute_gold_ratio(self, data):
        """Gold to SPY ratio (safe haven indicator)"""
        if 'GLD' in data.columns:
            return data['GLD'] / data['Adj Close']
        return pd.Series(index=data.index, dtype=float)
    
    def compute_bond_ratio(self, data):
        """Bonds to SPY ratio (risk-off indicator)"""
        if 'TLT' in data.columns:
            return data['TLT'] / data['Adj Close']
        return pd.Series(index=data.index, dtype=float)
    
    # ========== NEW: CREDIT SPREADS ==========
    def compute_credit_spread(self, data):
        """Credit spread (BAA10Y - DGS10)"""
        if 'BAA10Y' in data.columns and 'DGS10' in data.columns:
            return data['BAA10Y'] - data['DGS10']
        return pd.Series(index=data.index, dtype=float)
    
    # ========== FEATURE SET CREATION ==========
    def create_baseline_features(self, data):
        """Baseline features - NO calculated features"""
        features = data.copy()
        required = ['log_return', 'volatility_change', 'UNRATE_change', 'CPIAUCSL_change', 
                   'FEDFUNDS_change', 'DGS10', 'T10Y2Y']
        missing = [col for col in required if col not in features.columns]
        if missing:
            print(f"    Warning: Missing baseline features: {missing}")
        return features
    
    def create_engineered_features(self, data):
        """Engineered features - ALL original + new calculated features"""
        features = data.copy()
        
        print("    Computing original technical indicators...")
        features['rsi'] = self.compute_rsi(features)
        features['macd_signal'] = self.compute_macd(features)
        features['bollinger_width'] = self.compute_bollinger_bands(features)
        features['distance_from_ma200'] = self.compute_ma_distance(features)
        
        print("    Computing new volatility indicators...")
        features['atr'] = self.compute_atr(features)
        features['realized_vol_10d'] = self.compute_realized_volatility(features, window=10)
        features['volatility_change_10d'] = features['realized_vol_10d'].diff()
        
        print("    Computing new technical indicators...")
        features['roc'] = self.compute_roc(features)
        features['stochastic'] = self.compute_stochastic(features)
        features['williams_r'] = self.compute_williams_r(features)
        features['mfi'] = self.compute_mfi(features)
        
        print("    Computing sector rotation signals...")
        features['sector_rotation_tech_fin'] = self.compute_sector_rotation_tech_fin(features)
        features['sector_rotation_def_cyc'] = self.compute_sector_rotation_def_cyc(features)
        features['sector_breadth'] = self.compute_sector_breadth(features)
        
        print("    Computing inter-market ratios...")
        features['gold_ratio'] = self.compute_gold_ratio(features)
        features['bond_ratio'] = self.compute_bond_ratio(features)
        
        print("    Computing credit spreads...")
        features['credit_spread'] = self.compute_credit_spread(features)
        
        initial_len = len(features)
        features = features.dropna()
        dropped = initial_len - len(features)
        
        if dropped > 0:
            self.feature_stats['dropped_rows'] = dropped
            print(f"    Dropped {dropped} rows with NaN values")
        
        return features
    
    def process_dataset(self, train, val, test, feature_set='baseline'):
        """Process dataset with specified feature set"""
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
        
        self.feature_stats[feature_set] = {
            'train_shape': train_feat.shape,
            'val_shape': val_feat.shape,
            'test_shape': test_feat.shape,
            'n_features': len(feature_list)
        }
        
        return train_feat, val_feat, test_feat, feature_list
    
    def save_features(self, train, val, test, feature_set='baseline'):
        """Save processed features to disk"""
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
    print("FEATURE ENGINEERING - FEATURE EXPANSION")
    print("="*70)
    
    train = pd.read_csv(config.PROCESSED_DATA_DIR / "train_data.csv", index_col=0, parse_dates=True)
    val = pd.read_csv(config.PROCESSED_DATA_DIR / "val_data.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(config.PROCESSED_DATA_DIR / "test_data.csv", index_col=0, parse_dates=True)
    
    engineer = FeatureEngineer()
    
    results = {}
    for feature_set in ['baseline', 'engineered']:
        print(f"\nProcessing {feature_set.upper()} features...")
        train_feat, val_feat, test_feat, feat_list = engineer.process_dataset(
            train, val, test, feature_set
        )
        engineer.save_features(train_feat, val_feat, test_feat, feature_set)
        results[feature_set] = (train_feat, val_feat, test_feat)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Feature Set':<15} {'Train':<15} {'Val':<15} {'Test':<15} {'Features':<10}")
    print("-"*70)
    for fs in ['baseline', 'engineered']:
        stats = engineer.feature_stats[fs]
        print(f"{fs.capitalize():<15} {str(stats['train_shape']):<15} {str(stats['val_shape']):<15} "
              f"{str(stats['test_shape']):<15} {stats['n_features']:<10}")
    
    full_data = pd.concat([train, val, test])
    print("\nSPY Price Chart")
    engineer.plot_spy_price(full_data, save_path=config.DATA_DIR / "spy_price.png")
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70 + "\n")