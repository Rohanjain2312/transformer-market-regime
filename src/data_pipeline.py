"""Data Pipeline: Acquisition, merging, and preprocessing with ALFRED protocol"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

class DataPipeline:
    
    def __init__(self):
        self.fred = Fred(api_key=config.FRED_API_KEY)
        
    def download_spy_data(self):
        """Download SPY price data"""
        spy = yf.download(config.TICKER, start=config.START_DATE, end=config.END_DATE, 
                         progress=False, auto_adjust=False)
        
        # Handle multi-level columns if present
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        
        spy['log_return'] = np.log(spy['Adj Close'] / spy['Adj Close'].shift(1))
        spy['volatility'] = spy['log_return'].rolling(window=20).std()
        spy = spy.dropna()
        spy.to_csv(config.RAW_DATA_DIR / "spy_data.csv")
        return spy
    
    def download_fred_daily_data(self):
        """Download FRED daily indicators"""
        daily_data = {}
        for indicator in config.FRED_DAILY_INDICATORS:
            try:
                series = self.fred.get_series(indicator, observation_start=config.START_DATE, 
                                              observation_end=config.END_DATE)
                daily_data[indicator] = series
            except Exception as e:
                daily_data[indicator] = pd.Series(dtype=float)
        return pd.DataFrame(daily_data)
    
    def download_fred_monthly_data(self):
        """Download FRED monthly indicators"""
        monthly_data = {}
        for indicator in config.FRED_MONTHLY_INDICATORS:
            if config.USE_ALFRED_VINTAGES and indicator in config.ALFRED_INDICATORS:
                continue
            try:
                series = self.fred.get_series(indicator, observation_start=config.START_DATE, 
                                              observation_end=config.END_DATE)
                monthly_data[indicator] = series
            except Exception as e:
                monthly_data[indicator] = pd.Series(dtype=float)
        return pd.DataFrame(monthly_data)
    
    def download_alfred_data(self):
        """Download ALFRED vintage data"""
        if not config.USE_ALFRED_VINTAGES:
            return pd.DataFrame()
        
        alfred_data = {}
        for indicator in config.ALFRED_INDICATORS:
            try:
                series = pdr.get_data_fred(indicator, start=config.START_DATE, end=config.END_DATE)
                alfred_data[indicator] = series[indicator]
            except Exception as e:
                try:
                    series = self.fred.get_series(indicator, observation_start=config.START_DATE, 
                                                  observation_end=config.END_DATE)
                    alfred_data[indicator] = series
                except:
                    alfred_data[indicator] = pd.Series(dtype=float)
        return pd.DataFrame(alfred_data)
    
    def apply_publication_lags(self, monthly_data):
        """Apply publication lags to avoid look-ahead bias"""
        lagged_data = monthly_data.copy()
        for indicator in lagged_data.columns:
            if indicator in config.PUBLICATION_LAGS:
                lag_days = config.PUBLICATION_LAGS[indicator]
                lagged_data[indicator] = lagged_data[indicator].shift(1)
                lagged_data.index = lagged_data.index + pd.DateOffset(days=lag_days)
        return lagged_data
    
    def merge_to_daily_frequency(self, spy_data, daily_data, monthly_data):
        """Merge all data to daily frequency"""
        merged = spy_data[['Adj Close', 'log_return', 'volatility']].copy()
        
        if not daily_data.empty:
            merged = merged.join(daily_data, how='left')
        
        if not monthly_data.empty:
            merged = merged.join(monthly_data, how='left')
            for col in monthly_data.columns:
                merged[col] = merged[col].fillna(method='ffill')
        
        merged = merged.dropna()
        return merged
    
    def compute_macro_changes(self, data):
        """Compute changes in macroeconomic indicators"""
        macro_indicators = config.FRED_MONTHLY_INDICATORS + (config.ALFRED_INDICATORS if config.USE_ALFRED_VINTAGES else [])
        
        for indicator in macro_indicators:
            if indicator in data.columns:
                change_col = f"{indicator}_change"
                if 'CPI' in indicator or 'PRICE' in indicator:
                    data[change_col] = data[indicator].pct_change()
                else:
                    data[change_col] = data[indicator].diff()
        
        if 'volatility' in data.columns:
            data['volatility_change'] = data['volatility'].diff()
        
        return data
    
    def split_train_val_test(self, data):
        """Split data into train/val/test"""
        n = len(data)
        train_end = int(n * config.TRAIN_RATIO)
        val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        train_data.to_csv(config.PROCESSED_DATA_DIR / "train_data.csv")
        val_data.to_csv(config.PROCESSED_DATA_DIR / "val_data.csv")
        test_data.to_csv(config.PROCESSED_DATA_DIR / "test_data.csv")
        
        return train_data, val_data, test_data
    
    def run_pipeline(self):
        """Run complete data pipeline"""
        spy_data = self.download_spy_data()
        daily_data = self.download_fred_daily_data()
        monthly_data = self.download_fred_monthly_data()
        alfred_data = self.download_alfred_data()
        
        if not alfred_data.empty:
            monthly_data = pd.concat([monthly_data, alfred_data], axis=1)
        
        monthly_data = self.apply_publication_lags(monthly_data)
        merged_data = self.merge_to_daily_frequency(spy_data, daily_data, monthly_data)
        merged_data = self.compute_macro_changes(merged_data)
        train_data, val_data, test_data = self.split_train_val_test(merged_data)
        
        return {'train': train_data, 'val': val_data, 'test': test_data, 'full': merged_data}