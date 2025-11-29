"""Data Pipeline: Acquisition, merging, and preprocessing with ALFRED protocol - Feature Expansion"""

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
        print("\n[1/10] Downloading SPY data...")
        spy = yf.download(config.TICKER, start=config.START_DATE, end=config.END_DATE, progress=False, auto_adjust=False)
        
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        
        spy['log_return'] = np.log(spy['Adj Close'] / spy['Adj Close'].shift(1))
        spy['volatility'] = spy['log_return'].rolling(window=20).std()
        spy = spy.dropna()
        spy.to_csv(config.RAW_DATA_DIR / "spy_data.csv")
        print(f"    Downloaded {len(spy)} days ({spy.index[0].date()} to {spy.index[-1].date()})")
        return spy
    
    def download_vix_data(self):
        """NEW: Download VIX volatility index"""
        print("\n[2/10] Downloading VIX data...")
        try:
            vix = yf.download(config.VIX_TICKER, start=config.START_DATE, end=config.END_DATE, progress=False)
            
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            
            vix_data = vix[['Close']].rename(columns={'Close': 'VIX'})
            print(f"    VIX: {len(vix_data)} observations")
            return vix_data
        except Exception as e:
            print(f"    Warning: VIX download failed - {e}")
            return pd.DataFrame()
    
    def download_sector_etfs(self):
        """NEW: Download sector ETF data"""
        print("\n[3/10] Downloading Sector ETF data...")
        sector_data = {}
        
        for sector in config.SECTOR_ETFS:
            try:
                etf = yf.download(sector, start=config.START_DATE, end=config.END_DATE, progress=False)
                
                if isinstance(etf.columns, pd.MultiIndex):
                    etf.columns = etf.columns.get_level_values(0)
                
                sector_data[sector] = etf['Close']
                print(f"    {sector}: {len(etf)} observations")
            except Exception as e:
                print(f"    Warning: {sector} failed - {e}")
                sector_data[sector] = pd.Series(dtype=float)
        
        return pd.DataFrame(sector_data)
    
    def download_intermarket_data(self):
        """NEW: Download inter-market assets (Gold, Bonds, Dollar)"""
        print("\n[4/10] Downloading Inter-market data...")
        intermarket_data = {}
        
        ticker_mapping = {
            'GLD': 'GLD',
            'TLT': 'TLT',
            'DX-Y.NYB': 'DXY'  # Rename Dollar Index
        }
        
        for ticker in config.INTERMARKET_TICKERS:
            try:
                asset = yf.download(ticker, start=config.START_DATE, end=config.END_DATE, progress=False)
                
                if isinstance(asset.columns, pd.MultiIndex):
                    asset.columns = asset.columns.get_level_values(0)
                
                col_name = ticker_mapping.get(ticker, ticker)
                intermarket_data[col_name] = asset['Close']
                print(f"    {col_name}: {len(asset)} observations")
            except Exception as e:
                print(f"    Warning: {ticker} failed - {e}")
                intermarket_data[ticker_mapping.get(ticker, ticker)] = pd.Series(dtype=float)
        
        return pd.DataFrame(intermarket_data)
    
    def download_fred_daily_data(self):
        print("\n[5/10] Downloading FRED daily data...")
        daily_data = {}
        for indicator in config.FRED_DAILY_INDICATORS:
            try:
                series = self.fred.get_series(indicator, observation_start=config.START_DATE, observation_end=config.END_DATE)
                daily_data[indicator] = series
                print(f"    {indicator}: {len(series)} observations")
            except Exception as e:
                print(f"    Warning: {indicator} failed - {e}")
                daily_data[indicator] = pd.Series(dtype=float)
        return pd.DataFrame(daily_data)
    
    def download_fred_monthly_data(self):
        print("\n[6/10] Downloading FRED monthly data...")
        monthly_data = {}
        for indicator in config.FRED_MONTHLY_INDICATORS:
            if config.USE_ALFRED_VINTAGES and indicator in config.ALFRED_INDICATORS:
                continue
            try:
                series = self.fred.get_series(indicator, observation_start=config.START_DATE, observation_end=config.END_DATE)
                monthly_data[indicator] = series
                print(f"    {indicator}: {len(series)} observations")
            except Exception as e:
                print(f"    Warning: {indicator} failed - {e}")
                monthly_data[indicator] = pd.Series(dtype=float)
        return pd.DataFrame(monthly_data)
    
    def download_alfred_data(self):
        print("\n[7/10] Downloading ALFRED vintage data...")
        if not config.USE_ALFRED_VINTAGES:
            print("    ALFRED disabled")
            return pd.DataFrame()
        
        alfred_data = {}
        for indicator in config.ALFRED_INDICATORS:
            try:
                series = pdr.get_data_fred(indicator, start=config.START_DATE, end=config.END_DATE)
                alfred_data[indicator] = series[indicator]
                print(f"    {indicator}: {len(series)} observations (vintage)")
            except Exception as e:
                print(f"    Warning: {indicator} ALFRED failed, using standard FRED")
                try:
                    series = self.fred.get_series(indicator, observation_start=config.START_DATE, observation_end=config.END_DATE)
                    alfred_data[indicator] = series
                except:
                    alfred_data[indicator] = pd.Series(dtype=float)
        return pd.DataFrame(alfred_data)
    
    def apply_publication_lags(self, monthly_data):
        print("\n[8/10] Applying publication lags...")
        lagged_data = monthly_data.copy()
        for indicator in lagged_data.columns:
            if indicator in config.PUBLICATION_LAGS:
                lag_days = config.PUBLICATION_LAGS[indicator]
                lagged_data[indicator] = lagged_data[indicator].shift(1)
                lagged_data.index = lagged_data.index + pd.DateOffset(days=lag_days)
                print(f"    {indicator}: +{lag_days} days")
        return lagged_data
    
    def merge_to_daily_frequency(self, spy_data, vix_data, sector_data, intermarket_data, daily_data, monthly_data):
        """UPDATED: Merge all data sources to daily frequency"""
        print("\n[9/10] Merging to daily frequency...")
        merged = spy_data[['Adj Close', 'log_return', 'volatility', 'High', 'Low', 'Volume']].copy()
        
        # VIX
        if not vix_data.empty:
            merged = merged.join(vix_data, how='left')
            merged['VIX'] = merged['VIX'].fillna(method='ffill')
        
        # Sector ETFs
        if not sector_data.empty:
            merged = merged.join(sector_data, how='left')
            for col in sector_data.columns:
                merged[col] = merged[col].fillna(method='ffill')
        
        # Inter-market
        if not intermarket_data.empty:
            merged = merged.join(intermarket_data, how='left')
            for col in intermarket_data.columns:
                merged[col] = merged[col].fillna(method='ffill')
        
        # FRED daily
        if not daily_data.empty:
            merged = merged.join(daily_data, how='left')
        
        # FRED monthly
        if not monthly_data.empty:
            merged = merged.join(monthly_data, how='left')
            for col in monthly_data.columns:
                merged[col] = merged[col].fillna(method='ffill')
        
        initial_len = len(merged)
        merged = merged.dropna()
        print(f"    Final: {len(merged)} days, {len(merged.columns)} columns (dropped {initial_len - len(merged)} NaN rows)")
        return merged
    
    def compute_macro_changes(self, data):
        print("\n[10/10] Computing macro changes...")
        macro_indicators = config.FRED_MONTHLY_INDICATORS + (config.ALFRED_INDICATORS if config.USE_ALFRED_VINTAGES else [])
        
        for indicator in macro_indicators:
            if indicator in data.columns:
                change_col = f"{indicator}_change"
                if 'CPI' in indicator or 'PRICE' in indicator:
                    data[change_col] = data[indicator].pct_change()
                else:
                    data[change_col] = data[indicator].diff()
                print(f"    Created {change_col}")
        
        if 'volatility' in data.columns:
            data['volatility_change'] = data['volatility'].diff()
            print(f"    Created volatility_change")
        
        return data
    
    def split_train_val_test(self, data):
        n = len(data)
        train_end = int(n * config.TRAIN_RATIO)
        val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        train_data.to_csv(config.PROCESSED_DATA_DIR / "train_data.csv")
        val_data.to_csv(config.PROCESSED_DATA_DIR / "val_data.csv")
        test_data.to_csv(config.PROCESSED_DATA_DIR / "test_data.csv")
        
        print(f"\nData split complete:")
        print(f"    Train: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        print(f"    Val:   {len(val_data)} days ({val_data.index[0].date()} to {val_data.index[-1].date()})")
        print(f"    Test:  {len(test_data)} days ({test_data.index[0].date()} to {test_data.index[-1].date()})")
        
        return train_data, val_data, test_data
    
    def run_pipeline(self):
        print("\n" + "="*60)
        print("DATA PIPELINE STARTED - FEATURE EXPANSION")
        print("="*60)
        
        spy_data = self.download_spy_data()
        vix_data = self.download_vix_data()
        sector_data = self.download_sector_etfs()
        intermarket_data = self.download_intermarket_data()
        daily_data = self.download_fred_daily_data()
        monthly_data = self.download_fred_monthly_data()
        alfred_data = self.download_alfred_data()
        
        if not alfred_data.empty:
            monthly_data = pd.concat([monthly_data, alfred_data], axis=1)
        
        monthly_data = self.apply_publication_lags(monthly_data)
        merged_data = self.merge_to_daily_frequency(spy_data, vix_data, sector_data, intermarket_data, daily_data, monthly_data)
        merged_data = self.compute_macro_changes(merged_data)
        train_data, val_data, test_data = self.split_train_val_test(merged_data)
        
        print("\n" + "="*60)
        print("DATA PIPELINE COMPLETE")
        print("="*60 + "\n")
        
        return {'train': train_data, 'val': val_data, 'test': test_data, 'full': merged_data}


if __name__ == "__main__":
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()
    print(f"\nFinal dataset shape: {data['full'].shape}")
    print(f"Columns: {list(data['full'].columns)}\n")
