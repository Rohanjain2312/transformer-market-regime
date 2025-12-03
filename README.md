# Transformer-Based Market Regime Classification

**Deep Learning for Financial Time Series**: Binary regime classification (Bearish/Bullish) using Transformer architectures with macroeconomic features and technical indicators.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

This project implements a **Transformer-based deep learning pipeline** for classifying S&P 500 (SPY) market regimes using a combination of:
- **Macroeconomic indicators** (Federal Reserve data via ALFRED protocol)
- **Technical indicators** (RSI, MACD, Bollinger Bands)
- **Unsupervised HMM regime labeling** for ground truth generation

### Key Features
- **7 Transformer architectures** tested and compared
- **Dynamic date handling** (T-2 months for real-time applicability)
- **ALFRED protocol** for point-in-time macroeconomic data (no look-ahead bias)
- **Threshold optimization** using Matthews Correlation Coefficient (MCC)
- **Comprehensive evaluation** with precision, recall, F1-score, and MCC metrics
- **Interactive demo** with 5 realistic market scenarios

---

## Results Summary

### Best Model: CNN-Transformer Hybrid

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **81.78%** |
| **MCC** | **0.480** |
| **F1-Score** | **72.70%** |
| **Bearish F1** | 56.95% |
| **Bullish F1** | 88.44% |
| **Parameters** | 828,354 |

### All Models Comparison

| Model | Test Acc | MCC | F1-Score | Parameters |
|-------|----------|-----|----------|------------|
| **CNN-Transformer** | **81.78%** | **0.480** | **72.70%** | 828,354 |
| Engineered Transformer | 74.75% | 0.506 | 69.05% | 102,882 |
| Large Capacity | 79.48% | 0.396 | 68.79% | 803,010 |
| Multi-Scale | 78.19% | 0.315 | 65.29% | 2,433,410 |
| Attention Pooling | 69.73% | 0.405 | 63.70% | 804,675 |
| LSTM-Transformer | 63.41% | 0.309 | 57.81% | 862,786 |
| Baseline | 61.16% | -0.036 | 43.80% | 102,626 |

### Demo Scenarios (4/5 Correct Predictions)

| Scenario | Expected | Predicted | Confidence |
|----------|----------|-----------|------------|
| Strong Bull Market | ✓ Bullish | ✓ Bullish | 63.0% |
| Bear Market Decline | ✓ Bearish | ✓ Bearish | 62.8% |
| Market Recovery | ✓ Bullish | ✓ Bullish | 61.0% |
| High Volatility Bull | Bullish | ✗ Bearish | 45.9% |
| Calm Bull Market | ✓ Bullish | ✓ Bullish | 64.5% |

---

## Architecture

### Model Variants

1. **Baseline Transformer** (Model 0)
   - Vanilla transformer encoder
   - 7 baseline features
   - 102K parameters

2. **Engineered Transformer** (Model 1)
   - Vanilla transformer encoder
   - 11 engineered features (+ technical indicators)
   - 102K parameters

3. **Large Capacity Transformer** (Model 2)
   - Deeper architecture (d_model=128, 4 layers, 8 heads)
   - 803K parameters

4. **Attention Pooling Transformer** (Model 3)
   - Attention-weighted pooling + residual connections
   - 804K parameters

5. **CNN-Transformer Hybrid** (Model 4) ⭐ **BEST**
   - 1D CNN feature extraction → Transformer encoder
   - Captures both local patterns and long-range dependencies
   - 828K parameters

6. **Multi-Scale Transformer** (Model 5)
   - Parallel transformers on different window sizes (30/60/90 days)
   - 2.4M parameters

7. **LSTM-Transformer Hybrid** (Model 6)
   - LSTM → Transformer encoder
   - 862K parameters

### Feature Engineering

**Baseline Features (7):**
- `log_return`: Daily log returns
- `volatility_change`: Change in 20-day rolling volatility
- `UNRATE_change`: Unemployment rate change
- `CPIAUCSL_change`: CPI inflation change
- `FEDFUNDS_change`: Federal funds rate change
- `DGS10`: 10-year Treasury rate
- `T10Y2Y`: 10Y-2Y yield spread

**Engineered Features (+4):**
- `rsi`: 14-day Relative Strength Index
- `macd_signal`: MACD signal line
- `bollinger_width`: Bollinger band width
- `distance_from_ma200`: Distance from 200-day MA

---

## Project Structure

```
transformer-market-regime/
├── config.py                      # All hyperparameters and configurations
├── requirements.txt               # Python dependencies
│
├── src/
│   ├── data_pipeline.py          # Data acquisition (ALFRED protocol)
│   ├── feature_engineering.py    # Technical indicators
│   ├── regime_labeling.py        # HMM-based regime labeling
│   ├── dataset.py                # PyTorch Dataset (60-day sequences)
│   ├── model.py                  # 7 Transformer architectures
│   └── trainer.py                # Training loop with LR scheduling
│
├── scripts/
│   ├── prepare_data.py           # Complete data preparation pipeline
│   ├── train_model.py            # Train single model
│   ├── train_all_models.py       # Train all 7 models
│   ├── tune_all_thresholds.py   # Optimize decision thresholds
│   ├── compare_models.py         # Compare all models with visualizations
│   └── demo_scenario.py          # Interactive demo with 5 scenarios
│
├── data/                          # Generated data (gitignored)
│   ├── raw/                      # Downloaded SPY + FRED data
│   ├── processed/                # Engineered features
│   ├── regimes/                  # HMM-labeled data
│   └── *.png, *.csv              # Results and visualizations
│
└── checkpoints/                   # Saved models (gitignored)
    └── {model_id}/
        └── best.pth              # Best checkpoint by MCC
```

---

## Quick Start

### Local Setup (CPU Testing - 5 Epochs)

```bash
# Clone repository
git clone https://github.com/yourusername/transformer-market-regime.git
cd transformer-market-regime

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (data prep + train baseline model)
python scripts/prepare_data.py
python scripts/train_model.py --model_id model_0_baseline

# Compare all trained models
python scripts/compare_models.py

# Run interactive demo
python scripts/demo_scenario.py
```

### Google Colab (GPU Training - 100 Epochs)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/transformer-market-regime/blob/main/regime_transformer.ipynb)

**Steps:**
1. Open the Colab notebook (link above)
2. Enable GPU: `Runtime > Change runtime type > GPU`
3. Run all cells sequentially

**Colab Workflow:**
```python
# 1. Setup
!git clone https://github.com/yourusername/transformer-market-regime.git
%cd transformer-market-regime
!pip install -r requirements.txt

# 2. Prepare data (with visualization)
!python scripts/prepare_data.py

# 3. Train all models
!python scripts/train_all_models.py

# 4. Optimize thresholds
!python scripts/tune_all_thresholds.py

# 5. Compare models
!python scripts/compare_models.py

# 6. Run demo
!python scripts/demo_scenario.py

# 7. Download results
from google.colab import files
files.download('/content/transformer-market-regime/data/model_results.csv')
```

---

## Configuration

Edit `config.py` to customize:

### Data Configuration
```python
START_DATE = "2000-01-01"
END_DATE = "2024-11-24"  # Dynamic: T-2 months
TICKER = "SPY"

# ALFRED Protocol (prevents look-ahead bias)
USE_ALFRED_VINTAGES = True
PUBLICATION_LAGS = {
    'CPIAUCSL': 15,  # CPI published 15 days after month end
    'UNRATE': 7,
    'PAYEMS': 7,
}
```

### Model Configuration
```python
SEQUENCE_LENGTH = 60  # 60-day lookback window

# Model architecture (Model 4 - CNN-Transformer)
'd_model': 128,
'n_heads': 8,
'n_layers': 4,
'd_ff': 512,
'dropout': 0.1,
'cnn_channels': 64,
'cnn_kernel': 3
```

### Training Configuration
```python
# GPU settings (Colab)
BATCH_SIZE_GPU = 64
NUM_EPOCHS_GPU = 100
LEARNING_RATE = 5e-4

# CPU settings (Local testing)
BATCH_SIZE_CPU = 32
NUM_EPOCHS_CPU = 5

# Optimization
USE_LR_SCHEDULER = True
EARLY_STOPPING_PATIENCE = 20
```

---

## Visualizations

### 1. Market Regimes Over Time
![Market Regimes](data/spy_regimes_binary.png)

### 2. MCC Comparison
![MCC Comparison](data/plot1_mcc.png)

### 3. Per-Class F1-Score Heatmap
![F1 Heatmap](data/plot3_per_class_f1.png)

### 4. Best Model Confusion Matrix
![Confusion Matrix](data/plot4_best_model_confusion.png)

### 5. Demo Predictions
![Demo Predictions](data/demo_predictions.png)

---

## Data and Methodology

### Input Data Specifications

**Market Data (Primary Signal)**
- **Source**: S&P 500 ETF (SPY) via Yahoo Finance
- **Frequency**: Daily (business days only)
- **Time Period**: January 1, 2000 to November 24, 2024 (dynamic T-2 months)
- **Total Trading Days**: ~6,200 observations
- **Primary Features**:
  - **Adjusted Close Price**: Split and dividend adjusted closing price
  - **Log Returns**: `log(P_t / P_{t-1})`
  - **Volatility**: 20-day rolling standard deviation of log returns

**Macroeconomic Indicators (Federal Reserve Data)**

*Monthly Indicators (FRED API with ALFRED protocol):*
- **UNRATE**: Unemployment Rate
  - Units: Percent, seasonally adjusted
  - Publication lag: 7 days after month end
  - Example: 3.7% (low) to 14.7% (COVID peak)
  
- **CPIAUCSL**: Consumer Price Index for All Urban Consumers
  - Units: Index (1982-1984 = 100)
  - Publication lag: 15 days after month end
  - Used as: Year-over-year % change for inflation
  
- **FEDFUNDS**: Federal Funds Effective Rate
  - Units: Percent per annum
  - Publication lag: 1 day
  - Range: 0.00% (zero lower bound) to 5.33% (recent)
  
- **PAYEMS**: Total Nonfarm Payrolls
  - Units: Thousands of employees, seasonally adjusted
  - Publication lag: 7 days
  - Used as: Month-over-month change
  
- **UMCSENT**: University of Michigan Consumer Sentiment Index
  - Units: Index (1966:Q1 = 100)
  - Publication lag: Same day
  - Range: ~50 (recession) to ~100 (expansion)

*Daily Indicators (FRED API):*
- **DGS10**: 10-Year Treasury Constant Maturity Rate
  - Units: Percent per annum
  - Range: 0.52% (2020 low) to 5.0% (2007)
  
- **T10Y2Y**: 10-Year minus 2-Year Treasury Yield Spread
  - Units: Percentage points
  - Critical indicator: Negative = yield curve inversion (recession signal)
  
- **DEXUSEU**: U.S. / Euro Foreign Exchange Rate
  - Units: U.S. Dollars per Euro
  
- **DCOILWTICO**: Crude Oil Prices: West Texas Intermediate
  - Units: Dollars per barrel

**Technical Indicators (Engineered Features)**
- **RSI (14-day)**: Relative Strength Index
  - Range: 0-100
  - Oversold: < 30, Overbought: > 70
  - Computation: `100 - (100 / (1 + RS))` where RS = avg gain / avg loss
  
- **MACD Signal**: Moving Average Convergence Divergence
  - Fast EMA: 12 days
  - Slow EMA: 26 days
  - Signal line: 9-day EMA of MACD
  
- **Bollinger Band Width**: Volatility measure
  - Period: 20 days
  - Standard deviations: 2
  - Width = (Upper Band - Lower Band) / Middle Band
  
- **Distance from MA200**: Price momentum
  - Calculation: `(Price - MA200) / MA200`
  - Positive = above long-term average (bullish)

### Data Processing Pipeline

**1. Data Acquisition**
```
Raw Data Sources → Merging → Resampling → Forward Fill
     ↓               ↓            ↓             ↓
  SPY Daily    Monthly to     Business      Missing
  FRED Daily    Daily freq     Days only     Values
  FRED Monthly
```

**2. ALFRED Protocol Implementation**
- **Purpose**: Prevent look-ahead bias by using only data available at each point in time
- **Method**: Apply publication lags to macroeconomic indicators
- **Example**: October 2024 CPI (published Nov 15) is not available until November 15 in model
- **Critical for**: Real-world deployment and accurate backtesting

**3. Feature Engineering**
```
Price Data → Log Returns → Volatility → Technical Indicators
    ↓            ↓             ↓              ↓
Macro Data → Changes/Diffs → Standardization
```

**4. Train/Validation/Test Split**
- **Method**: Time-based (no random shuffle)
- **Training Set**: 70% (4,253 samples)
  - Period: Dec 15, 2000 to Jan 24, 2018
  - Regimes: 33.3% Bearish, 66.7% Bullish
  
- **Validation Set**: 15% (755 samples)
  - Period: Nov 8, 2018 to Dec 2, 2021
  - Regimes: 29.1% Bearish, 70.9% Bullish
  
- **Test Set**: 15% (756 samples)
  - Period: Sep 21, 2022 to Oct 3, 2025
  - Regimes: 31.7% Bearish, 68.3% Bullish

### Model Training Specifications

**Input Representation**
- **Sequence Length**: 60 trading days (~3 months)
- **Input Shape**: `(batch_size, 60, n_features)` where n_features = 11
- **Normalization**: Per-feature standardization (z-score)

**Regime Labeling (Ground Truth)**
- **Method**: 3-state Gaussian Hidden Markov Model
- **HMM Features**: Log returns + volatility
- **HMM Parameters**:
  - States: 3 (Bearish, Neutral, Bullish)
  - Covariance: Full
  - Iterations: 100
  - Random seed: 42
- **Binary Conversion**: Drop Neutral class → 2-state problem
- **Training**: HMM fitted on training set only (no test leakage)
- **State Mapping**: Based on mean returns per state

**Model Architecture (CNN-Transformer - Best Model)**
```
Input (60, 11)
    ↓
1D Conv Layer (kernel=3, channels=64)
    ↓
ReLU + Dropout(0.1)
    ↓
1D Conv Layer (kernel=3, channels=128)
    ↓
Positional Encoding
    ↓
Transformer Encoder (4 layers)
    ├─ Multi-Head Attention (8 heads)
    ├─ Feed-Forward (d_ff=512)
    └─ Layer Norm + Residual
    ↓
Global Average Pooling
    ↓
Linear(128 → 64) + ReLU + Dropout
    ↓
Linear(64 → 2) [Bearish, Bullish]
    ↓
Softmax
```

**Training Hyperparameters**
- **Loss Function**: Cross-Entropy with class weights
  - Bearish weight: 1.5 (minority class)
  - Bullish weight: 1.0 (majority class)
  
- **Optimizer**: Adam
  - Learning rate: 5e-4
  - Weight decay: 1e-5 (L2 regularization)
  - Betas: (0.9, 0.999)
  
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Mode: Maximize validation accuracy
  - Factor: 0.5 (halve LR on plateau)
  - Patience: 10 epochs
  - Minimum LR: 1e-6
  
- **Early Stopping**: 
  - Patience: 20 epochs
  - Monitor: Validation MCC
  
- **Batch Size**: 
  - GPU (Colab): 64
  - CPU (Local): 32
  
- **Epochs**:
  - GPU (Colab): 100 (with early stopping)
  - CPU (Local): 5 (quick test)
  
- **Hardware**:
  - Colab: Tesla T4 GPU (16GB)
  - Training time: ~15 min/model on GPU

### Evaluation Metrics

**Primary Metric: Matthews Correlation Coefficient (MCC)**
- Range: -1 to +1
- Interpretation: 
  - +1: Perfect prediction
  - 0: Random prediction
  - -1: Perfect inverse prediction
- Advantage: Handles class imbalance better than accuracy
- Formula: `(TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`

**Secondary Metrics**
- **F1-Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Average of per-class recall
- **Per-Class Precision/Recall**: Detailed performance breakdown

### Threshold Optimization
- **Method**: Grid search over 81 candidate thresholds (0.10 to 0.90, step=0.01)
- **Optimization set**: Validation set
- **Objective**: Maximize MCC
- **Application**: Optimal threshold applied to test set
- **Best threshold (CNN-Transformer)**: 0.69

---

## Key Insights

### Model Performance
1. **CNN-Transformer wins** by capturing both local patterns (CNN) and long-range dependencies (Transformer)
2. **Technical indicators help**: Engineered features (RSI, MACD) improve baseline by 13.6%
3. **Bigger ≠ Better**: Multi-Scale (2.4M params) underperforms CNN-Transformer (828K params)
4. **Threshold tuning critical**: Can improve MCC by up to 5% on some models

### Regime Classification
- **Bullish easier to predict** (88.4% F1) than Bearish (57.0% F1)
- **Class imbalance**: 68% Bullish vs 32% Bearish in test set
- **MCC preferred over accuracy** due to imbalance (baseline 61% accuracy but -0.04 MCC)

### Challenges
- **High volatility scenarios** remain difficult (demo 4/5 correct)
- **Bear markets underrepresented** in training data (2000-2024 mostly bull)
- **Look-ahead bias prevention** essential (ALFRED protocol critical)

---

## References

### Academic Papers
- Vaswani et al. (2017) - "Attention Is All You Need"
- Wen et al. (2022) - "Transformers in Time Series"
- Hamilton (1989) - "A New Approach to Economic Analysis of Nonstationary Time Series"

### Libraries & Tools
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [hmmlearn](https://hmmlearn.readthedocs.io/) - Hidden Markov Models
- [yfinance](https://github.com/ranaroussi/yfinance) - Market data
- [fredapi](https://github.com/mortada/fredapi) - FRED macroeconomic data
- [ta](https://github.com/bukosabino/ta) - Technical analysis indicators

---

**University of Maryland - Master of Science in Machine Learning**  
**GitHub**: [@Rohanjain2312](https://github.com/Rohanjain2312)
