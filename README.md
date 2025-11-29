# Transformer Market Regime Classification (Binary)

Binary classification of SPY market regimes (Bearish vs Bullish) using Transformer neural networks.

## Project Overview

- **Task:** Classify market regimes as Bearish (negative returns) or Bullish (positive returns)
- **Best Model:** Attention Pooling Transformer - **82.99% test accuracy**
- **Data:** SPY (2000-2024) + FRED macroeconomic indicators
- **Approach:** HMM-based regime labeling → Binary classification with Transformers

---

## Quick Start (Google Colab)

```python
# 1. Clone and setup
!git clone https://github.com/Rohanjain2312/transformer-market-regime.git
%cd transformer-market-regime
!pip install -r requirements.txt

# 2. Create data and labels (~10 min)
!python main.py --mode prepare
!python create_binary_labels.py

# 3. Train best model (~30 min)
!python train_binary.py --model_id model_3_attention_pooling

# 4. Or train all models and compare (~3 hours)
# Train each model individually
!python train_binary.py --model_id model_0_baseline
!python train_binary.py --model_id model_1_engineered
!python train_binary.py --model_id model_2_large_capacity
!python train_binary.py --model_id model_3_attention_pooling
!python train_binary.py --model_id model_4_cnn_transformer
!python train_binary.py --model_id model_5_multiscale
!python train_binary.py --model_id model_6_lstm_transformer

# Compare all models
!python compare_binary_models.py
```

---

## Results

### Model Performance (Binary Classification)

| Model | Test Acc | Bearish Acc | Bullish Acc | Parameters |
|-------|----------|-------------|-------------|------------|
| **Attention Pooling** | **82.99%** | 75.44% | 97.33% | 804,740 |
| Engineered Transformer | 82.07% | 75.44% | 94.67% | 102,915 |
| CNN-Transformer | 78.39% | 82.46% | 70.67% | 828,419 |
| LSTM-Transformer | 78.16% | 81.05% | 72.67% | 862,851 |
| Large Capacity | 76.55% | 66.67% | 95.33% | 803,075 |
| Baseline | 64.06% | 0.00% | 100.00% | 102,659 |
| Multi-Scale | 60.00% | 39.30% | 99.33% | 2,433,539 |

**Average Accuracy:** 74.60%

---

## Project Structure

```
transformer-market-regime/
├── src/
│   ├── data_pipeline.py          # Data acquisition (SPY + FRED)
│   ├── feature_engineering.py    # Technical indicators
│   ├── regime_labeling.py        # HMM-based labeling
│   ├── dataset.py                # PyTorch Dataset
│   └── model.py                  # 7 Transformer architectures
│
├── create_binary_labels.py       # Convert 3-class to binary
├── train_binary.py                # Train single model
├── compare_binary_models.py       # Evaluate all models
│
├── config.py                      # All hyperparameters
├── requirements.txt               # Dependencies
└── README.md
```

---

## Features

### Data Sources
- **Price Data:** SPY (S&P 500 ETF)
- **Macroeconomic:** FRED indicators (unemployment, CPI, Fed funds rate, Treasury yields, etc.)
- **Period:** 2000-2024

### Feature Sets
**Baseline (7 features):**
- Log returns, volatility change
- Unemployment rate, CPI, Fed funds rate
- 10Y Treasury rate, 10Y-2Y spread

**Engineered (11 features):**
- All baseline features
- RSI (14-day), MACD signal
- Bollinger band width, Distance from MA200

### Model Architectures

1. **Vanilla Transformer** (Baseline & Engineered)
2. **Large Capacity Transformer** (128 d_model, 4 layers)
3. **Attention Pooling Transformer** ⭐ (Best: 82.99%)
4. **CNN-Transformer Hybrid**
5. **Multi-Scale Transformer**
6. **LSTM-Transformer Hybrid**

---

## Methodology

### 1. Data Pipeline
- Download SPY and FRED data
- Apply publication lags (prevent look-ahead bias)
- Create train/val/test splits (70/15/15)

### 2. Regime Labeling
- Train HMM on log returns and volatility
- Generate 3 regimes: Bearish, Neutral, Bullish
- Map regimes by mean return

### 3. Binary Conversion
- Keep Bearish (negative returns) → Class 0
- Drop Neutral (medium volatility)
- Keep Bullish (positive returns) → Class 1

### 4. Model Training
- Sequence length: 60 days
- Batch size: 64 (GPU)
- Optimizer: Adam (LR: 5e-4)
- Early stopping: 20 epochs patience

---

## Configuration

Edit `config.py` to modify:

**Data:**
```python
START_DATE = "2000-01-01"
END_DATE = "2024-11-24"
TICKER = "SPY"
```

**Model:**
```python
D_MODEL = 128        # Embedding dimension
N_HEADS = 8          # Attention heads
N_LAYERS = 4         # Encoder layers
DROPOUT = 0.1        # Dropout rate
```

**Training:**
```python
NUM_EPOCHS_GPU = 100
BATCH_SIZE_GPU = 64
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 20
```

---

## Why Binary Classification?

We initially tried 3-class classification (Bearish/Neutral/Bullish) but found:

- **Neutral regime** was actually **medium volatility**, not sideways markets
- Models struggled to predict Neutral (7-17% accuracy)
- HMM labels were based on volatility levels, not return direction

**Binary classification solved this:**
- Clear distinction: Negative returns vs Positive returns
- Much better performance: 83% vs 62%
- More actionable for trading strategies

---

## Key Insights

1. **Attention Pooling works best** - Weighted pooling over time beats simple averaging
2. **Engineered features help** - Technical indicators improve over baseline
3. **Model collapse risk** - Some models predict only one class (Baseline, Multi-Scale)
4. **Bullish easier than Bearish** - Models achieve 94-99% on Bullish, 39-82% on Bearish
5. **Size ≠ Performance** - Multi-Scale (2.4M params) underperforms Engineered (103K params)

---

## Future Improvements

- Ensemble top 3 models → 84-86% expected
- Add more volatility features (VIX, ATR)
- Test on different tickers (QQQ, IWM)
- Multi-task learning (regime + return prediction)
- Deploy as trading signal

---

## Requirements

```bash
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
yfinance>=0.2.28
pandas-datareader>=0.10.0
fredapi>=0.5.0
scikit-learn>=1.3.0
hmmlearn>=0.3.0
ta>=0.11.0
tqdm>=4.65.0
```