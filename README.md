# Transformer Market Regime Classification

End-to-end pipeline for classifying market regimes (Bearish/Neutral/Bullish) using Transformer architecture with ALFRED-compliant macroeconomic data.

## Project Structure

```
transformer-market-regime/
├── src/
│   ├── data_pipeline.py          # Data acquisition with ALFRED protocol
│   ├── feature_engineering.py    # Technical indicators
│   ├── regime_labeling.py        # HMM-based regime labeling
│   ├── dataset.py                # PyTorch Dataset
│   ├── model.py                  # Transformer architecture
│   └── trainer.py                # Training loop
├── data/                          # Generated data (gitignored)
├── checkpoints/                   # Saved models (gitignored)
├── config.py                      # All hyperparameters
├── main.py                        # Main orchestrator
└── requirements.txt               # Dependencies
```

## Local Setup (CPU Testing)

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (5 epochs on 25% data)
python main.py --mode all --feature_set baseline

# Or run individual phases
python main.py --mode data
python main.py --mode features
python main.py --mode label
python main.py --mode train
```

## Google Colab Setup (GPU Training)

### Step 1: Upload to GitHub

```bash
# Initialize git
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Run on Colab

1. Open Google Colab: https://colab.research.google.com
2. Enable GPU: Runtime > Change runtime type > GPU
3. Run these commands:

```python
# Clone repository
!git clone <your-github-repo-url>
%cd transformer-market-regime

# Install dependencies
!pip install -r requirements.txt

# Run full pipeline (50 epochs on full data)
!python main.py --mode all --feature_set baseline

# Check GPU usage
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Step 3: Download Results

```python
# Download trained model
from google.colab import files
files.download('checkpoints/baseline_transformer/best.pth')

# Download training curves
files.download('data/baseline_transformer_history.png')
```

## Model Comparison

Train both baseline and engineered models:

```bash
# Baseline (7 features)
python main.py --mode all --feature_set baseline

# Engineered (11 features with technical indicators)
python main.py --mode all --feature_set engineered
```

Results saved to:
- `checkpoints/baseline_transformer/`
- `checkpoints/engineered_transformer/`

## Configuration

Edit `config.py` to modify:

**Data:**
- `START_DATE`, `END_DATE`: Date range
- `FRED_API_KEY`: Your FRED API key
- `USE_ALFRED_VINTAGES`: Enable/disable ALFRED protocol
- `FIT_HMM_ON_TRAIN_ONLY`: Prevent look-ahead bias

**Model:**
- `D_MODEL`: Embedding dimension (default: 64)
- `N_HEADS`: Attention heads (default: 4)
- `N_LAYERS`: Encoder layers (default: 2)
- `DROPOUT`: Dropout rate (default: 0.1)

**Training:**
- `NUM_EPOCHS`: Training epochs (default: 50 for GPU, 5 for CPU)
- `BATCH_SIZE`: Batch size (default: 32)
- `LEARNING_RATE`: Learning rate (default: 1e-4)

## Features

**Baseline (7 features):**
- Log returns
- Volatility change
- Unemployment rate change
- CPI change
- Fed funds rate change
- 10Y Treasury rate
- 10Y-2Y spread

**Engineered (11 features):**
- All baseline features
- RSI (14-day)
- MACD signal
- Bollinger band width
- Distance from MA200

## Key Features

- **ALFRED Protocol**: Prevents look-ahead bias using vintage macroeconomic data
- **Publication Lags**: Accounts for real-world data release delays
- **HMM Regime Labeling**: Unsupervised regime detection
- **GPU Auto-Detection**: Automatically adjusts training for CPU/GPU
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Class Weighting**: Handles regime imbalance

## Current Performance

**Baseline Model (CPU - 5 epochs on 25% data):**
- Test Accuracy: 40.74%
- Note: Poor performance expected with limited training

**Expected Performance (GPU - 50 epochs on full data):**
- Target: 60-70% test accuracy
- Will update after GPU training

## Next Steps

1. Train on GPU (Google Colab)
2. Compare baseline vs engineered features
3. Experiment with model architectures
4. Add multi-task learning (regime + price prediction)

## Authors

UMD MSML - Semester 3 Project