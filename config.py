"""Configuration file for Transformer Market Regime Classification"""

from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REGIME_DATA_DIR = DATA_DIR / "regimes"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, REGIME_DATA_DIR, CHECKPOINT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Configuration
FRED_API_KEY = "6fbc1b76fb000f2c5259127469b3588b"

# Data Configuration
START_DATE = "2000-01-01"
END_DATE = "2024-11-24"
TICKER = "SPY"

# FRED Indicators
FRED_MONTHLY_INDICATORS = ['UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'PAYEMS', 'UMCSENT']
FRED_DAILY_INDICATORS = ['DGS10', 'T10Y2Y', 'DEXUSEU', 'DCOILWTICO']

# ALFRED Configuration
USE_ALFRED_VINTAGES = True
ALFRED_INDICATORS = ['CPIAUCSL', 'UNRATE', 'PAYEMS']

# Publication Lags (days)
PUBLICATION_LAGS = {
    'CPIAUCSL': 15,
    'UNRATE': 7,
    'PAYEMS': 7,
    'FEDFUNDS': 1,
    'UMCSENT': 0,
}

# Data Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEQUENCE_LENGTH = 60

# HMM Configuration
HMM_N_STATES = 3
HMM_N_ITER = 100
HMM_RANDOM_STATE = 42
FIT_HMM_ON_TRAIN_ONLY = True
HMM_FEATURES = ['log_return', 'volatility']
REGIME_NAMES = ['Bearish', 'Neutral', 'Bullish']

# Technical Indicators
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
MA_LONG_PERIOD = 200

# Feature Sets
BASELINE_FEATURES = [
    'log_return', 'volatility_change', 'UNRATE_change', 'CPIAUCSL_change',
    'FEDFUNDS_change', 'DGS10', 'T10Y2Y'
]

ENGINEERED_FEATURES = BASELINE_FEATURES + [
    'rsi', 'macd_signal', 'bollinger_width', 'distance_from_ma200'
]

# Model Configurations
MODEL_CONFIGS = {
    'model_0_baseline': {
        'name': 'Baseline Transformer',
        'features': BASELINE_FEATURES,
        'n_features': len(BASELINE_FEATURES),
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'architecture': 'vanilla'
    },
    'model_1_engineered': {
        'name': 'Engineered Transformer',
        'features': ENGINEERED_FEATURES,
        'n_features': len(ENGINEERED_FEATURES),
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'architecture': 'vanilla'
    },
    'model_2_large_capacity': {
        'name': 'Large Capacity Transformer',
        'features': ENGINEERED_FEATURES,
        'n_features': len(ENGINEERED_FEATURES),
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'architecture': 'vanilla'
    },
    'model_3_attention_pooling': {
        'name': 'Attention Pooling Transformer',
        'features': ENGINEERED_FEATURES,
        'n_features': len(ENGINEERED_FEATURES),
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'architecture': 'attention_pooling'
    },
    'model_4_cnn_transformer': {
        'name': 'CNN-Transformer Hybrid',
        'features': ENGINEERED_FEATURES,
        'n_features': len(ENGINEERED_FEATURES),
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'architecture': 'cnn_transformer',
        'cnn_channels': 64,
        'cnn_kernel': 3
    },
    'model_5_multiscale': {
        'name': 'Multi-Scale Transformer',
        'features': ENGINEERED_FEATURES,
        'n_features': len(ENGINEERED_FEATURES),
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'architecture': 'multiscale',
        'windows': [30, 60, 90]
    },
    'model_6_lstm_transformer': {
        'name': 'LSTM-Transformer Hybrid',
        'features': ENGINEERED_FEATURES,
        'n_features': len(ENGINEERED_FEATURES),
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'architecture': 'lstm_transformer',
        'lstm_hidden': 64,
        'lstm_layers': 2
    }
}

# Training Configuration
NUM_CLASSES = 3
BATCH_SIZE_CPU = 32
BATCH_SIZE_GPU = 64
LEARNING_RATE = 5e-4  # Increased from 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS_CPU = 5
NUM_EPOCHS_GPU = 100  # Increased from 50

# Learning Rate Scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MIN_LR = 1e-6

# Early Stopping
EARLY_STOPPING_PATIENCE = 20  # Increased from 10
SAVE_BEST_ONLY = True

# Random Seeds
RANDOM_SEED = 42

def get_model_config(model_id):
    """Get configuration for specific model"""
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_id]

def list_available_models():
    """List all available model configurations"""
    return list(MODEL_CONFIGS.keys())