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

# Model Architecture
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
NUM_CLASSES = 3

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10
SAVE_BEST_ONLY = True

# Active Configuration
ACTIVE_FEATURE_SET = 'baseline'
RANDOM_SEED = 42

def get_active_features():
    return BASELINE_FEATURES if ACTIVE_FEATURE_SET == 'baseline' else ENGINEERED_FEATURES