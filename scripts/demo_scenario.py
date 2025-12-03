"""Demo: Realistic synthetic scenarios based on real data distributions"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch

import config
from src.model import create_model


def find_best_model():
    """Find best model based on test MCC"""
    comparison_file = config.DATA_DIR / "binary_model_results.csv"
    
    if not comparison_file.exists():
        raise FileNotFoundError("Run compare_binary_models.py first!")
    
    results_df = pd.read_csv(comparison_file)
    results_df = results_df.sort_values('MCC', ascending=False)
    
    best_row = results_df.iloc[0]
    best_model_id = best_row['Model ID']
    
    threshold = 0.5
    threshold_file = config.DATA_DIR / f"threshold_tuning_{best_model_id}.csv"
    if threshold_file.exists():
        threshold_df = pd.read_csv(threshold_file)
        threshold = threshold_df['optimal_threshold'].values[0]
    
    print(f"\nBest Model: {best_row['Name']}")
    print(f"Model ID: {best_model_id}")
    print(f"Test MCC: {best_row['MCC']:.4f}")
    print(f"Test F1: {best_row['F1-Score (%)']:.1f}%")
    print(f"Threshold: {threshold:.3f}\n")
    
    return best_model_id, threshold, best_row['Name']


def create_realistic_scenario(scenario_type, test_features):
    """
    Create realistic 60-day scenarios by perturbing real data
    
    Uses real feature distributions and correlations
    """
    
    # Start from a random real 60-day window
    start_idx = np.random.randint(0, len(test_features) - 60)
    base_sequence = test_features[start_idx:start_idx+60].copy()
    
    if scenario_type == 1:
        # Strong Bull: Boost returns, lower vol, raise RSI
        print("\nScenario 1: STRONG BULL MARKET")
        print("  - Enhanced positive returns")
        print("  - Reduced volatility")
        print("  - Bullish technical indicators")
        
        base_sequence[:, 0] += 0.002  # Boost log_return
        base_sequence[:, 11] -= 3     # Lower VIX
        base_sequence[:, 7] += 10     # Raise RSI
        base_sequence[:, 8] += 0.1    # Positive MACD
        
    elif scenario_type == 2:
        # Bear Market: Negative returns, high vol, low RSI
        print("\nScenario 2: BEAR MARKET DECLINE")
        print("  - Negative returns")
        print("  - Elevated volatility")
        print("  - Bearish technical indicators")
        
        base_sequence[:, 0] -= 0.003  # Negative log_return
        base_sequence[:, 11] += 8     # Higher VIX
        base_sequence[:, 7] -= 15     # Lower RSI
        base_sequence[:, 8] -= 0.2    # Negative MACD
        base_sequence[:, 19] += 0.5   # Wider credit spreads
        
    elif scenario_type == 3:
        # Recovery: Improving trend
        print("\nScenario 3: MARKET RECOVERY")
        print("  - Gradually improving returns")
        print("  - Normalizing volatility")
        print("  - Recovering technicals")
        
        for i in range(60):
            progress = i / 59
            base_sequence[i, 0] += 0.001 * progress  # Improving returns
            base_sequence[i, 11] -= 4 * progress     # Declining VIX
            base_sequence[i, 7] += 8 * progress      # Rising RSI
        
    elif scenario_type == 4:
        # High Volatility: Amplify oscillations
        print("\nScenario 4: HIGH VOLATILITY")
        print("  - Amplified price swings")
        print("  - Elevated VIX")
        print("  - Choppy indicators")
        
        base_sequence[:, 0] *= 1.5    # Amplify returns
        base_sequence[:, 11] += 6     # Higher VIX
        base_sequence[:, 12] += 0.01  # Higher ATR
        
    elif scenario_type == 5:
        # Calm: Dampen everything
        print("\nScenario 5: CALM LOW-VOL MARKET")
        print("  - Subdued returns")
        print("  - Low volatility")
        print("  - Stable indicators")
        
        base_sequence[:, 0] *= 0.3    # Dampen returns
        base_sequence[:, 11] -= 4     # Lower VIX
        base_sequence[:, 12] -= 0.005 # Lower ATR
    
    # Clip to reasonable ranges to avoid extreme values
    base_sequence[:, 0] = np.clip(base_sequence[:, 0], -0.05, 0.05)  # Returns
    base_sequence[:, 11] = np.clip(base_sequence[:, 11], 10, 50)     # VIX
    base_sequence[:, 7] = np.clip(base_sequence[:, 7], 20, 80)       # RSI
    
    return base_sequence


def predict_scenario(model, device, scenario_days, threshold):
    """Make prediction"""
    x = torch.FloatTensor(scenario_days).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(x)
        outputs_binary = outputs[:, :2]
        probs = torch.softmax(outputs_binary, dim=1).cpu().numpy()[0]
    
    prediction = 0 if probs[0] > threshold else 1
    return prediction, probs


def main():
    print("\n" + "="*70)
    print("MARKET REGIME PREDICTION DEMO")
    print("Realistic Scenarios Based on Real Data Distributions")
    print("="*70)
    
    # Find best model
    best_model_id, threshold, model_name = find_best_model()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = create_model(best_model_id, device)
    
    checkpoint_path = config.CHECKPOINT_DIR / f"{best_model_id}_binary_focal" / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {device}\n")
    
    # Load test data for feature distributions
    test_data = pd.read_csv(
        config.REGIME_DATA_DIR / "test_labeled_engineered_binary.csv",
        index_col=0, parse_dates=True
    )
    test_features = test_data[config.ENGINEERED_FEATURES].values
    
    print("="*70)
    print("RUNNING 5 REALISTIC SCENARIOS")
    print("="*70)
    
    results = []
    
    for scenario_type in range(1, 6):
        print("\n" + "-"*70)
        
        scenario_days = create_realistic_scenario(scenario_type, test_features)
        
        prediction, probs = predict_scenario(model, device, scenario_days, threshold)
        
        regime = "BULLISH" if prediction == 1 else "BEARISH"
        confidence = probs[prediction] * 100
        
        print(f"\n  PREDICTION: {regime}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Probabilities: Bearish={probs[0]:.3f}, Bullish={probs[1]:.3f}")
        
        results.append({
            'Scenario': scenario_type,
            'Description': ['Bull Market', 'Bear Decline', 'Recovery', 'High Volatility', 'Calm Market'][scenario_type-1],
            'Prediction': regime,
            'Confidence': f"{confidence:.1f}%",
            'Prob_Bearish': f"{probs[0]:.3f}",
            'Prob_Bullish': f"{probs[1]:.3f}"
        })
    
    print("\n" + "="*70)
    print("DEMO RESULTS SUMMARY")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Test MCC: Available in binary_model_results.csv")
    print(f"Threshold: {threshold:.3f}\n")
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    results_df.to_csv(config.DATA_DIR / "demo_results.csv", index=False)
    print(f"\nSaved: demo_results.csv")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()