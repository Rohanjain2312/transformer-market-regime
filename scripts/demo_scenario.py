"""Demo: Realistic synthetic scenarios based on real data distributions"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import config
from src.model import create_model

# Check if in Colab
try:
    from IPython.display import Image, display
    IN_COLAB = True
except:
    IN_COLAB = False


def find_best_model():
    """Find best model based on test MCC"""
    results_path = config.DATA_DIR / "model_results.csv"
    
    if not results_path.exists():
        raise FileNotFoundError("Run compare_models.py first!")
    
    results_df = pd.read_csv(results_path)
    results_df = results_df.sort_values('MCC', ascending=False)
    
    best_row = results_df.iloc[0]
    best_model_id = best_row['Model ID']
    
    # Get threshold from tuning results
    threshold_path = config.DATA_DIR / "threshold_tuning_all_models.csv"
    threshold = 0.5
    if threshold_path.exists():
        threshold_df = pd.read_csv(threshold_path)
        threshold_row = threshold_df[threshold_df['model_id'] == best_model_id]
        if len(threshold_row) > 0:
            threshold = threshold_row['optimal_threshold'].values[0]
    
    return best_model_id, threshold, best_row['Name'], best_row['MCC']


def create_realistic_scenario(scenario_type, test_features):
    """Create realistic 60-day scenarios by perturbing real data"""
    
    # Start from random real 60-day window
    start_idx = np.random.randint(0, len(test_features) - 60)
    base_sequence = test_features[start_idx:start_idx+60].copy()
    
    scenarios = {
        1: ("STRONG BULL MARKET", lambda seq: (
            seq.__setitem__((slice(None), 0), seq[:, 0] + 0.002),  # Boost returns
            seq.__setitem__((slice(None), 7), np.clip(seq[:, 7] + 10, 20, 80))  # Raise RSI
        )),
        2: ("BEAR MARKET DECLINE", lambda seq: (
            seq.__setitem__((slice(None), 0), seq[:, 0] - 0.003),  # Negative returns
            seq.__setitem__((slice(None), 7), np.clip(seq[:, 7] - 15, 20, 80))  # Lower RSI
        )),
        3: ("MARKET RECOVERY", lambda seq: [
            seq.__setitem__((i, 0), seq[i, 0] + 0.001 * (i/59)) for i in range(60)
        ]),
        4: ("HIGH VOLATILITY", lambda seq: 
            seq.__setitem__((slice(None), 0), seq[:, 0] * 1.5)  # Amplify returns
        ),
        5: ("CALM LOW-VOL MARKET", lambda seq: 
            seq.__setitem__((slice(None), 0), seq[:, 0] * 0.3)  # Dampen returns
        )
    }
    
    name, transform = scenarios[scenario_type]
    transform(base_sequence)
    
    # Clip to reasonable ranges
    base_sequence[:, 0] = np.clip(base_sequence[:, 0], -0.05, 0.05)
    
    return name, base_sequence


def predict_scenario(model, device, scenario_days, threshold):
    """Make prediction"""
    x = torch.FloatTensor(scenario_days).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(x)
        outputs_binary = outputs[:, :2]
        probs = torch.softmax(outputs_binary, dim=1).cpu().numpy()[0]
    
    prediction = 0 if probs[0] > threshold else 1
    return prediction, probs


def plot_demo_results(results_df, model_name, mcc, save_path):
    """Create clean visualization of demo results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Predictions bar chart
    colors = ['red' if p == 'BEARISH' else 'green' for p in results_df['Prediction']]
    
    ax1.barh(results_df['Description'], 
             [float(c.strip('%')) for c in results_df['Confidence']], 
             color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Regime Predictions by Scenario', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # Add prediction labels
    for i, (desc, pred, conf) in enumerate(zip(results_df['Description'], 
                                                results_df['Prediction'], 
                                                results_df['Confidence'])):
        ax1.text(5, i, f'{pred}', va='center', fontweight='bold', 
                fontsize=10, color='white')
    
    # Plot 2: Probability heatmap
    probs = results_df[['Prob_Bearish', 'Prob_Bullish']].astype(float).values
    
    sns.heatmap(probs, annot=True, fmt='.3f', cmap='RdYlGn',
               xticklabels=['Bearish', 'Bullish'],
               yticklabels=results_df['Description'],
               cbar_kws={'label': 'Probability'}, ax=ax2,
               vmin=0, vmax=1, linewidths=1, linecolor='black')
    
    ax2.set_title('Prediction Probabilities', fontsize=13, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    plt.suptitle(f'Demo Predictions: {model_name} (MCC: {mcc:.3f})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {save_path.name}")
    
    if IN_COLAB:
        display(Image(filename=str(save_path)))
    else:
        plt.show()
    plt.close()


def main():
    print("\n" + "="*70)
    print("MARKET REGIME PREDICTION DEMO")
    print("="*70)
    
    # Find best model
    print("\nLoading best model...", end=" ", flush=True)
    best_model_id, threshold, model_name, mcc = find_best_model()
    print(f"✓ {model_name} (MCC: {mcc:.3f})")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = create_model(best_model_id, device)
    
    checkpoint_path = config.CHECKPOINT_DIR / best_model_id / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data for feature distributions
    test_data = pd.read_csv(
        config.REGIME_DATA_DIR / "test_labeled_engineered.csv",
        index_col=0, parse_dates=True
    )
    test_features = test_data[config.ENGINEERED_FEATURES].values
    
    print(f"Running 5 scenarios...", end=" ", flush=True)
    
    results = []
    
    for scenario_type in range(1, 6):
        scenario_name, scenario_days = create_realistic_scenario(scenario_type, test_features)
        prediction, probs = predict_scenario(model, device, scenario_days, threshold)
        
        regime = "BULLISH" if prediction == 1 else "BEARISH"
        confidence = probs[prediction] * 100
        
        results.append({
            'Scenario': scenario_type,
            'Description': scenario_name,
            'Prediction': regime,
            'Confidence': f"{confidence:.1f}%",
            'Prob_Bearish': f"{probs[0]:.3f}",
            'Prob_Bullish': f"{probs[1]:.3f}"
        })
    
    print("✓")
    
    results_df = pd.DataFrame(results)
    
    # Generate visualization
    print("Generating visualization...", end=" ", flush=True)
    viz_path = config.DATA_DIR / "demo_predictions.png"
    plot_demo_results(results_df, model_name, mcc, viz_path)
    
    # Print summary
    print("\n" + "="*70)
    print("DEMO RESULTS")
    print("="*70)
    print("\n" + results_df[['Description', 'Prediction', 'Confidence']].to_string(index=False))
    
    # Save results
    results_df.to_csv(config.DATA_DIR / "demo_results.csv", index=False)
    print(f"\n✓ Results saved: demo_results.csv")
    
    print("\n" + "="*70)
    print("✓ DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()