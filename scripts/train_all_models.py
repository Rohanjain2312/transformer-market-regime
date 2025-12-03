"""Train all Transformer models sequentially"""

import subprocess
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import config

def main():
    print("\n" + "="*70)
    print("TRAINING ALL MODELS")
    print("="*70)
    
    # Get Python executable path
    python_path = sys.executable
    print(f"\nUsing Python: {python_path}")
    
    models = config.list_available_models()
    print(f"\nModels to train: {len(models)}")
    for model_id in models:
        model_config = config.get_model_config(model_id)
        print(f"  - {model_id}: {model_config['name']}")
    
    results = {}
    
    for i, model_id in enumerate(models, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(models)}] {model_id}")
        print('='*70)
        
        try:
            result = subprocess.run(
                [python_path, 'scripts/train_model.py', '--model_id', model_id],
                check=True,
                capture_output=False
            )
            results[model_id] = 'SUCCESS'
            print(f"\n✓ {model_id} completed")
        except subprocess.CalledProcessError as e:
            results[model_id] = 'FAILED'
            print(f"\n✗ {model_id} failed")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    success_count = sum(1 for status in results.values() if status == 'SUCCESS')
    
    for model_id, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{symbol} {model_id}: {status}")
    
    print(f"\nTotal: {success_count}/{len(models)} models trained successfully")
    
    print("\n" + "="*70)
    print("✓ ALL MODELS TRAINING COMPLETE")
    print("="*70)
    print(f"\nNext step: python scripts/tune_all_thresholds.py")

if __name__ == "__main__":
    main()