"""Train All Models: Train all 7 transformer architectures"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import subprocess

def main():
    print("\n" + "="*70)
    print("TRAINING ALL MODELS")
    print("="*70)
    
    models = config.list_available_models()
    
    print(f"\nFound {len(models)} models to train:")
    for i, model_id in enumerate(models, 1):
        model_config = config.get_model_config(model_id)
        print(f"  {i}. {model_config['name']} ({model_id})")
    
    print("\n" + "="*70)
    
    results = []
    
    for i, model_id in enumerate(models, 1):
        model_config = config.get_model_config(model_id)
        print(f"\n[{i}/{len(models)}] Training: {model_config['name']}")
        print("-"*70)
        
        try:
            result = subprocess.run(
                ['python', 'scripts/train_binary_focal.py', '--model_id', model_id],
                check=True,
                capture_output=False
            )
            results.append({'model_id': model_id, 'status': 'SUCCESS'})
            print(f"\n✓ {model_id} completed successfully")
            
        except subprocess.CalledProcessError as e:
            results.append({'model_id': model_id, 'status': 'FAILED'})
            print(f"\n✗ {model_id} failed with error")
            continue
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    success = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']
    
    print(f"\nSuccessful: {len(success)}/{len(models)}")
    for r in success:
        print(f"  ✓ {r['model_id']}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(models)}")
        for r in failed:
            print(f"  ✗ {r['model_id']}")
    
    print("\n" + "="*70)
    print("ALL MODELS TRAINED")
    print("="*70)
    print("\nNext step: python scripts/tune_all_thresholds.py")
    print("\n")


if __name__ == "__main__":
    main()