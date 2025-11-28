"""Train All Models - Binary Classification"""

import subprocess
import pandas as pd
from pathlib import Path
import config

def train_all_binary_models():
    """Train all 7 models on binary classification task"""
    
    print("\n" + "="*70)
    print("TRAINING ALL MODELS - BINARY CLASSIFICATION")
    print("="*70)
    
    # All 7 models
    all_models = config.list_available_models()
    
    print(f"\nModels to train: {len(all_models)}")
    for i, model_id in enumerate(all_models, 1):
        model_config = config.get_model_config(model_id)
        print(f"  {i}. {model_config['name']} ({model_id})")
    
    print("\n" + "="*70)
    
    # Track results
    results = []
    
    for i, model_id in enumerate(all_models, 1):
        model_config = config.get_model_config(model_id)
        model_name = model_config['name']
        
        print(f"\n{'='*70}")
        print(f"[{i}/{len(all_models)}] TRAINING: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Run training
            result = subprocess.run(
                ['python', 'train_binary.py', '--model_id', model_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✓ {model_name} training completed successfully")
                results.append({
                    'model_id': model_id,
                    'name': model_name,
                    'status': 'success'
                })
            else:
                print(f"✗ {model_name} training failed")
                print(f"Error: {result.stderr}")
                results.append({
                    'model_id': model_id,
                    'name': model_name,
                    'status': 'failed'
                })
        
        except Exception as e:
            print(f"✗ {model_name} training failed with exception: {e}")
            results.append({
                'model_id': model_id,
                'name': model_name,
                'status': 'failed'
            })
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\nTotal Models: {len(all_models)}")
    print(f"Successful:   {success_count}")
    print(f"Failed:       {failed_count}")
    
    print("\nDetailed Results:")
    for r in results:
        status_icon = "✓" if r['status'] == 'success' else "✗"
        print(f"  {status_icon} {r['name']} - {r['status']}")
    
    print("\n" + "="*70)
    print("Next step: Run compare_binary_models.py to see performance comparison")
    print("="*70 + "\n")


if __name__ == "__main__":
    train_all_binary_models()