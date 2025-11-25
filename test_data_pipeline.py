"""Test script for data_pipeline.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_pipeline import DataPipeline

def main():
    print("\n" + "="*60)
    print("TESTING DATA PIPELINE")
    print("="*60)
    
    pipeline = DataPipeline()
    
    try:
        data = pipeline.run_pipeline()
        
        # Verification
        print("\n" + "="*60)
        print("VERIFICATION")
        print("="*60)
        
        assert 'train' in data and 'val' in data and 'test' in data
        print("    All splits present: PASS")
        
        assert len(data['train']) > 0 and len(data['val']) > 0 and len(data['test']) > 0
        print("    All splits non-empty: PASS")
        
        required_cols = ['log_return', 'volatility']
        for col in required_cols:
            assert col in data['full'].columns
        print(f"    Required columns present: PASS")
        
        assert data['train'].index[-1] < data['val'].index[0]
        assert data['val'].index[-1] < data['test'].index[0]
        print("    No temporal leakage: PASS")
        
        nan_counts = data['full'].isna().sum()
        if nan_counts.sum() > 0:
            print(f"\n    Warning: NaN values detected:")
            print(nan_counts[nan_counts > 0])
        else:
            print("    No NaN values: PASS")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"    Total: {len(data['full'])} days")
        print(f"    Train: {len(data['train'])} ({len(data['train'])/len(data['full'])*100:.1f}%)")
        print(f"    Val:   {len(data['val'])} ({len(data['val'])/len(data['full'])*100:.1f}%)")
        print(f"    Test:  {len(data['test'])} ({len(data['test'])/len(data['full'])*100:.1f}%)")
        print(f"    Features: {len(data['full'].columns)}")
        
        print("\n" + "="*60)
        print("TEST SUCCESSFUL - Proceed to feature_engineering.py")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)