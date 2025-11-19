"""
Run All Steps at Once
================================================

This script runs the entire MNIST to TT-NN conversion pipeline:
1. Train PyTorch model (if not already trained)
2. Convert weights to TT-NN format
3. Run TT-NN inference
4. Validate results

"""

import os
import subprocess
import sys

def run_script(script_name, description):
    """Run a Python script and display its output"""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    try:
        result = subprocess.run(
            [python_exe, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n[SUCCESS] {description} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {description} - FAILED")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] {description} - ERROR")
        print(f"Error: {e}")
        return False


def check_model_exists():
    """Check if model is already trained"""
    model_path = os.path.join('models', 'mnist_mlp.pth')
    return os.path.exists(model_path)


def main():
    """Run the complete pipeline"""
    print("=" * 70)
    print("MNIST to TT-NN COMPLETE PIPELINE")
    print("=" * 70)
    print("\nThis will run all 4 scripts in sequence:")
    print("  1. Train PyTorch Model (if needed)")
    print("  2. Convert Weights to TT-NN Format")
    print("  3. Run TT-NN Inference")
    print("  4. Validate Results")
    print()
    
    # Check if we need to train
    if check_model_exists():
        print("[OK] Found existing trained model (models/mnist_mlp.pth)")
        response = input("Do you want to retrain? (y/n): ").lower().strip()
        skip_training = response != 'y'
    else:
        print("[INFO] No trained model found - will train from scratch")
        skip_training = False
    
    input("\nPress Enter to start the pipeline...")
    
    results = []
    
    # Step 1: Train PyTorch Model (optional)
    if not skip_training:
        success = run_script('1_pytorch_mlp.py', 'Step 1: Train PyTorch Model')
        results.append(('Train Model', success))
    else:
        print("\n" + "=" * 70)
        print("SKIPPING: Step 1 - Using existing trained model")
        print("=" * 70)
        results.append(('Train Model', True))
    
    # Step 2: Convert Weights
    success = run_script('2_convert_weights.py', 'Step 2: Convert Weights to TT-NN Format')
    results.append(('Convert Weights', success))
    
    if not success:
        print("\n[ERROR] Pipeline stopped due to weight conversion failure")
        return
    
    # Step 3: Run TT-NN Inference
    success = run_script('3_ttnn_inference.py', 'Step 3: Run TT-NN Inference')
    results.append(('TT-NN Inference', success))
    
    # Step 4: Validate
    success = run_script('4_validate.py', 'Step 4: Validate Results')
    results.append(('Validation', success))
    
    # Summary
    print("\n\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    
    for step_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status:10} - {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n" + "=" * 70)
        print("[SUCCESS] All steps completed successfully!")
        print("=" * 70)
        print("\nOutput Files:")
        print("   - models/mnist_mlp.pth           (Trained PyTorch model)")
        print("   - models/ttnn_weights/           (Converted TT-NN weights)")
        print("   - models/predictions.png         (Sample predictions)")
        print("\nDocumentation:")
        print("   - README.md                      (Project overview)")
        print("   - SUMMARY.md                     (Task summary)")
        print("\nReady for submission!")
    else:
        print("\n" + "=" * 70)
        print("[WARNING] Some steps failed. Please check the errors above.")
        print("=" * 70)


if __name__ == '__main__':
    main()
