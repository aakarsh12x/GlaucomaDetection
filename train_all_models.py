import os
import subprocess
import sys
import time

def run_training_sequential():
    """
    Sequentially runs the training scripts for all 4 model architectures.
    """
    models = [
        {"name": "YOLOv11", "path": "models/yolov11"},
        {"name": "MambaOut", "path": "models/mamba_out"},
        {"name": "VisionMamba", "path": "models/vision_mamba"}
    ]

    print("="*60)
    print("STARTING GLOBAL MULTI-MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"Models to train: {[m['name'] for m in models]}")
    
    start_time = time.time()
    results = []

    for model in models:
        model_name = model["name"]
        model_dir = model["path"]
        
        print(f"\n>>> Training Model: {model_name}...")
        print(f"Target Directory: {model_dir}")
        
        # Determine the correct python command
        python_cmd = sys.executable if sys.executable else "python"
        
        try:
            # Change to model directory and run train.py
            # Pass epochs parameter to scripts
            process = subprocess.run(
                [python_cmd, "train.py"],
                cwd=model_dir,
                check=True
            )
            print(f"SUCCESS: {model_name} training completed.")
            results.append((model_name, "SUCCESS"))
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {model_name} training failed with exit code {e.returncode}.")
            results.append((model_name, f"FAILED (Code {e.returncode})"))
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while training {model_name}: {e}")
            results.append((model_name, "CRITICAL ERROR"))

    end_time = time.time()
    duration = (end_time - start_time) / 60

    print("\n" + "="*60)
    print("TRAINING PIPELINE SUMMARY")
    print("="*60)
    print(f"Total Duration: {duration:.2f} minutes")
    for name, status in results:
        print(f" - {name}: {status}")
    print("="*60)

if __name__ == "__main__":
    run_training_sequential()
