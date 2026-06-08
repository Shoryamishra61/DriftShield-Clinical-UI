"""
Wait and Run Script.
Waits for the background model training to finish by checking for the 
test_results.json file, and then automatically runs the rest of the pipeline.
"""
import time
import os
import subprocess
from pathlib import Path

def wait_and_run():
    print("Waiting for BioBERT model training to complete...")
    test_results_path = Path("checkpoints/test_results.json")
    
    # Wait until the test_results.json file is created (which happens at the very end of train.py)
    while not test_results_path.exists():
        time.sleep(10)
        
    print("Training complete! Model saved.")
    print("Running unit tests...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    
    # Run tests
    subprocess.run(["python", "-m", "pytest", "tests/"], env=env)
    
    print("\nRunning Sprint Orchestrator (Evaluation & Report Generation)...")
    # Run the remaining evaluation and reporting scripts
    subprocess.run(["python", "sprint_orchestrator.py"], env=env)
    
    print("\nAll tasks completed autonomously! Project is ready.")

if __name__ == "__main__":
    wait_and_run()
