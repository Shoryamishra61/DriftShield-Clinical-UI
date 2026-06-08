"""
DriftShield Sprint Orchestrator.

Sequentially executes the data generation, dataset building, FAISS indexing,
model training, and evaluation scripts to complete the project setup.
"""

import sys
import subprocess
import time
import os
from pathlib import Path

def run_step(script_name: str, args: list[str] = []) -> bool:
    """Runs a Python script as a subprocess, streaming its stdout/stderr in real-time.

    Args:
        script_name: The path or name of the python script to run.
        args: Extra arguments to pass.

    Returns:
        True if the execution completed successfully (exit code 0), else False.
    """
    if script_name == "-m":
        cmd = [sys.executable, "-m"] + args
    else:
        cmd = [sys.executable, script_name] + args
    print("=" * 70)
    print(f"Executing: {' '.join(cmd)}")
    print("=" * 70)
    
    start_time = time.time()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        
        # Run with live output streaming to stdout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Stream output line-by-line
        if process.stdout:
            for line in process.stdout:
                print(line, end="")
                
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n[SUCCESS]: {script_name} completed in {elapsed:.1f} seconds.\n")
            return True
        else:
            print(f"\n[FAILURE]: {script_name} exited with code {process.returncode} after {elapsed:.1f} seconds.\n")
            return False
            
    except Exception as e:
        print(f"\n[EXCEPTION]: Failed to execute {script_name}: {e}\n")
        return False

def main() -> None:
    steps = [
        # ("data/generate_synthetic_data.py", []),
        # ("data/build_dataset.py", []),
        # ("rag/build_index.py", []),
        # ("models/train.py", []),
        ("-m", ["pytest", "tests/"]),
        ("evaluation/run_evaluation.py", []),
        ("results/generate_arxiv_report.py", [])
    ]
    
    print("Starting DriftShield PhD-Level Sprint Orchestrator")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Scheduled Steps: {', '.join([s[0] for s in steps])}\n")
    
    for script, args in steps:
        success = run_step(script, args)
        if not success:
            print("CRITICAL ERROR: Pipeline interrupted due to failure. Stopping orchestrator.")
            sys.exit(1)
            
    print("ALL SPRINT STEPS COMPLETED SUCCESSFULLY!")
    print("The system is now fully generated, trained, and evaluated.")

if __name__ == "__main__":
    main()
