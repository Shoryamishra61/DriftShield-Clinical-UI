"""
DriftShield Test Runner.

Helper script to execute pytest with the correct path configuration.
"""

import sys
import pytest
from pathlib import Path

# Ensure the root directory is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

if __name__ == "__main__":
    sys.exit(pytest.main(["tests/", "-v"]))
