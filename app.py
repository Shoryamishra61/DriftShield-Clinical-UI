"""
Hugging Face Spaces Entrypoint for DriftShield Clinical UI.
"""
import os

# Fix for OpenMP duplicate library crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import the Gradio app from our package
from app.gradio_app import demo

if __name__ == "__main__":
    demo.launch()
