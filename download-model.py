#!/usr/bin/env python3
"""
Download and convert the sentence-transformers model to ONNX format.

Requirements:
    pip install optimum[exporters] transformers torch
"""

import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("models")

def main():
    print(f"Downloading {MODEL_NAME}...")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Download and convert model to ONNX
    print("Converting to ONNX format...")
    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_NAME,
        export=True
    )

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Save ONNX model
    model_path = OUTPUT_DIR / "model.onnx"
    model.save_pretrained(OUTPUT_DIR)

    # The model.onnx file should now be in the models directory
    # Rename if needed
    onnx_file = OUTPUT_DIR / "model.onnx"
    if not onnx_file.exists():
        # Look for the exported file
        for f in OUTPUT_DIR.glob("*.onnx"):
            f.rename(onnx_file)
            break

    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ Model downloaded successfully!")
    print(f"   Model: {OUTPUT_DIR / 'model.onnx'}")
    print(f"   Tokenizer: {OUTPUT_DIR / 'tokenizer.json'}")
    print(f"\nYou can now run: cargo run")

if __name__ == "__main__":
    main()
