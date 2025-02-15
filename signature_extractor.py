#!/usr/bin/env python3
"""
PNG Signature Extractor using rembg

This tool processes all images from a fixed Input folder to extract objects (e.g., signatures)
by removing the white background using the rembg library (which leverages a pre-trained U²-Net model).
All processed images are saved as PNG files (with transparent backgrounds) in a fixed Output folder.

Fixed paths:
  Input folder:  /Users/Notebook/2025.02.15_PNG_signature_extractor/Input
  Output folder: /Users/Notebook/2025.02.15_PNG_signature_extractor/Output

Optimal input: A high-quality image (PNG, JPG, or JPEG) of a signature on a white background.

Author: Yannik Trinn
License: MIT License
"""

import os
import logging
from rembg import remove
from PIL import Image
import io

# Configure logging for output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fixed folder paths
INPUT_FOLDER = "/Users/Notebook/2025.02.15_PNG_signature_extractor/Input"
OUTPUT_FOLDER = "/Users/Notebook/2025.02.15_PNG_signature_extractor/Output"

def process_image(input_path, output_path):
    """
    Process an image by removing its white background using rembg.
    The resulting image is saved as a PNG with a transparent background.
    """
    try:
        with open(input_path, 'rb') as infile:
            input_data = infile.read()
        # Remove background using the pre-trained U²-Net model
        output_data = remove(input_data)
        # Load the result as an image
        output_image = Image.open(io.BytesIO(output_data))
        output_image.save(output_path, "PNG")
        logging.info(f"Output saved to: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        return False

def main():
    # Ensure the output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = os.listdir(INPUT_FOLDER)
    logging.info(f"Found {len(files)} items in Input folder: {files}")
    
    processed = 0
    failed = 0
    # Process files with extensions .png, .jpg, .jpeg (case-insensitive)
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(INPUT_FOLDER, filename)
            name, _ = os.path.splitext(filename)
            output_filename = f"{name}_extracted.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            if process_image(input_path, output_path):
                processed += 1
            else:
                failed += 1

    logging.info(f"Processing complete. Success: {processed}, Failed: {failed}")

if __name__ == "__main__":
    main()
