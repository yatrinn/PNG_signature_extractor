#!/usr/bin/env python3
"""
PNG Signature Extractor

This tool processes all PNG images in a specified input folder to extract signatures by removing the white background.
It converts each image to RGBA and then computes the Euclidean distance from pure white for every pixel.
If the distance is below a specified threshold (default: 50), the pixel is set to fully transparent.
This ensures that only the signature (black, blue, etc.) remains visible.

Optimal input: A high-quality PNG screenshot of the signature on a white background.

Author: Yannik Trinn
License: MIT License
"""

import os
import argparse
import logging
import numpy as np
from PIL import Image

# Configure logging for informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image(input_path, output_path, threshold=50):
    """
    Process an image by removing its white background.
    
    Parameters:
      input_path: Path to the input PNG image.
      output_path: Path to save the processed PNG image.
      threshold: Maximum Euclidean distance from white (255,255,255) for a pixel to be considered background.
                 Lower values mean stricter removal of white. Default is 50.
    
    Returns:
      True if processing succeeds, False otherwise.
    """
    try:
        # Open the image and convert to RGBA
        im = Image.open(input_path).convert("RGBA")
        data = np.array(im)  # shape (height, width, 4)

        # Extract RGB channels and convert to float32 for distance calculation
        rgb = data[:, :, :3].astype(np.float32)
        
        # Compute Euclidean distance from pure white (255,255,255)
        diff = np.linalg.norm(rgb - 255, axis=2)
        
        # Create a new alpha channel: pixels with diff < threshold become transparent (0), else opaque (255)
        new_alpha = np.where(diff < threshold, 0, 255).astype(np.uint8)
        data[:, :, 3] = new_alpha

        # Create a new image from the updated array and save it
        new_im = Image.fromarray(data, mode="RGBA")
        new_im.save(output_path)
        logging.info(f"Output saved to: {output_path}")
        return True
    except Exception as e:
        logging.exception(f"Failed to process {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Extract signatures from PNG images by removing the white background."
    )
    parser.add_argument("input_folder", help="Path to the input folder containing PNG images.")
    parser.add_argument("output_folder", help="Path to the output folder for processed images.")
    parser.add_argument("--threshold", type=int, default=50, help="Euclidean distance threshold from white. Default is 50.")
    args = parser.parse_args()

    # Ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    processed = 0
    failed = 0
    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(args.input_folder, filename)
            name, _ = os.path.splitext(filename)
            output_filename = f"{name}_extracted.png"
            output_path = os.path.join(args.output_folder, output_filename)
            if process_image(input_path, output_path, args.threshold):
                processed += 1
            else:
                failed += 1

    logging.info(f"Processing complete. Success: {processed}, Failed: {failed}")

if __name__ == "__main__":
    main()
