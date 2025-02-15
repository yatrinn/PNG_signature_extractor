#!/usr/bin/env python3
"""
PNG Signature Extractor

This tool processes all PNG images in a specified input folder to extract signatures 
by completely removing the white background. It converts each image to RGBA and then, 
for each pixel, if the red, green, and blue values are greater than a specified threshold 
(default: 220), it sets that pixel’s alpha channel to 0 (fully transparent). 
This ensures that only the signature (in its original color) remains visible.

Optimal input: A high-quality PNG screenshot of the signature on a white background.

Author: Yannik Trinn
License: MIT License
"""

import os
import argparse
import logging
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image(input_path, output_path, threshold=220):
    try:
        # Open the image and convert to RGBA
        im = Image.open(input_path).convert("RGBA")
        data = np.array(im)  # shape: (height, width, 4)
        
        # Create a mask: if R, G, and B are all greater than threshold, mark as background
        white_mask = np.all(data[..., :3] > threshold, axis=-1)
        
        # Set alpha channel: background pixels become transparent (0), others fully opaque (255)
        data[..., 3] = np.where(white_mask, 0, 255).astype(np.uint8)
        
        # Create new image from modified data and save as PNG
        new_im = Image.fromarray(data, "RGBA")
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
    parser.add_argument("--threshold", type=int, default=220, help="Threshold for white detection (default: 220)")
    args = parser.parse_args()
    
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
