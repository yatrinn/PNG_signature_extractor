#!/usr/bin/env python3
"""
PNG Signature Extractor

This tool extracts signatures from PNG images by removing the white background.
It works by converting the image to RGBA and computing the per‐pixel difference from pure white.
Pixels that are nearly white (i.e. whose combined difference from [255,255,255] is below a specified threshold)
are set to fully transparent. This method ensures that the white background is removed completely,
leaving only the signature in its original color.

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

def process_image(input_path, output_path, diff_threshold=30):
    """
    Process an image by removing its white background.
    
    For each pixel, the sum of differences from pure white (255,255,255) is computed.
    If the sum is less than diff_threshold * 3 (i.e. the pixel is nearly white),
    the pixel's alpha is set to 0 (transparent). Otherwise, the pixel is kept opaque.
    """
    try:
        # Open image and convert to RGBA
        im = Image.open(input_path).convert("RGBA")
        data = np.array(im)  # shape (height, width, 4)
        
        # Compute difference from white for each pixel for R, G, B channels
        rgb = data[:, :, :3].astype(np.int16)
        diff = 255 - rgb  # difference from white
        sum_diff = np.sum(diff, axis=2)  # total difference per pixel
        
        # Identify nearly-white pixels: if sum_diff is below threshold * 3, mark as background
        background = sum_diff < (diff_threshold * 3)
        data[background, 3] = 0  # Set alpha channel to 0 for background pixels
        
        new_im = Image.fromarray(data, "RGBA")
        new_im.save(output_path)
        logging.info(f"Output saved to: {output_path}")
        return True
    except Exception as e:
        logging.exception(f"Error processing image {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Extract signatures from PNG images by removing the white background using numpy."
    )
    parser.add_argument("input_folder", help="Path to the input folder containing PNG images.")
    parser.add_argument("output_folder", help="Path to the output folder for processed images.")
    parser.add_argument("--diff_threshold", type=int, default=30, help="Difference threshold per channel (default: 30)")
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
            if process_image(input_path, output_path, args.diff_threshold):
                processed += 1
            else:
                failed += 1
    logging.info(f"Processing complete. Success: {processed}, Failed: {failed}")

if __name__ == "__main__":
    main()
