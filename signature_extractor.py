#!/usr/bin/env python3
"""
PNG Signature Extractor using OpenCV inRange() with fixed output folder

This tool processes all PNG images in a specified input folder to extract signatures 
by removing the white background. It uses OpenCV’s inRange() to create a mask for white 
pixels and then inverts that mask to set those pixels fully transparent.
All output images are stored in the fixed folder:
  /Users/Notebook/2025.02.15_PNG_signature_extractor/Output

Optimal input: A high-quality PNG screenshot of a signature on a white background.

Author: Yannik Trinn
License: MIT License
"""

import cv2
import numpy as np
import os
import argparse
import logging

# Set logging to DEBUG for detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image(input_path, output_path, lower_bound=240):
    logging.debug(f"Processing image: {input_path}")
    # Load the image in BGR format
    img = cv2.imread(input_path)
    if img is None:
        logging.error(f"Failed to load image: {input_path}")
        return False

    # Convert image to BGRA (to add an alpha channel)
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Define lower and upper bounds for white
    lower = np.array([lower_bound, lower_bound, lower_bound], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)
    
    # Create a mask where pixels in the white range are 255
    white_mask = cv2.inRange(img, lower, upper)
    logging.debug(f"White mask created with lower_bound={lower_bound}")
    
    # Invert the mask: white pixels become 0 (transparent), others 255 (opaque)
    alpha = cv2.bitwise_not(white_mask)
    logging.debug("Alpha mask computed by inverting the white mask")
    
    # Optional: smooth the alpha channel to reduce artifacts
    alpha = cv2.medianBlur(alpha, 3)
    
    # Set the alpha channel in the image
    img_bgra[:, :, 3] = alpha

    # Save the output image
    success = cv2.imwrite(output_path, img_bgra)
    if success:
        logging.info(f"Output saved to: {output_path}")
    else:
        logging.error(f"Failed to save output to: {output_path}")
    return success

def main():
    parser = argparse.ArgumentParser(
        description="Extract signatures from PNG images by removing the white background using OpenCV inRange()."
    )
    parser.add_argument("input_folder", help="Path to the input folder containing PNG images.")
    parser.add_argument("--lower_bound", type=int, default=240,
                        help="Lower bound for white detection (default: 240)")
    args = parser.parse_args()

    # Use the fixed output folder
    output_folder = "/Users/Notebook/2025.02.15_PNG_signature_extractor/Output"
    if not os.path.exists(output_folder):
        logging.error(f"Output folder does not exist: {output_folder}")
        return

    logging.debug(f"Input folder: {args.input_folder}")
    logging.debug(f"Fixed output folder: {output_folder}")
    
    files = os.listdir(args.input_folder)
    logging.info(f"Found {len(files)} items in input folder: {files}")
    
    processed = 0
    failed = 0
    for filename in files:
        if filename.lower().endswith(".png"):
            logging.debug(f"Found PNG file: {filename}")
            input_path = os.path.join(args.input_folder, filename)
            name, _ = os.path.splitext(filename)
            output_filename = f"{name}_extracted.png"
            output_path = os.path.join(output_folder, output_filename)
            if process_image(input_path, output_path, args.lower_bound):
                processed += 1
            else:
                failed += 1

    logging.info(f"Processing complete. Success: {processed}, Failed: {failed}")

if __name__ == "__main__":
    main()
