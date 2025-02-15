#!/usr/bin/env python3
"""
PNG Signature Extractor

This tool processes all PNG images in a specified input folder to extract signatures 
by removing the white background. It uses Pillow to convert each image to RGBA and 
sets nearly white pixels (R, G, and B >= threshold) to transparent. This should remove 
the white background completely and preserve the signature (in black, blue, etc.).

Optimal input: A high-quality PNG screenshot of the signature on a white background.

Author: Yannik Trinn
License: MIT License
"""

import os
import argparse
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image(input_path, output_path, threshold=240):
    try:
        # Open image and convert to RGBA
        image = Image.open(input_path).convert("RGBA")
        datas = image.getdata()
        newData = []
        for item in datas:
            # If R, G, and B are all above the threshold, treat pixel as background
            if item[0] >= threshold and item[1] >= threshold and item[2] >= threshold:
                newData.append((item[0], item[1], item[2], 0))
            else:
                newData.append(item)
        image.putdata(newData)
        image.save(output_path, "PNG")
        logging.info(f"Output saved to: {output_path}")
        return True
    except Exception as e:
        logging.exception(f"Error processing image {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Extract signatures from all PNG images in a folder by removing the white background using Pillow."
    )
    parser.add_argument("input_folder", help="Path to the input folder containing PNG images.")
    parser.add_argument("output_folder", help="Path to the output folder for processed images.")
    parser.add_argument("--threshold", type=int, default=240, help="Threshold for white pixels (default: 240)")
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
