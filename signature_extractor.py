#!/usr/bin/env python3
"""
PNG Signature Extractor

This tool processes all images in a specified input folder to extract signatures 
from documents using Otsu's thresholding and morphological operations.
It saves the resulting images as PNG files with a transparent background and 
appends "_extracted" to the original filename.

Author: Yannik Trinn
License: MIT License
"""

import cv2
import numpy as np
import os
import argparse

def process_image(input_path, output_path):
    """Process a single image to extract the signature."""
    # Load the input image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image {input_path}")
        return

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding (inverted to capture a dark signature on a white background)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up the mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Convert the original image to BGRA (to add an alpha channel)
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_bgra[:, :, 3] = mask  # Set the alpha channel to the mask

    # Save the resulting image as PNG
    cv2.imwrite(output_path, image_bgra)
    print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract signatures from all images in a folder using thresholding."
    )
    parser.add_argument("input_folder", help="Path to the input folder containing images.")
    parser.add_argument("output_folder", help="Path to the output folder for processed images.")
    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Process each image file in the input folder
    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            input_path = os.path.join(args.input_folder, filename)
            name, _ = os.path.splitext(filename)
            output_filename = f"{name}_extracted.png"
            output_path = os.path.join(args.output_folder, output_filename)
            process_image(input_path, output_path)

if __name__ == "__main__":
    main()
