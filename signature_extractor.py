#!/usr/bin/env python3
"""
PNG Signature Extractor

This tool processes all images in a specified input folder to extract signatures 
from documents using an improved extraction algorithm that combines adaptive 
thresholding and Otsu's thresholding with morphological operations. It provides 
robust error handling and logging to improve reliability across different inputs.
It saves the resulting images as PNG files with a transparent background and 
appends "_extracted" to the original filename.

Optimal input: A high-quality, large PNG screenshot of the signature on a white background.

Author: Yannik Trinn
License: MIT License
"""

import cv2
import numpy as np
import os
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image(input_path, output_path):
    """Process a single image to extract the signature using improved thresholding."""
    try:
        image = cv2.imread(input_path)
        if image is None:
            logging.error(f"Could not load image {input_path}")
            return False

        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 11, 2)
        # Apply Otsu's thresholding
        _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine both thresholding results for robust extraction
        mask = cv2.bitwise_or(adaptive_thresh, otsu_thresh)

        # Morphological operations to clean the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Warn if the mask appears nearly empty
        if cv2.countNonZero(mask) < 50:
            logging.warning(f"Mask for image {input_path} is nearly empty. Extraction may have failed.")

        # Convert the original image to BGRA and set the alpha channel to the mask
        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image_bgra[:, :, 3] = mask

        # Save the output image
        cv2.imwrite(output_path, image_bgra)
        logging.info(f"Output saved to: {output_path}")
        return True
    except Exception as e:
        logging.exception(f"Exception processing image {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Extract signatures from all images in a folder using improved thresholding."
    )
    parser.add_argument("input_folder", help="Path to the input folder containing images.")
    parser.add_argument("output_folder", help="Path to the output folder for processed images.")
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    processed_count = 0
    failed_count = 0

    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            input_path = os.path.join(args.input_folder, filename)
            name, _ = os.path.splitext(filename)
            output_filename = f"{name}_extracted.png"
            output_path = os.path.join(args.output_folder, output_filename)
            if process_image(input_path, output_path):
                processed_count += 1
            else:
                failed_count += 1

    logging.info(f"Processing complete. Success: {processed_count}, Failed: {failed_count}")

if __name__ == "__main__":
    main()
