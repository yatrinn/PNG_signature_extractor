#!/usr/bin/env python3
"""
PNG Signature Extractor

This tool processes all PNG images in a specified input folder to extract signatures 
from documents using multiple thresholding methods and adaptive logic. It attempts 
to select the best mask based on the extracted area and applies morphological operations 
to refine the result. While extraction is improved, optimal input conditions (a high-quality, 
large PNG screenshot of the signature on a white background) are still required.

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

def get_mask_adaptive(gray):
    """Return mask from adaptive thresholding."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    return adaptive

def get_mask_otsu(gray):
    """Return mask from Otsu's thresholding."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return otsu

def refine_mask(mask):
    """Apply morphological operations to clean up the mask."""
    kernel = np.ones((3, 3), np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_close

def select_best_mask(gray):
    """
    Compute masks using adaptive and Otsu methods, then choose the one with 
    an extracted area (nonzero pixels) within a plausible range.
    """
    mask_adaptive = refine_mask(get_mask_adaptive(gray))
    mask_otsu = refine_mask(get_mask_otsu(gray))
    
    # Combine both masks using bitwise OR
    combined = cv2.bitwise_or(mask_adaptive, mask_otsu)
    
    # Count nonzero pixels in each mask
    count_adaptive = cv2.countNonZero(mask_adaptive)
    count_otsu = cv2.countNonZero(mask_otsu)
    count_combined = cv2.countNonZero(combined)
    
    # Expected area range (these values might be tuned based on image resolution)
    # For example, the signature area should be at least 1% and at most 50% of the image area.
    image_area = gray.shape[0] * gray.shape[1]
    min_area = 0.01 * image_area
    max_area = 0.5 * image_area
    
    # Choose the mask that gives a nonzero count within the expected range, preferring combined.
    candidates = [(mask_adaptive, count_adaptive), (mask_otsu, count_otsu), (combined, count_combined)]
    best_mask = None
    for mask, count in candidates:
        if min_area < count < max_area:
            best_mask = mask
            break
    if best_mask is None:
        # If none is in range, fall back to combined mask
        best_mask = combined
    return best_mask

def process_image(input_path, output_path):
    """Process a single image to extract the signature robustly."""
    try:
        image = cv2.imread(input_path)
        if image is None:
            logging.error(f"Could not load image {input_path}")
            return False

        # Convert to grayscale and equalize histogram to improve contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # Select the best mask from multiple thresholding methods
        mask = select_best_mask(gray_eq)

        # Check if the mask has a plausible area
        if cv2.countNonZero(mask) < 50:
            logging.warning(f"Extracted mask for {input_path} is nearly empty. Extraction may have failed.")

        # Create BGRA image and apply the mask to the alpha channel
        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image_bgra[:, :, 3] = mask

        # Save the resulting image
        cv2.imwrite(output_path, image_bgra)
        logging.info(f"Output saved to: {output_path}")
        return True
    except Exception as e:
        logging.exception(f"Exception processing image {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Extract signatures from all images in a folder using robust thresholding."
    )
    parser.add_argument("input_folder", help="Path to the input folder containing images (PNG recommended).")
    parser.add_argument("output_folder", help="Path to the output folder for processed images.")
    args = parser.parse_args()

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
