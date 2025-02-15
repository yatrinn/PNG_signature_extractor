#!/usr/bin/env python3
"""
Signature Extractor Script
This script processes all image files in the "Input" directory, extracts the signature
(or object) from a white background, and saves the result as a PNG with a transparent background
in the "Output" directory.

Assumptions:
  - The signature is dark (black/blue) on a white background.
  - The threshold value (240) may be adjusted if necessary.
"""

import cv2
import numpy as np
import os
import logging 

# Set up logging with a standard format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the input and output directories
INPUT_FOLDER = "Input"
OUTPUT_FOLDER = "Output"

def process_image(filename):
    """
    Processes a single image:
      - Reads the image.
      - Converts it to grayscale.
      - Applies a binary inverse threshold to isolate the dark signature.
      - Performs morphological operations to remove noise.
      - Creates an alpha channel based on the mask.
      - Saves the final image with transparency.
    """
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0] + "_extracted.png")
    
    # Read the image in color
    image = cv2.imread(input_path)
    if image is None:
        logging.error(f"Failed to read image: {input_path}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary inverse threshold
    # All pixels with a value greater than 240 (nearly white) become 0 (background),
    # while darker pixels (the signature) become 255.
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Perform morphological operations to reduce noise and fill small gaps
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Split the original image into its color channels
    b, g, r = cv2.split(image)
    # Use the mask as the alpha channel: signature pixels are opaque (255) and background is transparent (0)
    alpha = mask
    # Merge the color channels with the alpha channel to create a BGRA image
    result = cv2.merge([b, g, r, alpha])
    
    # Save the resulting image as a PNG file (supports transparency)
    cv2.imwrite(output_path, result)
    logging.info(f"Saved extracted image to: {output_path}")

def main():
    """
    Main function:
      - Ensures the output directory exists.
      - Lists all image files in the input directory.
      - Processes each image sequentially.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        logging.info(f"Created output folder: {OUTPUT_FOLDER}")

    files = os.listdir(INPUT_FOLDER)
    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    logging.info(f"Found {len(image_files)} image file(s) in the input folder.")
    
    for file in image_files:
        logging.info(f"Processing image: {file}")
        process_image(file)
    
    logging.info("All images have been processed successfully.")

if __name__ == "__main__":
    main()
