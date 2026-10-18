#!/usr/bin/env python3
"""
Signature Extractor Script
This script removes the white background from images (e.g. signatures)
using a pre-trained neural network (via rembg) with alpha matting enabled
to preserve thin and delicate details.
"""

import os
import io
import logging
import concurrent.futures
from rembg import remove
from PIL import Image

# Define the input and output directories
INPUT_FOLDER = "Input"
OUTPUT_FOLDER = "Output"

# Configure logging in high-tech USA style English
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_image(filename):
    """
    Process a single image:
    1. Read the image from the input folder.
    2. Remove the background using rembg with alpha matting to preserve thin signature details.
    3. Save the resulting image as a PNG in the output folder.
    """
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0] + "_extracted.png")
    logging.info(f"Processing image: {filename}")
    
    # Open the input image file in binary mode
    try:
        with open(input_path, 'rb') as infile:
            input_data = infile.read()
    except Exception as e:
        logging.error(f"Failed to open {filename}: {e}")
        return

    # Remove background using rembg with alpha matting enabled.
    # These parameters are tuned to preserve thin signature lines.
    try:
        result_data = remove(
            input_data,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_structure_size=10
        )
    except Exception as e:
        logging.error(f"Background removal failed for {filename}: {e}")
        return

    # Convert the resulting bytes into an RGBA image and save as PNG
    try:
        result_image = Image.open(io.BytesIO(result_data)).convert("RGBA")
        result_image.save(output_path)
        logging.info(f"Saved extracted image to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save output for {filename}: {e}")

def main():
    """
    Main function:
    - Ensures the output directory exists.
    - Finds all image files in the input directory.
    - Processes all images concurrently.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # List image files (PNG, JPG, JPEG) in the input folder
    try:
        files = os.listdir(INPUT_FOLDER)
    except Exception as e:
        logging.error(f"Unable to list files in {INPUT_FOLDER}: {e}")
        return

    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    logging.info(f"Found {len(image_files)} image file(s) in the input folder.")

    # Process images concurrently using a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_image, file): file for file in image_files}
        for future in concurrent.futures.as_completed(futures):
            file = futures[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f"{file} generated an exception: {exc}")

    logging.info("All images have been processed successfully.")

if __name__ == "__main__":
    main()
