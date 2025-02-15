#!/usr/bin/env python3
"""
Super‑Quality Signature Extractor – RGB‑Based Alpha Refinement
--------------------------------------------------------------
This script processes images (PNG, JPG, JPEG) from the Input folder,
removes the background using rembg (U2‑Net on CPU), then reconstructs
a binary alpha mask based on the RGB channels so that pixels near white
become fully transparent. Finally, it enhances the signature's contrast
and sharpness before saving the final PNG output.
Folders:
  Input:  /Users/Notebook/2025.02.15_PNG_signature_extractor/Input
  Output: /Users/Notebook/2025.02.15_PNG_signature_extractor/Output
Usage:
  python signature_extractor.py
"""

import os
import io
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from rembg import remove, new_session
from PIL import Image, ImageEnhance
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set folder paths
INPUT_FOLDER = "/Users/Notebook/2025.02.15_PNG_signature_extractor/Input"
OUTPUT_FOLDER = "/Users/Notebook/2025.02.15_PNG_signature_extractor/Output"

# Global session variable
session_local = None
def get_session():
    global session_local
    if session_local is None:
        # Use U2-Net on CPU.
        session_local = new_session(model_name="u2net", providers=["CPUExecutionProvider"])
    return session_local

def refine_alpha_by_rgb(image):
    """
    Given an RGBA PIL image, this function creates a new alpha mask where
    pixels that are nearly white (i.e. likely background) are set to 0 (transparent)
    and all others are set to 255 (opaque).
    """
    # Convert the image to a NumPy array.
    data = np.array(image)  # shape: (H, W, 4)
    # Split into channels.
    r, g, b, a = cv2.split(data)
    # Create a mask: if all channels are above 240 (nearly white), mark as background.
    mask = ((r > 240) & (g > 240) & (b > 240)).astype(np.uint8) * 255
    # Replace the alpha channel: background becomes 0, others 255.
    data[:, :, 3] = np.where(mask == 255, 0, 255)
    return Image.fromarray(data, "RGBA")

def process_image(file_name):
    """
    Process a single image:
      1. Remove the background using rembg (with alpha_matting disabled).
      2. Convert the rembg output to an RGBA image.
      3. Reconstruct a binary alpha mask using the RGB channels.
      4. Enhance contrast and sharpness.
      5. Save the final PNG.
    """
    input_path = os.path.join(INPUT_FOLDER, file_name)
    name, _ = os.path.splitext(file_name)
    output_path = os.path.join(OUTPUT_FOLDER, f"{name}_extracted.png")
    logging.info(f"Processing image: {file_name}")
    try:
        with open(input_path, "rb") as f:
            input_data = f.read()
        session = get_session()
        # Remove background with rembg.
        output_data = remove(input_data, session=session, alpha_matting=False)
        # Open the resulting image.
        image = Image.open(io.BytesIO(output_data)).convert("RGBA")
        # Reconstruct alpha mask based solely on the RGB channels.
        refined_image = refine_alpha_by_rgb(image)
        # Enhance contrast and sharpness.
        refined_image = ImageEnhance.Contrast(refined_image).enhance(1.8)
        refined_image = ImageEnhance.Sharpness(refined_image).enhance(6.0)
        refined_image.save(output_path, "PNG")
        logging.info(f"Saved extracted image to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to process {file_name}: {e}")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    files = os.listdir(INPUT_FOLDER)
    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    logging.info(f"Found {len(image_files)} image file(s) in the input folder.")
    # Process images concurrently using ProcessPoolExecutor.
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_image, file): file for file in image_files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f"{file} generated an exception: {exc}")
    logging.info("All images have been processed successfully.")

if __name__ == "__main__":
    main()
