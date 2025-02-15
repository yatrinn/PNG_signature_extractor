#!/usr/bin/env python3
"""
PNG Signature Extractor

This script uses a pre-trained U^2-Net model (via the rembg library) to remove the white background
from an image containing a signature and saves the output as a PNG file.

Author: Yannik Trinn
"""

import argparse
from rembg import remove
from PIL import Image
import io

def process_image(input_path, output_path):
    # Open the input image file in binary mode
    with open(input_path, 'rb') as infile:
        input_data = infile.read()

    # Remove the background using the pre-trained U^2-Net model
    output_data = remove(input_data)

    # Open the output image from the byte data and save as PNG
    output_image = Image.open(io.BytesIO(output_data))
    output_image.save(output_path, 'PNG')
    print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract the signature by removing the white background from an image."
    )
    parser.add_argument("input_image", type=str, help="Path to the input image file.")
    parser.add_argument("output_image", type=str, help="Path to save the output PNG file.")
    args = parser.parse_args()

    process_image(args.input_image, args.output_image)

if __name__ == "__main__":
    main()
