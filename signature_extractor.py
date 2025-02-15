import cv2
import numpy as np
import os
import argparse

def process_image(input_path, output_path):
    # Load the input image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image {input_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding (invert to capture dark signature on white)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up the mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Convert original image to BGRA and set the alpha channel to the mask
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_bgra[:, :, 3] = mask

    # Save the result as PNG
    cv2.imwrite(output_path, image_bgra)
    print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract signatures from all images in a folder using thresholding."
    )
    parser.add_argument("input_folder", help="Path to the input folder containing images.")
    parser.add_argument("output_folder", help="Path to the output folder for processed images.")
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Process each image file in the input folder
    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            input_path = os.path.join(args.input_folder, filename)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_extracted.png"
            output_path = os.path.join(args.output_folder, output_filename)
            process_image(input_path, output_path)

if __name__ == "__main__":
    main()
