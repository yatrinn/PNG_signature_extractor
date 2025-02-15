# PNG Signature Extractor

PNG Signature Extractor is an AI-based tool that removes the white background from images containing signatures (blue or black ink) using an improved extraction algorithm that combines adaptive and Otsu thresholding with morphological operations.

**Important:** For best results, the input file must be a PNG and should be a large, high-quality screenshot of the signature on a white background. Extraction quality may still vary with image quality.

## Features
- Processes all images in an input folder.
- Uses multiple thresholding methods to robustly extract the signature.
- Applies adaptive logic to select the best mask based on the extracted area.
- Provides detailed logging and error handling.
- Saves the extracted signature as a PNG with a transparent background, appending "_extracted" to the original filename.

## Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:yatrinn/PNG_signature_extractor.git
   cd PNG_signature_extractor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade rembg Pillow opencv-python onnxruntime-silicon "numpy<2"
   ```

4. **Run the tool:**
   Place your high-quality PNG images (screenshots of the signature on a white background) in the  folder and run:
   ```bash
   python signature_extractor.py /Users/Notebook/2025.02.15_PNG_signature_extractor/Input /Users/Notebook/2025.02.15_PNG_signature_extractor/Output
   ```

## Sample Inputs and Outputs

The repository includes a sample input image in the  folder and its corresponding output in the  folder. These samples demonstrate optimal conditions and expected results.

## Troubleshooting

- **Extraction Issues:** If extraction fails or yields poor results, verify that the input image is high-quality, with a clear, high-contrast signature on a white background.
- **Logging:** The tool outputs detailed logs. Review the console output for any warnings or errors.

## Limitations

While the improved algorithm increases robustness, no method is perfect for all images. Variations in lighting, image quality, or signature style may affect results.

## License

This project is licensed under the MIT License.

## Author

Yannik Trinn
