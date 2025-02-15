# PNG Signature Extractor

PNG Signature Extractor is an AI-based tool that removes the white background from images containing signatures (blue or black ink) using advanced image processing and thresholding techniques.

**Important:** The input file must be a PNG and should be a large, high-quality screenshot of the signature on a white background for optimal extraction.

## Features
- Processes all images in an input folder.
- Extracts the signature and saves it as a PNG with a transparent background.
- Appends "_extracted" to the original filename.
- Uses classical thresholding with morphological operations for reliable signature extraction.

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

4. **Run the script:**
   ```bash
   python signature_extractor.py /Users/Notebook/2025.02.15_PNG_signature_extractor/Input /Users/Notebook/2025.02.15_PNG_signature_extractor/Output
   ```

## License

This project is licensed under the MIT License.

## Author

Yannik Trinn
