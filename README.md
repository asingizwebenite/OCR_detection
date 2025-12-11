# OCR Text Scanner

A GUI application for scanning printed text from images using PyTesseract and PyQt5.

## Features
- Load images from file
- Use live camera to capture text
- Select specific areas (ROI) to scan
- See detected text overlaid on the image
- Copy/save extracted text

## How to Run

### 1. Install Tesseract OCR
**Windows:** Download from [here](https://github.com/UB-Mannheim/tesseract/wiki)

**Mac/Linux:** Use package manager (brew/apt)

### 2. Install Python packages
```bash
pip install PyQt5 opencv-python pillow pytesseract numpy
