# Image Preprocessing for Microscopy Images

Automated preprocessing pipeline to remove scale bars and text from microscopy TIF images.

## Features

- Processes all TIF images in nested folder structures
- Automatically detects and removes scale bars and text from bottom-right corner
- Uses intelligent inpainting to fill removed areas naturally
- Preserves original folder structure in output
- Handles complex TIF formats common in microscopy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python process_images.py
```

The script will:
1. Scan `./Data/` folder for all TIF images
2. Process each image to remove scale bars and text
3. Save processed images to `./data_processed/` with same folder structure
4. Prefix each processed file with `processed_`

## How It Works

1. **Detection**: Identifies bright pixels (>200 intensity) in bottom-right corner (15% width, 10% height)
2. **Masking**: Creates a binary mask of scale bar and text areas
3. **Inpainting**: Uses OpenCV's TELEA algorithm to intelligently fill masked regions based on surrounding pixels
4. **Output**: Saves processed images maintaining original format and structure

## Project Structure

```
Preprocessing_KIT/
├── Data/                    # Input images (your microscopy data)
│   ├── 24h/
│   │   ├── Cont/
│   │   ├── ePL100/
│   │   └── ...
│   └── 48h/
│       └── ...
├── data_processed/          # Output folder (created automatically)
│   └── [mirrors Data structure]
├── process_images.py        # Main processing script
└── requirements.txt         # Dependencies
```

## Requirements

- Python 3.7+
- OpenCV (opencv-python)
- NumPy

## Notes

- Input format: `.tif` or `.tiff`
- Output format: Same as input (TIFF)
- Processing is non-destructive (original files unchanged)
- Adjust region size in code if scale bars appear in different locations

