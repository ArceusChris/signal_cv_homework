#!/bin/bash

echo "============================================================"
echo "Running all image processing scripts"
echo "============================================================"
echo ""

echo "[1/5] Running ideal filter..."
python ideal_filter.py
echo ""

echo "[2/5] Running elliptical filter..."
python elliptical_filter.py
echo ""

echo "[3/5] Running Butterworth filter..."
python butterworth_filter.py
echo ""

echo "[4/5] Running Gaussian filter..."
python gaussian_filter.py
echo ""

echo "[5/5] Running edge detection (Sobel & Canny)..."
python edge_detection.py
echo ""

echo "============================================================"
echo "All processing completed!"
echo "============================================================"
echo ""
echo "Output directories:"
echo "  - ideal_filter_output/        (Ideal filters)"
echo "  - elliptical_filter_output/   (Elliptical filters)"
echo "  - butterworth_filter_output/  (Butterworth filters)"
echo "  - gaussian_filter_output/     (Gaussian convolution)"
echo "  - edge_detection_output/      (Sobel & Canny edge detection)"
echo ""
