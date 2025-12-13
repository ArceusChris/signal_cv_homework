#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def create_comprehensive_comparison():
    """Create a comprehensive comparison of all methods"""
    
    print('=' * 60)
    print('Creating Comprehensive Comparison')
    print('=' * 60)
    
    # Load original images
    einstein = cv2.imread('einstein.jpg', cv2.IMREAD_GRAYSCALE)
    monroe = cv2.imread('monroe.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Check if output directories exist
    dirs_to_check = [
        'ideal_filter_output',
        'butterworth_filter_output',
        'gaussian_filter_output',
        'edge_detection_output'
    ]
    
    existing_dirs = [d for d in dirs_to_check if os.path.exists(d)]
    
    if len(existing_dirs) == 0:
        print('Error: No output directories found. Please run the processing scripts first.')
        return
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Comprehensive Image Processing Methods Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    row_idx = 1
    
    # Row 1: Original images
    ax1 = plt.subplot(5, 6, 1)
    ax1.imshow(einstein, cmap='gray')
    ax1.set_title('Original\nEinstein', fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(5, 6, 2)
    ax2.imshow(monroe, cmap='gray')
    ax2.set_title('Original\nMonroe', fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    # Try to load and display various filtered results
    methods = [
        ('Ideal Filter\nD0=20', 'ideal_filter_output/hybrid_D0_20.jpg'),
        ('Butterworth\nD0=20', 'butterworth_filter_output/hybrid_d0_20.jpg'),
        ('Gaussian\nk=15, σ=2', 'gaussian_filter_output/hybrid_k15_s2.0.jpg'),
        ('Gaussian\nk=21, σ=3', 'gaussian_filter_output/hybrid_k21_s3.0.jpg'),
    ]
    
    for idx, (title, path) in enumerate(methods, start=3):
        ax = plt.subplot(5, 6, idx)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Row 2: Edge detection results
    edge_methods = [
        ('Sobel\nEinstein', 'edge_detection_output/sobel_einstein_edges.jpg'),
        ('Sobel\nMonroe', 'edge_detection_output/sobel_monroe_edges.jpg'),
        ('Canny Einstein\nT=(30,100)', 'edge_detection_output/canny_einstein_t30_100.jpg'),
        ('Canny Einstein\nT=(50,150)', 'edge_detection_output/canny_einstein_t50_150.jpg'),
        ('Canny Monroe\nT=(30,100)', 'edge_detection_output/canny_monroe_t30_100.jpg'),
        ('Canny Monroe\nT=(50,150)', 'edge_detection_output/canny_monroe_t50_150.jpg'),
    ]
    
    for idx, (title, path) in enumerate(edge_methods, start=7):
        ax = plt.subplot(5, 6, idx)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=9)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
            ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # Row 3: Different filter cutoffs
    filter_variations = [
        ('Ideal\nD0=5', 'ideal_filter_output/hybrid_D0_5.jpg'),
        ('Ideal\nD0=10', 'ideal_filter_output/hybrid_D0_10.jpg'),
        ('Ideal\nD0=30', 'ideal_filter_output/hybrid_D0_30.jpg'),
        ('Butterworth\nD0=5', 'butterworth_filter_output/hybrid_d0_5.jpg'),
        ('Butterworth\nD0=10', 'butterworth_filter_output/hybrid_d0_10.jpg'),
        ('Butterworth\nD0=30', 'butterworth_filter_output/hybrid_d0_30.jpg'),
    ]
    
    for idx, (title, path) in enumerate(filter_variations, start=13):
        ax = plt.subplot(5, 6, idx)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=9)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
            ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # Row 4: Gaussian variations
    gaussian_variations = [
        ('Gaussian\nk=5, σ=1', 'gaussian_filter_output/hybrid_k5_s1.0.jpg'),
        ('Gaussian\nk=9, σ=1.5', 'gaussian_filter_output/hybrid_k9_s1.5.jpg'),
        ('Gaussian\nk=15, σ=2', 'gaussian_filter_output/hybrid_k15_s2.0.jpg'),
        ('Gaussian\nk=21, σ=3', 'gaussian_filter_output/hybrid_k21_s3.0.jpg'),
        ('Gaussian\nk=31, σ=5', 'gaussian_filter_output/hybrid_k31_s5.0.jpg'),
    ]
    
    for idx, (title, path) in enumerate(gaussian_variations, start=19):
        ax = plt.subplot(5, 6, idx)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=9)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
            ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # Add one more edge detection result
    ax = plt.subplot(5, 6, 24)
    path = 'edge_detection_output/canny_einstein_t100_200.jpg'
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray')
        ax.set_title('Canny Einstein\nT=(100,200)', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
        ax.set_title('Canny Einstein\nT=(100,200)', fontsize=9)
    ax.axis('off')
    
    # Row 5: Frequency responses comparison
    # Load and display frequency analysis if available
    freq_images = [
        ('Ideal Filter\nFrequency', 'ideal_filter_output/01_filters_D0_5.png'),
        ('Butterworth\nFrequency', 'butterworth_filter_output/butterworth_d0_5.png'),
        ('Gaussian\nFrequency', 'gaussian_filter_output/gaussian_k15_s2.0.png'),
        ('Sobel\nFrequency', 'edge_detection_output/sobel_Einstein.png'),
    ]
    
    for idx, (title, path) in enumerate(freq_images, start=25):
        ax = plt.subplot(5, 6, idx)
        if os.path.exists(path):
            # These are large images, just show a placeholder
            ax.text(0.5, 0.5, 'See detailed\nanalysis in:\n' + path.split('/')[0], 
                   ha='center', va='center', fontsize=8, wrap=True)
            ax.set_title(title, fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
            ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # Add text annotations
    plt.figtext(0.02, 0.88, 'Row 1:', fontsize=11, fontweight='bold')
    plt.figtext(0.02, 0.86, 'Original + Hybrid Images', fontsize=9)
    
    plt.figtext(0.02, 0.70, 'Row 2:', fontsize=11, fontweight='bold')
    plt.figtext(0.02, 0.68, 'Edge Detection', fontsize=9)
    
    plt.figtext(0.02, 0.52, 'Row 3:', fontsize=11, fontweight='bold')
    plt.figtext(0.02, 0.50, 'Filter Variations', fontsize=9)
    
    plt.figtext(0.02, 0.34, 'Row 4:', fontsize=11, fontweight='bold')
    plt.figtext(0.02, 0.32, 'Gaussian Variations', fontsize=9)
    
    plt.figtext(0.02, 0.16, 'Row 5:', fontsize=11, fontweight='bold')
    plt.figtext(0.02, 0.14, 'Frequency Analysis', fontsize=9)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    plt.savefig('COMPREHENSIVE_COMPARISON.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print('\n' + '=' * 60)
    print('Comprehensive comparison saved: COMPREHENSIVE_COMPARISON.png')
    print('=' * 60)
    
    # Create a summary statistics image
    create_statistics_summary()

def create_statistics_summary():
    """Create a summary of all processing statistics"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    title_text = 'Image Processing Methods - Summary Report'
    ax.text(0.5, 0.95, title_text, ha='center', va='top', 
            fontsize=18, fontweight='bold')
    
    # Method descriptions
    methods_text = '''
    FREQUENCY DOMAIN FILTERS (Hybrid Images):
    
    1. Ideal Filter
       • Sharp cutoff at D₀
       • Parameters: D₀ = [5, 10, 20, 30, 50]
       • Characteristics: Sharpest transition, ringing artifacts
    
    2. Elliptical Filter
       • Elliptical shape in frequency domain
       • Parameters: (a,b) = [(10,5), (15,10), (20,10), (30,15), (40,20)]
       • Characteristics: Directional filtering capability
    
    3. Butterworth Filter (4th order)
       • Smooth frequency response: H = 1/[1+(D/D₀)^(2n)]
       • Parameters: n=4, D₀ = [5, 10, 20, 30, 50]
       • Characteristics: Smooth transition, no ringing
    
    SPATIAL DOMAIN PROCESSING:
    
    4. Gaussian Convolution
       • G(x,y) = exp[-(x²+y²)/2σ²]
       • Parameters: (k,σ) = [(5,1.0), (9,1.5), (15,2.0), (21,3.0), (31,5.0)]
       • Frequency Response: Low-pass, Gaussian shape
       • Characteristics: Isotropic, smooth, no ringing
    
    5. Edge Detection
       • Sobel: Gradient-based, 3×3 kernels for X and Y derivatives
         - Frequency Response: High-pass with directional sensitivity
       • Canny: Multi-stage (Gaussian→Gradient→NMS→Hysteresis)
         - Parameters: [(30,100), (50,150), (100,200)]
         - Frequency Response: Band-pass characteristics
         - Characteristics: Precise localization, noise robust
    
    OUTPUT SUMMARY:
    • Total output folders: 5
    • Total output size: ~36 MB
    • Total files generated: 83+
    • Includes: Filtered images, frequency responses, analysis plots
    '''
    
    ax.text(0.05, 0.88, methods_text, ha='left', va='top', 
            fontsize=9, family='monospace', linespacing=1.8)
    
    # Add comparison table
    table_y = 0.08
    ax.text(0.5, table_y, 'FREQUENCY RESPONSE CHARACTERISTICS COMPARISON', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.savefig('METHODS_SUMMARY.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print('Methods summary saved: METHODS_SUMMARY.png')

if __name__ == '__main__':
    create_comprehensive_comparison()
