#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = 'gaussian_filter_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel"""
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize
    kernel = kernel / np.sum(kernel)
    return kernel

def get_frequency_response(kernel, image_shape):
    """Get frequency response of a spatial domain kernel"""
    # Pad kernel to image size
    padded_kernel = np.zeros(image_shape)
    kh, kw = kernel.shape
    padded_kernel[:kh, :kw] = kernel
    
    # Shift kernel to center for proper frequency response
    padded_kernel = np.roll(padded_kernel, -(kh//2), axis=0)
    padded_kernel = np.roll(padded_kernel, -(kw//2), axis=1)
    
    # Get frequency response
    freq_response = np.fft.fft2(padded_kernel)
    freq_response_shifted = np.fft.fftshift(freq_response)
    
    return freq_response_shifted

def process_gaussian(einstein, monroe, kernel_size, sigma):
    """Process images with Gaussian kernel"""
    # Create Gaussian kernel
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    
    # Apply Gaussian filter to get LOW FREQUENCY components
    einstein_low = cv2.filter2D(einstein, -1, gaussian_kernel)
    monroe_low = cv2.filter2D(monroe, -1, gaussian_kernel)
    
    # Get HIGH FREQUENCY components (original - low frequency)
    einstein_high = einstein.astype(np.float32) - einstein_low.astype(np.float32)
    monroe_high = monroe.astype(np.float32) - monroe_low.astype(np.float32)
    
    # Get frequency response
    freq_response = get_frequency_response(gaussian_kernel, einstein.shape)
    freq_magnitude = np.abs(freq_response)
    
    # Create hybrid image: Einstein HIGH frequency + Monroe LOW frequency
    hybrid = einstein_high + monroe_low.astype(np.float32)
    hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Kernel and frequency response
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(gaussian_kernel, cmap='hot')
    ax1.set_title(f'Gaussian Kernel\nSize={kernel_size}, σ={sigma}')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(np.log(freq_magnitude + 1), cmap='hot')
    ax2.set_title('Frequency Response\n(Log Scale)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3D surface plot of kernel
    ax3 = plt.subplot(3, 4, 3, projection='3d')
    x = np.arange(kernel_size)
    y = np.arange(kernel_size)
    X, Y = np.meshgrid(x, y)
    ax3.plot_surface(X, Y, gaussian_kernel, cmap='hot')
    ax3.set_title('Kernel 3D View')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Value')
    
    # Cross-section of frequency response
    ax4 = plt.subplot(3, 4, 4)
    center = freq_magnitude.shape[0] // 2
    ax4.plot(freq_magnitude[center, :], 'b-', linewidth=2)
    ax4.set_title('Frequency Response Profile')
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('Magnitude')
    ax4.grid(True)
    
    # Row 2: Original and filtered images
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(einstein, cmap='gray')
    ax5.set_title('Original Einstein')
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.imshow(np.clip(einstein_high + 128, 0, 255).astype(np.uint8), cmap='gray')
    ax6.set_title('Einstein HIGH freq\n(Original - Gaussian)')
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 4, 7)
    ax7.imshow(monroe, cmap='gray')
    ax7.set_title('Original Monroe')
    ax7.axis('off')
    
    ax8 = plt.subplot(3, 4, 8)
    ax8.imshow(monroe_low, cmap='gray')
    ax8.set_title('Monroe LOW freq\n(Gaussian filtered)')
    ax8.axis('off')
    
    # Row 3: Frequency domain analysis
    ax9 = plt.subplot(3, 4, 9)
    einstein_fft = np.fft.fftshift(np.fft.fft2(einstein))
    ax9.imshow(np.log(np.abs(einstein_fft) + 1), cmap='gray')
    ax9.set_title('Einstein FFT')
    ax9.axis('off')
    
    ax10 = plt.subplot(3, 4, 10)
    einstein_high_fft = np.fft.fftshift(np.fft.fft2(einstein_high))
    ax10.imshow(np.log(np.abs(einstein_high_fft) + 1), cmap='hot')
    ax10.set_title('Einstein High FFT\n(High-pass)')
    ax10.axis('off')
    
    ax11 = plt.subplot(3, 4, 11)
    ax11.imshow(hybrid, cmap='gray')
    ax11.set_title('Hybrid Image\n(E_high + M_low)')
    ax11.axis('off')
    
    ax12 = plt.subplot(3, 4, 12)
    hybrid_fft = np.fft.fftshift(np.fft.fft2(hybrid))
    ax12.imshow(np.log(np.abs(hybrid_fft) + 1), cmap='gray')
    ax12.set_title('Hybrid FFT')
    ax12.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gaussian_k{kernel_size}_s{sigma}.png'), 
                dpi=100, bbox_inches='tight')
    plt.close()
    
    # Save individual results
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'einstein_high_k{kernel_size}_s{sigma}.jpg'),
                np.clip(einstein_high + 128, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'monroe_low_k{kernel_size}_s{sigma}.jpg'),
                monroe_low)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'hybrid_k{kernel_size}_s{sigma}.jpg'),
                hybrid)
    
    print(f'  Done: kernel_size={kernel_size}, sigma={sigma}')
    return hybrid, freq_magnitude

def main():
    print('=' * 60)
    print('Gaussian Convolution Kernel Processing')
    print('=' * 60)
    
    # Load images
    einstein = cv2.imread('einstein.jpg', cv2.IMREAD_GRAYSCALE)
    monroe = cv2.imread('monroe.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Resize to same size
    min_height = min(einstein.shape[0], monroe.shape[0])
    min_width = min(einstein.shape[1], monroe.shape[1])
    einstein = cv2.resize(einstein, (min_width, min_height))
    monroe = cv2.resize(monroe, (min_width, min_height))
    
    print(f'Image size: {einstein.shape}')
    
    # Test different kernel sizes and sigma values
    params = [
        (5, 1.0),
        (9, 1.5),
        (15, 2.0),
        (21, 3.0),
        (31, 5.0)
    ]
    
    print(f'\nProcessing {len(params)} parameter sets:')
    print('(kernel_size, sigma)')
    
    results = []
    for kernel_size, sigma in params:
        hybrid, freq_mag = process_gaussian(einstein, monroe, kernel_size, sigma)
        results.append((kernel_size, sigma, hybrid))
    
    # Create comparison plot
    n_results = len(results)
    fig, axes = plt.subplots(2, n_results, figsize=(4*n_results, 8))
    
    for idx, (k_size, sig, hybrid) in enumerate(results):
        # Show hybrid images
        axes[0, idx].imshow(hybrid, cmap='gray')
        axes[0, idx].set_title(f'k={k_size}, σ={sig}')
        axes[0, idx].axis('off')
        
        # Show frequency response
        kernel = create_gaussian_kernel(k_size, sig)
        freq_resp = get_frequency_response(kernel, einstein.shape)
        axes[1, idx].imshow(np.log(np.abs(freq_resp) + 1), cmap='hot')
        axes[1, idx].set_title(f'Freq Response')
        axes[1, idx].axis('off')
    
    axes[0, 0].set_ylabel('Hybrid Images', fontsize=12, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel('Frequency Response', fontsize=12, rotation=90, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_all.png'), dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f'\n{"=" * 60}')
    print(f'Complete! Results in: {OUTPUT_DIR}/')
    print('=' * 60)

if __name__ == '__main__':
    main()
