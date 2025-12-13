#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = 'edge_detection_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_sobel_kernels():
    """Create Sobel operators for edge detection"""
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    return sobel_x, sobel_y

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

def extract_edges_sobel(image):
    """Extract edges using Sobel operator"""
    sobel_x, sobel_y = create_sobel_kernels()
    
    # Apply Sobel operators
    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return magnitude_norm, grad_x, grad_y

def create_hybrid_with_edges(einstein, monroe):
    """Create hybrid image: Einstein edges + Monroe without edges"""
    
    print('\n[Step 1] Extracting edges from Einstein using Sobel...')
    # Extract edges from Einstein
    einstein_edges, grad_x, grad_y = extract_edges_sobel(einstein)
    
    print('[Step 2] Extracting edges from Monroe using Sobel...')
    # Extract edges from Monroe
    monroe_edges, _, _ = extract_edges_sobel(monroe)
    
    print('[Step 3] Creating Monroe without edges (original - edges)...')
    # Monroe without edges = Original - Edges
    # Normalize edges to match image range
    monroe_edges_float = monroe_edges.astype(np.float32)
    monroe_float = monroe.astype(np.float32)
    
    # Subtract edges from Monroe
    monroe_without_edges = monroe_float - monroe_edges_float  # Scale factor 2.0 for better visual effect
    monroe_without_edges = np.clip(monroe_without_edges, 0, 255).astype(np.uint8)
    
    print('[Step 4] Creating hybrid image (Einstein edges + Monroe without edges)...')
    # Create hybrid: Einstein edges + Monroe without edges
    hybrid = einstein_edges.astype(np.float32) + monroe_without_edges.astype(np.float32) * 0.55
    hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)
    
    return einstein_edges, monroe_edges, monroe_without_edges, hybrid, grad_x, grad_y

def visualize_frequency_response(einstein, monroe):
    """Visualize Sobel operator frequency response"""
    sobel_x, sobel_y = create_sobel_kernels()
    
    # Get frequency responses
    freq_resp_x = get_frequency_response(sobel_x, einstein.shape)
    freq_resp_y = get_frequency_response(sobel_y, einstein.shape)
    
    freq_mag_x = np.abs(freq_resp_x)
    freq_mag_y = np.abs(freq_resp_y)
    freq_mag_combined = np.sqrt(freq_mag_x**2 + freq_mag_y**2)
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Sobel Operator - Frequency Response Characteristics', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Sobel kernels and frequency responses
    ax1 = plt.subplot(2, 5, 1)
    im1 = ax1.imshow(sobel_x, cmap='RdBu_r', vmin=-2, vmax=2)
    ax1.set_title('Sobel X Kernel\n(Vertical Edges)')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(2, 5, 2)
    im2 = ax2.imshow(sobel_y, cmap='RdBu_r', vmin=-2, vmax=2)
    ax2.set_title('Sobel Y Kernel\n(Horizontal Edges)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = plt.subplot(2, 5, 3)
    im3 = ax3.imshow(np.log(freq_mag_x + 1), cmap='hot')
    ax3.set_title('Frequency Response X\n(Log Scale)')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = plt.subplot(2, 5, 4)
    im4 = ax4.imshow(np.log(freq_mag_y + 1), cmap='hot')
    ax4.set_title('Frequency Response Y\n(Log Scale)')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    ax5 = plt.subplot(2, 5, 5)
    im5 = ax5.imshow(np.log(freq_mag_combined + 1), cmap='hot')
    ax5.set_title('Combined Response\n(High-pass characteristic)')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # Row 2: Frequency profiles
    center = freq_mag_combined.shape[0] // 2
    
    ax6 = plt.subplot(2, 5, 6)
    ax6.plot(freq_mag_x[center, :], 'b-', label='Sobel X', linewidth=2)
    ax6.plot(freq_mag_y[center, :], 'r-', label='Sobel Y', linewidth=2)
    ax6.plot(freq_mag_combined[center, :], 'g-', label='Combined', linewidth=2, alpha=0.7)
    ax6.set_title('Horizontal Frequency Profile')
    ax6.set_xlabel('Frequency (pixels)')
    ax6.set_ylabel('Magnitude')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(2, 5, 7)
    ax7.plot(freq_mag_x[:, center], 'b-', label='Sobel X', linewidth=2)
    ax7.plot(freq_mag_y[:, center], 'r-', label='Sobel Y', linewidth=2)
    ax7.plot(freq_mag_combined[:, center], 'g-', label='Combined', linewidth=2, alpha=0.7)
    ax7.set_title('Vertical Frequency Profile')
    ax7.set_xlabel('Frequency (pixels)')
    ax7.set_ylabel('Magnitude')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 3D surface plot of combined frequency response
    ax8 = plt.subplot(2, 5, 8, projection='3d')
    y_range = range(center-40, center+40)
    x_range = range(center-40, center+40)
    X, Y = np.meshgrid(x_range, y_range)
    Z = freq_mag_combined[center-40:center+40, center-40:center+40]
    surf = ax8.plot_surface(X, Y, Z, cmap='hot', alpha=0.8)
    ax8.set_title('3D Frequency Response')
    ax8.set_xlabel('u')
    ax8.set_ylabel('v')
    ax8.set_zlabel('Magnitude')
    
    # Polar plot showing directional characteristics
    ax9 = plt.subplot(2, 5, 9, projection='polar')
    angles = np.linspace(0, 2*np.pi, 360)
    radius_profile = []
    for angle in angles:
        # Sample frequency response at this angle
        r = 50  # Fixed radius
        u = int(center + r * np.cos(angle))
        v = int(center + r * np.sin(angle))
        if 0 <= u < freq_mag_combined.shape[0] and 0 <= v < freq_mag_combined.shape[1]:
            radius_profile.append(freq_mag_combined[v, u])
        else:
            radius_profile.append(0)
    ax9.plot(angles, radius_profile, 'g-', linewidth=2)
    ax9.set_title('Directional Response\n(Polar Plot)', pad=20)
    ax9.grid(True)
    
    # Text explanation
    ax10 = plt.subplot(2, 5, 10)
    ax10.axis('off')
    explanation = '''
Sobel Operator Characteristics:

1. Type: First-order derivative

2. Frequency Response:
   • HIGH-PASS filter
   • Emphasizes high frequencies
   • Suppresses low frequencies
   
3. Directional Sensitivity:
   • Sobel X: Vertical edges
   • Sobel Y: Horizontal edges
   
4. Advantages:
   • Simple and fast
   • Good noise suppression
   • Clear directional info
   
5. Mathematical Form:
   Magnitude = √(Gx² + Gy²)
   Direction = atan2(Gy, Gx)
    '''
    ax10.text(0.1, 0.9, explanation, transform=ax10.transAxes, 
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sobel_frequency_response.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print('  Frequency response analysis saved.')

def main():
    print('=' * 60)
    print('Sobel Edge Detection and Hybrid Image Creation')
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
    
    # Create frequency response visualization
    print('\n[Analysis] Creating frequency response visualization...')
    visualize_frequency_response(einstein, monroe)
    
    # Create hybrid image
    einstein_edges, monroe_edges, monroe_without_edges, hybrid, grad_x, grad_y = \
        create_hybrid_with_edges(einstein, monroe)
    
    # Save individual results
    print('\n[Saving] Saving individual results...')
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'einstein_edges.jpg'), einstein_edges)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'monroe_edges.jpg'), monroe_edges)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'monroe_without_edges.jpg'), monroe_without_edges)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'hybrid_edge_based.jpg'), hybrid)
    
    # Create comprehensive visualization
    print('[Visualization] Creating comprehensive comparison...')
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Sobel-based Hybrid Image: Einstein Edges + Monroe (without edges)', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Original images
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(einstein, cmap='gray')
    ax1.set_title('Original Einstein', fontsize=12)
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(monroe, cmap='gray')
    ax2.set_title('Original Monroe', fontsize=12)
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(einstein_edges, cmap='gray')
    ax3.set_title('Einstein Edges\n(Sobel)', fontsize=12)
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 4, 4)
    ax4.imshow(monroe_edges, cmap='gray')
    ax4.set_title('Monroe Edges\n(Sobel)', fontsize=12)
    ax4.axis('off')
    
    # Row 2: Processing steps
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(grad_x, cmap='RdBu_r')
    ax5.set_title('Einstein Gradient X', fontsize=12)
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.imshow(grad_y, cmap='RdBu_r')
    ax6.set_title('Einstein Gradient Y', fontsize=12)
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 4, 7)
    ax7.imshow(monroe_without_edges, cmap='gray')
    ax7.set_title('Monroe Without Edges\n(Original - Edges)', fontsize=12)
    ax7.axis('off')
    
    ax8 = plt.subplot(3, 4, 8)
    ax8.imshow(hybrid, cmap='gray')
    ax8.set_title('Hybrid Image\n(E_edges + M_no_edges)', fontsize=12, 
                 fontweight='bold', color='darkred')
    ax8.axis('off')
    
    # Row 3: FFT Analysis
    einstein_fft = np.fft.fftshift(np.fft.fft2(einstein))
    ax9 = plt.subplot(3, 4, 9)
    ax9.imshow(np.log(np.abs(einstein_fft) + 1), cmap='gray')
    ax9.set_title('Einstein FFT', fontsize=11)
    ax9.axis('off')
    
    edges_fft = np.fft.fftshift(np.fft.fft2(einstein_edges))
    ax10 = plt.subplot(3, 4, 10)
    ax10.imshow(np.log(np.abs(edges_fft) + 1), cmap='hot')
    ax10.set_title('Edges FFT\n(High-pass)', fontsize=11)
    ax10.axis('off')
    
    monroe_no_edges_fft = np.fft.fftshift(np.fft.fft2(monroe_without_edges))
    ax11 = plt.subplot(3, 4, 11)
    ax11.imshow(np.log(np.abs(monroe_no_edges_fft) + 1), cmap='gray')
    ax11.set_title('Monroe (no edges) FFT\n(Low-pass)', fontsize=11)
    ax11.axis('off')
    
    hybrid_fft = np.fft.fftshift(np.fft.fft2(hybrid))
    ax12 = plt.subplot(3, 4, 12)
    ax12.imshow(np.log(np.abs(hybrid_fft) + 1), cmap='gray')
    ax12.set_title('Hybrid FFT', fontsize=11)
    ax12.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comprehensive_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create comparison at different scales
    print('[Visualization] Creating multi-scale comparison...')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hybrid Image at Different Viewing Distances', fontsize=16, fontweight='bold')
    
    scales = [(1.0, 'Original Size'), (0.5, 'Medium Distance (50%)'), 
              (0.25, 'Far Distance (25%)')]
    
    for idx, (scale, title) in enumerate(scales):
        if scale < 1.0:
            h, w = hybrid.shape
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(hybrid, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Resize back for display
            scaled_display = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            scaled_display = hybrid
        
        axes[0, idx].imshow(scaled_display, cmap='gray')
        axes[0, idx].set_title(title, fontsize=12)
        axes[0, idx].axis('off')
    
    # Show components
    axes[1, 0].imshow(einstein_edges, cmap='gray')
    axes[1, 0].set_title('Einstein Edges\n(High frequency)', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(monroe_without_edges, cmap='gray')
    axes[1, 1].set_title('Monroe (no edges)\n(Low frequency)', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(hybrid, cmap='gray')
    axes[1, 2].set_title('Combined Result', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multi_scale_view.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'\n{"=" * 60}')
    print(f'Complete! Results in: {OUTPUT_DIR}/')
    print('=' * 60)
    print('\nGenerated files:')
    print('  1. sobel_frequency_response.png - Frequency response analysis')
    print('  2. comprehensive_analysis.png - Complete processing pipeline')
    print('  3. multi_scale_view.png - Viewing distance comparison')
    print('  4. einstein_edges.jpg - Einstein edge map')
    print('  5. monroe_without_edges.jpg - Monroe with edges removed')
    print('  6. hybrid_edge_based.jpg - Final hybrid image')
    print('\nMethod:')
    print('  • Extract edges from Einstein using Sobel operator')
    print('  • Remove edges from Monroe (Original - Edges)')
    print('  • Combine: Einstein_edges + Monroe_without_edges')
    print('  • Result: Near view shows Einstein edges, far view shows Monroe shape')

if __name__ == '__main__':
    main()
