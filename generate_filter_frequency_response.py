"""
Generate frequency response images for filters/operators themselves
显示滤波器本身的频率响应特性
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def ideal_filter_response(rows, cols, D0, filter_type='lowpass'):
    """理想滤波器的频率响应"""
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if filter_type == 'lowpass':
                mask[i, j] = 1.0 if D <= D0 else 0.0
            else:  # highpass
                mask[i, j] = 0.0 if D <= D0 else 1.0
    return mask

def butterworth_filter_response(rows, cols, D0, n=4, filter_type='lowpass'):
    """巴特沃斯滤波器的频率响应"""
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if D == 0:
                D = 0.01
            if filter_type == 'lowpass':
                mask[i, j] = 1 / (1 + (D / D0) ** (2 * n))
            else:  # highpass
                mask[i, j] = 1 / (1 + (D0 / D) ** (2 * n))
    return mask

def elliptical_filter_response(rows, cols, a, b, filter_type='lowpass'):
    """椭圆滤波器的频率响应"""
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            u, v = i - crow, j - ccol
            ellipse_eq = (u**2 / a**2) + (v**2 / b**2)
            if filter_type == 'lowpass':
                mask[i, j] = 1.0 if ellipse_eq <= 1 else 0.0
            else:  # highpass
                mask[i, j] = 0.0 if ellipse_eq <= 1 else 1.0
    return mask

def gaussian_kernel_frequency_response(kernel_size, sigma):
    """高斯核的频率响应"""
    # Create Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T
    
    # Pad to larger size for better visualization
    size = 256
    padded = np.zeros((size, size))
    k_half = kernel_size // 2
    center = size // 2
    padded[center-k_half:center-k_half+kernel_size, 
           center-k_half:center-k_half+kernel_size] = kernel
    
    # Compute FFT
    f = np.fft.fft2(padded)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Normalize
    magnitude = magnitude / np.max(magnitude)
    return magnitude

def sobel_frequency_response():
    """Sobel算子的频率响应"""
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)
    
    # Pad to larger size
    size = 256
    padded_x = np.zeros((size, size))
    padded_y = np.zeros((size, size))
    center = size // 2
    padded_x[center-1:center+2, center-1:center+2] = sobel_x
    padded_y[center-1:center+2, center-1:center+2] = sobel_y
    
    # Compute FFT
    fx = np.fft.fft2(padded_x)
    fy = np.fft.fft2(padded_y)
    fshift_x = np.fft.fftshift(fx)
    fshift_y = np.fft.fftshift(fy)
    
    mag_x = np.abs(fshift_x)
    mag_y = np.abs(fshift_y)
    mag_combined = np.sqrt(mag_x**2 + mag_y**2)
    
    # Normalize
    mag_x = mag_x / np.max(mag_x)
    mag_y = mag_y / np.max(mag_y)
    mag_combined = mag_combined / np.max(mag_combined)
    
    return mag_x, mag_y, mag_combined

def plot_filter_response(response, title, save_path):
    """Plot and save filter frequency response"""
    plt.figure(figsize=(10, 8))
    plt.imshow(response, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    plt.colorbar(label='Magnitude Response', shrink=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Normalized Frequency u', fontsize=12)
    plt.ylabel('Normalized Frequency v', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_all_filter_responses():
    """Generate frequency response for all filters"""
    output_dir = Path('filter_frequency_responses')
    output_dir.mkdir(exist_ok=True)
    
    size = 256
    
    print("\n=== Generating Filter Frequency Responses ===\n")
    
    # 1. Ideal Filter
    print("1. Ideal Filter...")
    ideal_lp = ideal_filter_response(size, size, 30, 'lowpass')
    ideal_hp = ideal_filter_response(size, size, 30, 'highpass')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = axes[0].imshow(ideal_lp, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[0].set_title('Ideal Low-Pass Filter (D0=30)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Normalized Frequency u')
    axes[0].set_ylabel('Normalized Frequency v')
    plt.colorbar(im1, ax=axes[0], label='H(u,v)')
    
    im2 = axes[1].imshow(ideal_hp, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[1].set_title('Ideal High-Pass Filter (D0=30)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Normalized Frequency u')
    axes[1].set_ylabel('Normalized Frequency v')
    plt.colorbar(im2, ax=axes[1], label='H(u,v)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ideal_filter_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: ideal_filter_response.png")
    
    # 2. Butterworth Filter
    print("2. Butterworth Filter...")
    butter_lp = butterworth_filter_response(size, size, 30, n=4, filter_type='lowpass')
    butter_hp = butterworth_filter_response(size, size, 30, n=4, filter_type='highpass')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = axes[0].imshow(butter_lp, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[0].set_title('Butterworth Low-Pass Filter (D0=30, n=4)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Normalized Frequency u')
    axes[0].set_ylabel('Normalized Frequency v')
    plt.colorbar(im1, ax=axes[0], label='H(u,v)')
    
    im2 = axes[1].imshow(butter_hp, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[1].set_title('Butterworth High-Pass Filter (D0=30, n=4)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Normalized Frequency u')
    axes[1].set_ylabel('Normalized Frequency v')
    plt.colorbar(im2, ax=axes[1], label='H(u,v)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'butterworth_filter_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: butterworth_filter_response.png")
    
    # 3. Elliptical Filter
    print("3. Elliptical Filter...")
    ellip_lp = elliptical_filter_response(size, size, 40, 20, 'lowpass')
    ellip_hp = elliptical_filter_response(size, size, 40, 20, 'highpass')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = axes[0].imshow(ellip_lp, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[0].set_title('Elliptical Low-Pass Filter (a=40, b=20)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Normalized Frequency u')
    axes[0].set_ylabel('Normalized Frequency v')
    plt.colorbar(im1, ax=axes[0], label='H(u,v)')
    
    im2 = axes[1].imshow(ellip_hp, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[1].set_title('Elliptical High-Pass Filter (a=40, b=20)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Normalized Frequency u')
    axes[1].set_ylabel('Normalized Frequency v')
    plt.colorbar(im2, ax=axes[1], label='H(u,v)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'elliptical_filter_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: elliptical_filter_response.png")
    
    # 4. Gaussian Kernel
    print("4. Gaussian Kernel...")
    gauss_resp = gaussian_kernel_frequency_response(15, 2.0)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(gauss_resp, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    plt.colorbar(label='Magnitude Response', shrink=0.8)
    plt.title('Gaussian Kernel Frequency Response (k=15, σ=2.0)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Normalized Frequency u', fontsize=12)
    plt.ylabel('Normalized Frequency v', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'gaussian_kernel_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: gaussian_kernel_response.png")
    
    # 5. Sobel Operator
    print("5. Sobel Operator...")
    sobel_x, sobel_y, sobel_combined = sobel_frequency_response()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = axes[0].imshow(sobel_x, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[0].set_title('Sobel X (Horizontal Edge)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Normalized Frequency u')
    axes[0].set_ylabel('Normalized Frequency v')
    plt.colorbar(im1, ax=axes[0], label='|H(u,v)|')
    
    im2 = axes[1].imshow(sobel_y, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[1].set_title('Sobel Y (Vertical Edge)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Normalized Frequency u')
    axes[1].set_ylabel('Normalized Frequency v')
    plt.colorbar(im2, ax=axes[1], label='|H(u,v)|')
    
    im3 = axes[2].imshow(sobel_combined, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[2].set_title('Sobel Combined Magnitude', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('Normalized Frequency u')
    axes[2].set_ylabel('Normalized Frequency v')
    plt.colorbar(im3, ax=axes[2], label='|H(u,v)|')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sobel_operator_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: sobel_operator_response.png")
    
    # 6. Comparison of all filters
    print("6. Generating comparison figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    filters = [
        (ideal_lp, 'Ideal LP (D0=30)'),
        (butter_lp, 'Butterworth LP (D0=30, n=4)'),
        (ellip_lp, 'Elliptical LP (a=40, b=20)'),
        (gauss_resp, 'Gaussian (k=15, σ=2.0)'),
        (ideal_hp, 'Ideal HP (D0=30)'),
        (butter_hp, 'Butterworth HP (D0=30, n=4)')
    ]
    
    for idx, (filt, title) in enumerate(filters):
        row = idx // 3
        col = idx % 3
        im = axes[row, col].imshow(filt, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
        axes[row, col].set_title(title, fontweight='bold', fontsize=11)
        axes[row, col].set_xlabel('Frequency u', fontsize=9)
        axes[row, col].set_ylabel('Frequency v', fontsize=9)
        plt.colorbar(im, ax=axes[row, col], label='H(u,v)', shrink=0.8)
    
    plt.suptitle('Filter Frequency Response Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_filters_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: all_filters_comparison.png")
    
    print(f"\n{'='*60}")
    print(f"All filter frequency responses saved in: {output_dir}/")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    print("="*60)
    print("Generating Filter Frequency Response Images")
    print("="*60)
    
    generate_all_filter_responses()
