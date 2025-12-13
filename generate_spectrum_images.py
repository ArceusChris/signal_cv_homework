"""
Generate spectrum images for the report
1. Original images FFT spectrum
2. Filtered images FFT spectrum for each filter method
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def compute_fft_spectrum(image):
    """Compute FFT spectrum of an image"""
    # Convert to float
    f = np.fft.fft2(image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def plot_fft_spectrum(image, title, save_path):
    """Plot and save FFT spectrum"""
    spectrum = compute_fft_spectrum(image)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(spectrum, cmap='gray')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar(label='Magnitude (dB)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_original_spectrums():
    """Generate FFT spectrums for original images"""
    print("\n=== Generating Original Image Spectrums ===")
    
    # Load original images
    einstein = cv2.imread('einstein.jpg', cv2.IMREAD_GRAYSCALE)
    monroe = cv2.imread('monroe.jpg', cv2.IMREAD_GRAYSCALE)
    
    if einstein is None or monroe is None:
        print("Error: Could not load original images!")
        return
    
    # Create output directory
    output_dir = Path('original_spectrums')
    output_dir.mkdir(exist_ok=True)
    
    # Generate spectrums
    plot_fft_spectrum(einstein, 'Einstein Original - FFT Spectrum', 
                     output_dir / 'einstein_spectrum.png')
    plot_fft_spectrum(monroe, 'Monroe Original - FFT Spectrum', 
                     output_dir / 'monroe_spectrum.png')
    
    print(f"Original spectrums saved in: {output_dir}/")

def generate_filtered_spectrums():
    """Generate FFT spectrums for filtered hybrid images"""
    print("\n=== Generating Filtered Image Spectrums ===")
    
    filters = [
        ('ideal_filter_output', 'hybrid_D0_20.jpg', 'Ideal Filter (D0=20)'),
        ('butterworth_filter_output', 'hybrid_d0_20.jpg', 'Butterworth Filter (D0=20, n=4)'),
        ('elliptical_filter_output', 'hybrid_a20_b10.jpg', 'Elliptical Filter (a=20, b=10)'),
        ('gaussian_filter_output', 'hybrid_k15_s2.0.jpg', 'Gaussian Filter (σ=2.0)'),
        ('edge_detection_output', 'hybrid_edge_based.jpg', 'Sobel Edge Mixing'),
    ]
    
    # Create output directory
    output_dir = Path('filtered_spectrums')
    output_dir.mkdir(exist_ok=True)
    
    for folder, filename, title in filters:
        image_path = Path(folder) / filename
        if image_path.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                output_name = f"{folder.replace('_output', '')}_spectrum.png"
                plot_fft_spectrum(image, f'{title} - FFT Spectrum', 
                                output_dir / output_name)
            else:
                print(f"Warning: Could not load {image_path}")
        else:
            print(f"Warning: File not found - {image_path}")
    
    print(f"Filtered spectrums saved in: {output_dir}/")

def generate_comparison_figure():
    """Generate a comparison figure with original and filtered spectrums"""
    print("\n=== Generating Comparison Figures ===")
    
    # Load original images
    einstein = cv2.imread('einstein.jpg', cv2.IMREAD_GRAYSCALE)
    monroe = cv2.imread('monroe.jpg', cv2.IMREAD_GRAYSCALE)
    
    if einstein is None or monroe is None:
        print("Error: Could not load original images!")
        return
    
    # Create figure for original images comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Einstein image
    axes[0, 0].imshow(einstein, cmap='gray')
    axes[0, 0].set_title('Einstein - Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Einstein spectrum
    einstein_spectrum = compute_fft_spectrum(einstein)
    im1 = axes[0, 1].imshow(einstein_spectrum, cmap='gray')
    axes[0, 1].set_title('Einstein - FFT Spectrum', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Monroe image
    axes[1, 0].imshow(monroe, cmap='gray')
    axes[1, 0].set_title('Monroe - Original Image', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Monroe spectrum
    monroe_spectrum = compute_fft_spectrum(monroe)
    im2 = axes[1, 1].imshow(monroe_spectrum, cmap='gray')
    axes[1, 1].set_title('Monroe - FFT Spectrum', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('original_spectrums/comparison_originals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: original_spectrums/comparison_originals.png")
    
    # Create figure for all filtered results comparison
    filters = [
        ('ideal_filter_output', 'hybrid_D0_20.jpg', 'Ideal Filter'),
        ('butterworth_filter_output', 'hybrid_d0_20.jpg', 'Butterworth'),
        ('elliptical_filter_output', 'hybrid_a20_b10.jpg', 'Elliptical'),
        ('gaussian_filter_output', 'hybrid_k15_s2.0.jpg', 'Gaussian'),
        ('edge_detection_output', 'hybrid_edge_based.jpg', 'Sobel Edge'),
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for idx, (folder, filename, title) in enumerate(filters):
        row = idx // 5
        col = idx % 5
        
        image_path = Path(folder) / filename
        if image_path.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Show hybrid image
                axes[0, col].imshow(image, cmap='gray')
                axes[0, col].set_title(f'{title}\nHybrid Image', fontweight='bold', fontsize=10)
                axes[0, col].axis('off')
                
                # Show spectrum
                spectrum = compute_fft_spectrum(image)
                im = axes[1, col].imshow(spectrum, cmap='gray')
                axes[1, col].set_title(f'{title}\nFFT Spectrum', fontweight='bold', fontsize=10)
                axes[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('filtered_spectrums/comparison_all_filters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: filtered_spectrums/comparison_all_filters.png")

if __name__ == '__main__':
    print("="*60)
    print("Generating Spectrum Images for Report")
    print("="*60)
    
    # Generate all spectrum images
    generate_original_spectrums()
    generate_filtered_spectrums()
    generate_comparison_figure()
    
    print("\n" + "="*60)
    print("All spectrum images generated successfully!")
    print("="*60)
