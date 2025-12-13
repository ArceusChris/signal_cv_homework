#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = 'butterworth_filter_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def butterworth_hp_filter(shape, d0, n=4):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    d = np.sqrt(u**2 + v**2)
    d[crow, ccol] = 1e-10
    h = 1 / (1 + (d0 / d)**(2*n))
    return h

def butterworth_lp_filter(shape, d0, n=4):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    d = np.sqrt(u**2 + v**2)
    h = 1 / (1 + (d / d0)**(2*n))
    return h

def process_cutoff(einstein_gray, monroe_gray, d0):
    f_einstein = np.fft.fft2(einstein_gray)
    f_monroe = np.fft.fft2(monroe_gray)
    fshift_einstein = np.fft.fftshift(f_einstein)
    fshift_monroe = np.fft.fftshift(f_monroe)
    
    hp_filter = butterworth_hp_filter(einstein_gray.shape, d0)
    lp_filter = butterworth_lp_filter(monroe_gray.shape, d0)
    
    fshift_hp = fshift_einstein * hp_filter
    fshift_lp = fshift_monroe * lp_filter
    
    f_hp = np.fft.ifftshift(fshift_hp)
    f_lp = np.fft.ifftshift(fshift_lp)
    einstein_hp = np.fft.ifft2(f_hp)
    monroe_lp = np.fft.ifft2(f_lp)
    einstein_hp = np.abs(einstein_hp)
    monroe_lp = np.abs(monroe_lp)
    
    hybrid = einstein_hp + monroe_lp
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(hp_filter, cmap='gray')
    axes[0, 0].set_title(f'HP Filter D0={d0}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(lp_filter, cmap='gray')
    axes[0, 1].set_title(f'LP Filter D0={d0}')
    axes[0, 1].axis('off')
    
    center = hp_filter.shape[0] // 2
    axes[0, 2].plot(hp_filter[center, :], label='HP')
    axes[0, 2].plot(lp_filter[center, :], label='LP')
    axes[0, 2].set_title('Filter Profile')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    axes[1, 0].imshow(einstein_hp, cmap='gray')
    axes[1, 0].set_title('Einstein HP')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(monroe_lp, cmap='gray')
    axes[1, 1].set_title('Monroe LP')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(hybrid, cmap='gray')
    axes[1, 2].set_title('Hybrid Image')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'butterworth_d0_{d0}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'hybrid_d0_{d0}.jpg'), 
                np.clip(hybrid, 0, 255).astype(np.uint8))
    
    print(f'  Done D0={d0}')
    return hybrid

def main():
    print('=' * 60)
    print('Butterworth Filter Processing (Order 4)')
    print('=' * 60)
    
    einstein = cv2.imread('einstein.jpg', cv2.IMREAD_GRAYSCALE)
    monroe = cv2.imread('monroe.jpg', cv2.IMREAD_GRAYSCALE)
    
    min_height = min(einstein.shape[0], monroe.shape[0])
    min_width = min(einstein.shape[1], monroe.shape[1])
    einstein = cv2.resize(einstein, (min_width, min_height))
    monroe = cv2.resize(monroe, (min_width, min_height))
    
    print(f'Image size: {einstein.shape}')
    
    cutoff_freqs = [5, 10, 20, 30, 50]
    print(f'\nProcessing {len(cutoff_freqs)} cutoff frequencies: {cutoff_freqs}')
    
    results = []
    for d0 in cutoff_freqs:
        hybrid = process_cutoff(einstein, monroe, d0)
        results.append((d0, hybrid))
    
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results, figsize=(5*n_results, 5))
    if n_results == 1:
        axes = [axes]
    
    for idx, (d0, hybrid) in enumerate(results):
        axes[idx].imshow(hybrid, cmap='gray')
        axes[idx].set_title(f'D0={d0}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_all.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'\n{"=" * 60}')
    print(f'Complete! Results in: {OUTPUT_DIR}/')
    print('=' * 60)

if __name__ == '__main__':
    main()
