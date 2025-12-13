#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'ideal_filter_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ideal_high_pass_filter(shape, cutoff):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v, u)
    D = np.sqrt((U - center_col)**2 + (V - center_row)**2)
    H = np.zeros((rows, cols))
    H[D > cutoff] = 1
    return H


def ideal_low_pass_filter(shape, cutoff):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v, u)
    D = np.sqrt((U - center_col)**2 + (V - center_row)**2)
    H = np.zeros((rows, cols))
    H[D <= cutoff] = 1
    return H


def process_cutoff(einstein, monroe, eins_fft, mon_fft, cutoff, idx):
    hp_filt = ideal_high_pass_filter(einstein.shape, cutoff)
    lp_filt = ideal_low_pass_filter(monroe.shape, cutoff)
    
    eins_filtered_fft = eins_fft * hp_filt
    mon_filtered_fft = mon_fft * lp_filt
    
    eins_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(eins_filtered_fft)))
    mon_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(mon_filtered_fft)))
    
    combined = eins_filtered + mon_filtered
    combined_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(hp_filt, cmap='gray')
    axes[0].set_title(f'Ideal HP Filter (D0={cutoff})')
    axes[0].axis('off')
    axes[1].imshow(lp_filt, cmap='gray')
    axes[1].set_title(f'Ideal LP Filter (D0={cutoff})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{idx:02d}_filters_D0_{cutoff}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(eins_filtered, cmap='gray')
    axes[0].set_title(f'Einstein HP (D0={cutoff})')
    axes[0].axis('off')
    axes[1].imshow(mon_filtered, cmap='gray')
    axes[1].set_title(f'Monroe LP (D0={cutoff})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{idx:02d}_filtered_D0_{cutoff}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_norm, cmap='gray')
    plt.title(f'Hybrid Image (D0={cutoff})')
    plt.axis('off')
    plt.savefig(f'{OUTPUT_DIR}/{idx:02d}_hybrid_D0_{cutoff}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    cv2.imwrite(f'{OUTPUT_DIR}/hybrid_D0_{cutoff}.jpg', combined_norm)
    print(f"  Done D0={cutoff}")
    return combined_norm


def main():
    print("="*60)
    print("Ideal Filter Processing")
    print("="*60)
    
    einstein = cv2.imread('einstein.jpg', cv2.IMREAD_GRAYSCALE)
    monroe = cv2.imread('monroe.jpg', cv2.IMREAD_GRAYSCALE)
    
    h = min(einstein.shape[0], monroe.shape[0])
    w = min(einstein.shape[1], monroe.shape[1])
    einstein = cv2.resize(einstein, (w, h))
    monroe = cv2.resize(monroe, (w, h))
    print(f"Image size: {einstein.shape}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(einstein, cmap='gray')
    axes[0].set_title('Einstein')
    axes[0].axis('off')
    axes[1].imshow(monroe, cmap='gray')
    axes[1].set_title('Monroe')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/00_original.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    eins_fft = np.fft.fftshift(np.fft.fft2(einstein))
    mon_fft = np.fft.fftshift(np.fft.fft2(monroe))
    
    cutoffs = [5, 10, 20, 30, 50]
    print(f"\nProcessing {len(cutoffs)} cutoff frequencies: {cutoffs}")
    
    results = []
    for i, cutoff in enumerate(cutoffs):
        result = process_cutoff(einstein, monroe, eins_fft, mon_fft, cutoff, i+1)
        results.append((cutoff, result))
    
    fig, axes = plt.subplots(2, len(cutoffs), figsize=(4*len(cutoffs), 8))
    for i, (cutoff, result) in enumerate(results):
        filt = ideal_high_pass_filter(einstein.shape, cutoff)
        axes[0, i].imshow(filt, cmap='gray')
        axes[0, i].set_title(f'D0={cutoff}')
        axes[0, i].axis('off')
        axes[1, i].imshow(result, cmap='gray')
        axes[1, i].set_title(f'Hybrid')
        axes[1, i].axis('off')
    plt.suptitle('Cutoff Frequency Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/99_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print(f"Complete! Results in: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
