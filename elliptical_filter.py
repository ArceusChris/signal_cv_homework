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

OUTPUT_DIR = 'elliptical_filter_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def elliptical_hp_filter(shape, a, b, angle=0):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v, u)
    U_c = U - center_col
    V_c = V - center_row
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    U_r = U_c * cos_a + V_c * sin_a
    V_r = -U_c * sin_a + V_c * cos_a
    ellipse_val = (U_r ** 2) / (a ** 2) + (V_r ** 2) / (b ** 2)
    H = np.zeros((rows, cols))
    H[ellipse_val > 1] = 1
    return H


def elliptical_lp_filter(shape, a, b, angle=0):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v, u)
    U_c = U - center_col
    V_c = V - center_row
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    U_r = U_c * cos_a + V_c * sin_a
    V_r = -U_c * sin_a + V_c * cos_a
    ellipse_val = (U_r ** 2) / (a ** 2) + (V_r ** 2) / (b ** 2)
    H = np.zeros((rows, cols))
    H[ellipse_val <= 1] = 1
    return H


def process_params(einstein, monroe, eins_fft, mon_fft, a, b, idx):
    hp_filt = elliptical_hp_filter(einstein.shape, a, b, 0)
    lp_filt = elliptical_lp_filter(monroe.shape, a, b, 0)
    
    eins_filtered_fft = eins_fft * hp_filt
    mon_filtered_fft = mon_fft * lp_filt
    
    eins_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(eins_filtered_fft)))
    mon_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(mon_filtered_fft)))
    
    combined = eins_filtered + mon_filtered
    combined_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(hp_filt, cmap='gray')
    axes[0].set_title(f'Elliptical HP Filter (a={a}, b={b})')
    axes[0].axis('off')
    axes[1].imshow(lp_filt, cmap='gray')
    axes[1].set_title(f'Elliptical LP Filter (a={a}, b={b})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{idx:02d}_filters_a{a}_b{b}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(eins_filtered, cmap='gray')
    axes[0].set_title(f'Einstein HP (a={a}, b={b})')
    axes[0].axis('off')
    axes[1].imshow(mon_filtered, cmap='gray')
    axes[1].set_title(f'Monroe LP (a={a}, b={b})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{idx:02d}_filtered_a{a}_b{b}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_norm, cmap='gray')
    plt.title(f'Hybrid Image (a={a}, b={b})')
    plt.axis('off')
    plt.savefig(f'{OUTPUT_DIR}/{idx:02d}_hybrid_a{a}_b{b}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    cv2.imwrite(f'{OUTPUT_DIR}/hybrid_a{a}_b{b}.jpg', combined_norm)
    print(f"  Done a={a}, b={b}")
    return combined_norm


def main():
    print("="*60)
    print("Elliptical Filter Processing")
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
    
    params = [(10, 5), (15, 10), (20, 10), (30, 15), (40, 20)]
    print(f"\nProcessing {len(params)} parameter sets: {params}")
    
    results = []
    for i, (a, b) in enumerate(params):
        result = process_params(einstein, monroe, eins_fft, mon_fft, a, b, i+1)
        results.append((a, b, result))
    
    fig, axes = plt.subplots(2, len(params), figsize=(4*len(params), 8))
    for i, (a, b, result) in enumerate(results):
        filt = elliptical_hp_filter(einstein.shape, a, b, 0)
        axes[0, i].imshow(filt, cmap='gray')
        axes[0, i].set_title(f'a={a}, b={b}')
        axes[0, i].axis('off')
        axes[1, i].imshow(result, cmap='gray')
        axes[1, i].set_title(f'Hybrid')
        axes[1, i].axis('off')
    plt.suptitle('Parameter Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/99_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print(f"Complete! Results in: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
