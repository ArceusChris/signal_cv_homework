"""
Quantitative Evaluation of Hybrid Images
定量评估混合图像的质量指标
包括：频率分离度、对比度指标、能量分布等
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

def compute_fft_spectrum(image):
    """计算图像的FFT频谱"""
    f = np.fft.fft2(image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    return fshift, magnitude_spectrum

def compute_frequency_energy_distribution(magnitude_spectrum, D0):
    """
    计算频率能量分布
    
    参数:
        magnitude_spectrum: FFT幅度谱
        D0: 截止频率（半径）
    
    返回:
        low_freq_energy: 低频能量占比
        high_freq_energy: 高频能量占比
        energy_concentration: 能量集中度
    """
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    
    # 创建距离矩阵
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - crow)**2 + (x - ccol)**2)
    
    # 计算总能量
    total_energy = np.sum(magnitude_spectrum**2)
    
    # 低频能量（距离中心 <= D0）
    low_freq_mask = distance <= D0
    low_freq_energy = np.sum((magnitude_spectrum * low_freq_mask)**2)
    low_freq_ratio = low_freq_energy / total_energy
    
    # 高频能量（距离中心 > D0）
    high_freq_mask = distance > D0
    high_freq_energy = np.sum((magnitude_spectrum * high_freq_mask)**2)
    high_freq_ratio = high_freq_energy / total_energy
    
    # 能量集中度（使用熵的倒数）
    # 归一化频谱为概率分布
    prob_spectrum = (magnitude_spectrum**2) / total_energy
    prob_spectrum = prob_spectrum[prob_spectrum > 0]  # 移除零值
    entropy = -np.sum(prob_spectrum * np.log(prob_spectrum))
    energy_concentration = 1.0 / entropy if entropy > 0 else 0
    
    return low_freq_ratio, high_freq_ratio, energy_concentration

def compute_frequency_separation_metric(magnitude_spectrum, D0):
    """
    计算频率分离度
    理想混合图像应该在低频和高频都有显著能量
    
    返回:
        separation_score: 分离度评分（0-1，越接近1表示分离越好）
        balance_ratio: 高低频能量平衡比（理想值为1）
    """
    low_ratio, high_ratio, _ = compute_frequency_energy_distribution(magnitude_spectrum, D0)
    
    # 分离度：两者都有能量才好（几何平均）
    separation_score = np.sqrt(low_ratio * high_ratio) * 2  # *2归一化到0-1
    
    # 平衡比：高频/低频，理想值为1
    balance_ratio = high_ratio / low_ratio if low_ratio > 0 else 0
    
    return separation_score, balance_ratio

def compute_rms_contrast(image):
    """
    计算RMS（均方根）对比度
    
    参数:
        image: 灰度图像
    
    返回:
        rms_contrast: RMS对比度值
    """
    mean_intensity = np.mean(image)
    rms_contrast = np.sqrt(np.mean((image - mean_intensity)**2))
    return rms_contrast

def compute_michelson_contrast(image):
    """
    计算Michelson对比度
    
    参数:
        image: 灰度图像
    
    返回:
        michelson_contrast: Michelson对比度值（0-1）
    """
    I_max = np.max(image)
    I_min = np.min(image)
    
    if I_max + I_min == 0:
        return 0
    
    michelson_contrast = (I_max - I_min) / (I_max + I_min)
    return michelson_contrast

def compute_weber_contrast(image):
    """
    计算Weber对比度
    
    参数:
        image: 灰度图像
    
    返回:
        weber_contrast: Weber对比度值
    """
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    if mean_intensity == 0:
        return 0
    
    weber_contrast = std_intensity / mean_intensity
    return weber_contrast

def compute_edge_density(image):
    """
    计算边缘密度（边缘像素占比）
    
    参数:
        image: 灰度图像
    
    返回:
        edge_density: 边缘密度（0-1）
        edge_strength: 平均边缘强度
    """
    # 使用Sobel算子检测边缘
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 边缘阈值（使用Otsu自动阈值）
    edge_normalized = (edge_magnitude / edge_magnitude.max() * 255).astype(np.uint8)
    _, edge_binary = cv2.threshold(edge_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 边缘密度
    edge_density = np.sum(edge_binary > 0) / edge_binary.size
    
    # 平均边缘强度
    edge_strength = np.mean(edge_magnitude)
    
    return edge_density, edge_strength

def evaluate_single_image(image_path, D0=20):
    """
    评估单张图像的所有指标
    
    参数:
        image_path: 图像路径
        D0: 截止频率
    
    返回:
        metrics: 包含所有指标的字典
    """
    # 读取图像
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Could not load {image_path}")
        return None
    
    # 计算FFT
    fshift, magnitude_spectrum = compute_fft_spectrum(image)
    
    # 频域指标
    low_freq_ratio, high_freq_ratio, energy_concentration = \
        compute_frequency_energy_distribution(magnitude_spectrum, D0)
    
    separation_score, balance_ratio = \
        compute_frequency_separation_metric(magnitude_spectrum, D0)
    
    # 空域指标
    rms_contrast = compute_rms_contrast(image)
    michelson_contrast = compute_michelson_contrast(image)
    weber_contrast = compute_weber_contrast(image)
    
    edge_density, edge_strength = compute_edge_density(image)
    
    # 整合所有指标
    metrics = {
        'image_path': str(image_path),
        'image_name': Path(image_path).name,
        'frequency_metrics': {
            'low_freq_energy_ratio': float(low_freq_ratio),
            'high_freq_energy_ratio': float(high_freq_ratio),
            'energy_concentration': float(energy_concentration),
            'separation_score': float(separation_score),
            'balance_ratio': float(balance_ratio),
        },
        'contrast_metrics': {
            'rms_contrast': float(rms_contrast),
            'michelson_contrast': float(michelson_contrast),
            'weber_contrast': float(weber_contrast),
        },
        'edge_metrics': {
            'edge_density': float(edge_density),
            'edge_strength': float(edge_strength),
        }
    }
    
    return metrics

def evaluate_all_hybrid_images(D0=20):
    """
    评估所有滤波器生成的混合图像
    
    参数:
        D0: 截止频率
    
    返回:
        all_results: 所有图像的评估结果
    """
    # 定义要评估的图像
    image_configs = [
        ('ideal_filter_output', 'hybrid_D0_20.jpg', 'Ideal Filter (D0=20)'),
        ('butterworth_filter_output', 'hybrid_d0_20.jpg', 'Butterworth Filter (D0=20, n=4)'),
        ('elliptical_filter_output', 'hybrid_a20_b10.jpg', 'Elliptical Filter (a=20, b=10)'),
        ('gaussian_filter_output', 'hybrid_k15_s2.0.jpg', 'Gaussian Filter (σ=2.0)'),
        ('edge_detection_output', 'hybrid_edge_based.jpg', 'Sobel Edge Mixing'),
    ]
    
    all_results = []
    
    print("\n" + "="*70)
    print("混合图像定量评估 - Quantitative Evaluation")
    print("="*70)
    print(f"截止频率 D0 = {D0}\n")
    
    for folder, filename, description in image_configs:
        image_path = Path(folder) / filename
        
        if not image_path.exists():
            print(f"⚠️  未找到: {image_path}")
            continue
        
        print(f"\n处理: {description}")
        print(f"文件: {image_path}")
        
        metrics = evaluate_single_image(image_path, D0)
        
        if metrics:
            metrics['description'] = description
            all_results.append(metrics)
            
            # 打印关键指标
            print(f"  ├─ 频率分离度: {metrics['frequency_metrics']['separation_score']:.4f}")
            print(f"  ├─ 高低频平衡比: {metrics['frequency_metrics']['balance_ratio']:.4f}")
            print(f"  ├─ 低频能量占比: {metrics['frequency_metrics']['low_freq_energy_ratio']:.2%}")
            print(f"  ├─ 高频能量占比: {metrics['frequency_metrics']['high_freq_energy_ratio']:.2%}")
            print(f"  ├─ RMS对比度: {metrics['contrast_metrics']['rms_contrast']:.2f}")
            print(f"  ├─ Michelson对比度: {metrics['contrast_metrics']['michelson_contrast']:.4f}")
            print(f"  ├─ 边缘密度: {metrics['edge_metrics']['edge_density']:.2%}")
            print(f"  └─ 边缘强度: {metrics['edge_metrics']['edge_strength']:.2f}")
    
    return all_results

def save_results_to_json(results, output_file='evaluation_results.json'):
    """保存评估结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 评估结果已保存到: {output_file}")

def generate_comparison_table(results):
    """生成对比表格（LaTeX格式）"""
    print("\n" + "="*70)
    print("LaTeX表格代码")
    print("="*70)
    
    latex_code = r"""
\begin{table}[h]
\centering
\caption{混合图像定量评估结果对比}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{滤波方法} & \textbf{分离度} & \textbf{平衡比} & \textbf{RMS对比度} & \textbf{Michelson} & \textbf{边缘密度} \\
\hline
"""
    
    for result in results:
        name = result['description'].split('(')[0].strip()
        sep = result['frequency_metrics']['separation_score']
        bal = result['frequency_metrics']['balance_ratio']
        rms = result['contrast_metrics']['rms_contrast']
        mic = result['contrast_metrics']['michelson_contrast']
        edge = result['edge_metrics']['edge_density']
        
        latex_code += f"{name} & {sep:.3f} & {bal:.3f} & {rms:.2f} & {mic:.3f} & {edge:.2%} \\\\\n"
    
    latex_code += r"""\hline
\end{tabular}
\end{table}
"""
    
    print(latex_code)
    
    # 保存到文件
    with open('comparison_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_code)
    print("✅ LaTeX表格已保存到: comparison_table.tex")

def plot_comparison_charts(results, output_dir='evaluation_plots'):
    """绘制对比图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    methods = [r['description'].split('(')[0].strip() for r in results]
    
    # 1. 频率分离度和平衡比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    separation_scores = [r['frequency_metrics']['separation_score'] for r in results]
    balance_ratios = [r['frequency_metrics']['balance_ratio'] for r in results]
    
    ax1.bar(methods, separation_scores, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Separation Score', fontsize=12)
    ax1.set_title('Frequency Separation Score', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax2.bar(methods, balance_ratios, color='coral', alpha=0.8)
    ax2.set_ylabel('Balance Ratio (High/Low)', fontsize=12)
    ax2.set_title('High/Low Frequency Balance Ratio', fontsize=14, fontweight='bold')
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Ideal Balance (1.0)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {output_dir}/frequency_metrics_comparison.png")
    
    # 2. 对比度指标
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rms_values = [r['contrast_metrics']['rms_contrast'] for r in results]
    michelson_values = [r['contrast_metrics']['michelson_contrast'] for r in results]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x - width/2, rms_values, width, label='RMS Contrast', color='green', alpha=0.8)
    ax.bar(x + width/2, [m*100 for m in michelson_values], width, 
           label='Michelson Contrast (×100)', color='purple', alpha=0.8)
    
    ax.set_ylabel('Contrast Value', fontsize=12)
    ax.set_title('Contrast Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'contrast_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {output_dir}/contrast_metrics_comparison.png")
    
    # 3. 能量分布
    fig, ax = plt.subplots(figsize=(12, 6))
    
    low_freq = [r['frequency_metrics']['low_freq_energy_ratio']*100 for r in results]
    high_freq = [r['frequency_metrics']['high_freq_energy_ratio']*100 for r in results]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x - width/2, low_freq, width, label='Low Frequency Energy', color='blue', alpha=0.8)
    ax.bar(x + width/2, high_freq, width, label='High Frequency Energy', color='red', alpha=0.8)
    
    ax.set_ylabel('Energy Ratio (%)', fontsize=12)
    ax.set_title('Frequency Energy Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {output_dir}/energy_distribution_comparison.png")
    
    # 4. 综合雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 归一化指标到0-1范围
    categories = ['Separation\nScore', 'Balance\n(normalized)', 'RMS Contrast\n(normalized)', 
                  'Michelson\nContrast', 'Edge\nDensity']
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for idx, result in enumerate(results):
        values = [
            result['frequency_metrics']['separation_score'],
            min(result['frequency_metrics']['balance_ratio'], 2) / 2,  # 归一化到0-1
            result['contrast_metrics']['rms_contrast'] / 100,  # 归一化
            result['contrast_metrics']['michelson_contrast'],
            result['edge_metrics']['edge_density'],
        ]
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=methods[idx], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Comprehensive Quality Metrics Radar Chart', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_radar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {output_dir}/comprehensive_radar_chart.png")

if __name__ == '__main__':
    print("="*70)
    print("混合图像定量评估系统")
    print("Quantitative Evaluation System for Hybrid Images")
    print("="*70)
    
    # 评估所有混合图像
    results = evaluate_all_hybrid_images(D0=20)
    
    if results:
        # 保存结果到JSON
        save_results_to_json(results)
        
        # 生成LaTeX对比表格
        generate_comparison_table(results)
        
        # 绘制对比图表
        plot_comparison_charts(results)
        
        print("\n" + "="*70)
        print("✅ 评估完成！生成的文件：")
        print("  1. evaluation_results.json - 完整评估数据")
        print("  2. comparison_table.tex - LaTeX表格代码")
        print("  3. evaluation_plots/ - 对比图表")
        print("     ├─ frequency_metrics_comparison.png")
        print("     ├─ contrast_metrics_comparison.png")
        print("     ├─ energy_distribution_comparison.png")
        print("     └─ comprehensive_radar_chart.png")
        print("="*70)
    else:
        print("\n⚠️  没有找到可评估的图像")
