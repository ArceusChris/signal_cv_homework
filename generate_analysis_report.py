"""
生成定量评估结果的详细分析报告
"""

import json
import numpy as np

def load_results(json_file='evaluation_results.json'):
    """加载评估结果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_analysis_report(results):
    """生成详细分析报告"""
    
    print("\n" + "="*80)
    print("混合图像定量评估 - 详细分析报告")
    print("Quantitative Evaluation - Detailed Analysis Report")
    print("="*80)
    
    # 1. 频率分离度分析
    print("\n【一、频率分离度分析】")
    print("-" * 80)
    print("频率分离度（Separation Score）衡量混合图像中高频和低频成分的共存程度")
    print("理想值：接近1表示高低频能量平衡良好，有利于产生双稳态视觉效果\n")
    
    sep_scores = [(r['description'].split('(')[0].strip(), 
                   r['frequency_metrics']['separation_score']) for r in results]
    sep_scores_sorted = sorted(sep_scores, key=lambda x: x[1], reverse=True)
    
    for i, (method, score) in enumerate(sep_scores_sorted, 1):
        stars = '★' * int(score * 10)
        print(f"{i}. {method:25s}: {score:.4f} {stars}")
        
        if score > 0.5:
            comment = "✓ 优秀 - 高低频分离清晰，混合效果显著"
        elif score > 0.3:
            comment = "○ 良好 - 分离度适中，有明显混合效果"
        elif score > 0.2:
            comment = "△ 一般 - 分离度较低，混合效果不够明显"
        else:
            comment = "✗ 较差 - 分离度很低，几乎没有混合效果"
        print(f"   {comment}")
    
    # 2. 高低频平衡分析
    print("\n【二、高低频能量平衡分析】")
    print("-" * 80)
    print("平衡比（Balance Ratio = 高频能量/低频能量）")
    print("理想值：接近1.0表示高低频能量相当，视觉效果最佳\n")
    
    for r in results:
        method = r['description'].split('(')[0].strip()
        bal = r['frequency_metrics']['balance_ratio']
        low = r['frequency_metrics']['low_freq_energy_ratio'] * 100
        high = r['frequency_metrics']['high_freq_energy_ratio'] * 100
        
        print(f"{method:25s}: 平衡比={bal:.4f}")
        print(f"  └─ 低频: {low:5.2f}%  │  高频: {high:4.2f}%")
        
        if bal > 0.5:
            comment = "接近理想平衡"
        elif bal > 0.1:
            comment = "高频占比适中"
        elif bal > 0.01:
            comment = "高频占比偏低，低频主导"
        else:
            comment = "高频占比极低，几乎为纯低频图像"
        print(f"     评价: {comment}\n")
    
    # 3. 对比度分析
    print("\n【三、对比度指标分析】")
    print("-" * 80)
    print("RMS对比度：衡量图像灰度值的标准差，值越大对比越强")
    print("Michelson对比度：衡量最亮与最暗区域的相对差异，范围0-1\n")
    
    rms_scores = [(r['description'].split('(')[0].strip(), 
                   r['contrast_metrics']['rms_contrast']) for r in results]
    rms_scores_sorted = sorted(rms_scores, key=lambda x: x[1], reverse=True)
    
    for i, (method, rms) in enumerate(rms_scores_sorted, 1):
        mic = next(r['contrast_metrics']['michelson_contrast'] 
                   for r in results if r['description'].split('(')[0].strip() == method)
        
        print(f"{i}. {method:25s}: RMS={rms:6.2f}, Michelson={mic:.4f}")
        
        if rms > 80:
            comment = "高对比度 - 图像层次丰富，细节清晰"
        elif rms > 60:
            comment = "适中对比度 - 视觉效果良好"
        else:
            comment = "较低对比度 - 图像偏灰，细节较少"
        print(f"   {comment}")
    
    # 4. 边缘特性分析
    print("\n【四、边缘特性分析】")
    print("-" * 80)
    print("边缘密度：边缘像素占图像的百分比，反映细节丰富程度")
    print("边缘强度：平均边缘梯度幅值，反映边缘的清晰度\n")
    
    for r in results:
        method = r['description'].split('(')[0].strip()
        density = r['edge_metrics']['edge_density'] * 100
        strength = r['edge_metrics']['edge_strength']
        
        print(f"{method:25s}:")
        print(f"  ├─ 边缘密度: {density:5.2f}% ", end="")
        
        if density > 18:
            print("(高 - 细节丰富)")
        elif density > 14:
            print("(中 - 适度细节)")
        else:
            print("(低 - 细节较少)")
        
        print(f"  └─ 边缘强度: {strength:6.2f} ", end="")
        
        if strength > 70:
            print("(强 - 边缘清晰锐利)")
        elif strength > 50:
            print("(中 - 边缘较为清晰)")
        else:
            print("(弱 - 边缘较为模糊)")
    
    # 5. 综合排名
    print("\n【五、综合性能排名】")
    print("-" * 80)
    print("基于多个指标的加权综合评分\n")
    
    # 计算综合得分（归一化后加权）
    综合得分 = []
    for r in results:
        method = r['description'].split('(')[0].strip()
        
        # 归一化各指标到0-1
        sep_norm = r['frequency_metrics']['separation_score']  # 已经0-1
        bal_norm = min(r['frequency_metrics']['balance_ratio'] / 0.5, 1.0)  # 0.5为满分
        rms_norm = min(r['contrast_metrics']['rms_contrast'] / 100, 1.0)
        edge_norm = min(r['edge_metrics']['edge_strength'] / 100, 1.0)
        
        # 加权求和（权重可调）
        score = (sep_norm * 0.35 +      # 分离度权重35%
                 bal_norm * 0.25 +      # 平衡性权重25%
                 rms_norm * 0.20 +      # 对比度权重20%
                 edge_norm * 0.20)      # 边缘质量权重20%
        
        综合得分.append((method, score, {
            '分离度': sep_norm,
            '平衡性': bal_norm,
            '对比度': rms_norm,
            '边缘质量': edge_norm
        }))
    
    综合得分_sorted = sorted(综合得分, key=lambda x: x[1], reverse=True)
    
    for i, (method, score, details) in enumerate(综合得分_sorted, 1):
        medal = ['🥇', '🥈', '🥉', '  ', '  '][i-1] if i <= 5 else '  '
        print(f"{medal} {i}. {method:25s}: 综合得分 {score:.4f}")
        print(f"      ├─ 分离度: {details['分离度']:.3f}")
        print(f"      ├─ 平衡性: {details['平衡性']:.3f}")
        print(f"      ├─ 对比度: {details['对比度']:.3f}")
        print(f"      └─ 边缘质量: {details['边缘质量']:.3f}\n")
    
    # 6. 关键发现和建议
    print("\n【六、关键发现与建议】")
    print("-" * 80)
    
    best_sep = max(results, key=lambda x: x['frequency_metrics']['separation_score'])
    best_bal = max(results, key=lambda x: x['frequency_metrics']['balance_ratio'])
    best_contrast = max(results, key=lambda x: x['contrast_metrics']['rms_contrast'])
    
    print(f"\n✓ 最佳频率分离度: {best_sep['description']}")
    print(f"  → 特点: 高低频成分共存最佳，双稳态视觉效果最明显")
    
    print(f"\n✓ 最佳频率平衡: {best_bal['description']}")
    print(f"  → 特点: 高低频能量分布最均衡，远近观看效果对比最强")
    
    print(f"\n✓ 最高对比度: {best_contrast['description']}")
    print(f"  → 特点: 图像细节最丰富，层次最分明")
    
    print("\n【建议】")
    print("• 对于强调边缘细节的应用，推荐使用 Sobel Edge Mixing 或 Butterworth 滤波器")
    print("• 对于追求自然平滑过渡，推荐使用 Gaussian 或 Butterworth 滤波器")
    print("• 理想滤波器虽然分离度较低，但边缘保持性能较好")
    print("• 椭圆滤波器提供方向性选择，适合特定方向特征的处理")
    
    print("\n" + "="*80)
    print("报告生成完成")
    print("="*80 + "\n")

if __name__ == '__main__':
    results = load_results()
    generate_analysis_report(results)
