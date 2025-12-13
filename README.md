# Image Filter Processing Scripts

图像处理脚本集合 - 包含频域滤波、空域卷积和边缘检测

## 文件说明

### 脚本文件

**频域滤波器：**
- `ideal_filter.py` - 理想滤波器（高通和低通）
- `elliptical_filter.py` - 椭圆滤波器
- `butterworth_filter.py` - 四阶巴特沃斯滤波器

**空域处理：**
- `gaussian_filter.py` - 高斯卷积核滤波器
- `edge_detection.py` - Sobel 和 Canny 边缘检测算子

**工具脚本：**
- `run_all_filters.sh` - 批量运行所有脚本

### 输入图像
- `einstein.jpg` - 爱因斯坦图像（用于高通滤波）
- `monroe.jpg` - 梦露图像（用于低通滤波）

## 使用方法

### 单独运行某个脚本

```bash
# 频域滤波器
python ideal_filter.py           # 理想滤波器
python elliptical_filter.py      # 椭圆滤波器
python butterworth_filter.py     # 巴特沃斯滤波器

# 空域处理
python gaussian_filter.py        # 高斯卷积核
python edge_detection.py         # Sobel 和 Canny 边缘检测
```

### 批量运行所有脚本

```bash
./run_all_filters.sh
```

## 输出结果

每个滤波器都会在独立的文件夹中保存结果：

### 1. ideal_filter_output/
理想滤波器输出，处理 5 个不同的截止频率：
- D0 = 5, 10, 20, 30, 50

每个截止频率生成：
- `01-05_filters_D0_*.png` - 滤波器可视化
- `01-05_filtered_D0_*.png` - 滤波后的图像
- `01-05_hybrid_D0_*.png` - 混合图像
- `hybrid_D0_*.jpg` - 最终混合图像（JPEG格式）
- `99_comparison.png` - 所有结果对比图

### 2. elliptical_filter_output/
椭圆滤波器输出，处理 5 组不同的参数：
- (a=10, b=5), (15, 10), (20, 10), (30, 15), (40, 20)

每组参数生成：
- `01-05_filters_a*_b*.png` - 椭圆滤波器可视化
- `01-05_filtered_a*_b*.png` - 滤波后的图像
- `01-05_hybrid_a*_b*.png` - 混合图像
- `hybrid_a*_b*.jpg` - 最终混合图像（JPEG格式）
- `99_comparison.png` - 所有结果对比图

### 3. butterworth_filter_output/
四阶巴特沃斯滤波器输出，处理 5 个不同的截止频率：
- D0 = 5, 10, 20, 30, 50（阶数 n=4）

每个截止频率生成：
- `butterworth_d0_*.png` - 包含滤波器、滤波图像和混合图像的完整可视化
- `hybrid_d0_*.jpg` - 最终混合图像（JPEG格式）
- `comparison_all.png` - 所有结果对比图

## 技术细节

### 滤波器类型

1. **理想滤波器 (Ideal Filter)**
   - 特点：在截止频率处有尖锐的过渡
   - 高通：H(u,v) = 0 if D(u,v) ≤ D0, else 1
   - 低通：H(u,v) = 1 if D(u,v) ≤ D0, else 0

2. **椭圆滤波器 (Elliptical Filter)**
   - 特点：在频率域使用椭圆形状的滤波器
   - 参数：a（长轴）、b（短轴）
   - 可以针对不同方向的频率分量进行不同的处理

3. **巴特沃斯滤波器 (Butterworth Filter)**
   - 特点：平滑的频率响应，无振铃效应
   - 公式：H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
   - 阶数：n=4（提供较好的平滑过渡）

### 处理流程

1. 读取输入图像（灰度化）
2. 将图像统一调整到相同尺寸
3. 对图像进行二维FFT变换
4. 在频率域应用滤波器
5. 进行逆FFT变换得到滤波后的图像
6. 合成混合图像（高通+低通）
7. 保存所有可视化结果

## 依赖库

- numpy - 数值计算和FFT
- opencv-python (cv2) - 图像读写
- matplotlib - 可视化（使用Agg后端，无需显示窗口）

## 注意事项

- 脚本使用 matplotlib 的 Agg 后端，不会弹出窗口
- 适合在SSH远程环境或无图形界面的服务器上运行
- 所有输出图像自动保存到对应的文件夹
- 高通和低通滤波器使用相同的截止频率参数

## 效果说明

### 频域滤波混合图像
生成的混合图像具有以下特点：
- 近距离观看：能看到爱因斯坦的高频细节
- 远距离观看：能看到梦露的低频轮廓
- 这种视觉效果展示了人类视觉系统对不同频率成分的处理方式

### 4. gaussian_filter_output/

高斯卷积核处理结果，测试 5 组不同的参数：
- (kernel_size=5, σ=1.0), (9, 1.5), (15, 2.0), (21, 3.0), (31, 5.0)

每组参数生成：
- `gaussian_k*_s*.png` - 完整分析（包含：卷积核、频率响应、3D视图、滤波结果、FFT分析）
- `einstein_filtered_k*_s*.jpg` - 爱因斯坦滤波结果
- `monroe_filtered_k*_s*.jpg` - 梦露滤波结果
- `hybrid_k*_s*.jpg` - 混合图像
- `comparison_all.png` - 所有参数对比图

**高斯滤波器特点：**
- 频率响应：低通特性，平滑抑制高频噪声
- σ越大，平滑效果越强，频率响应越窄
- 无振铃效应，过渡平滑
- 在空域和频域都是高斯函数（傅里叶变换对）

### 5. edge_detection_output/

**Sobel算子边缘混合图像处理结果：**

生成文件：
- `sobel_frequency_response.png` - Sobel算子频率响应特性分析（包含：卷积核、频域响应、剖面图、3D视图、极坐标图）
- `comprehensive_analysis.png` - 完整处理流程（原图、边缘提取、梯度分量、混合结果、FFT分析）
- `multi_scale_view.png` - 不同观看距离的视觉效果对比
- `einstein_edges.jpg` - 爱因斯坦边缘图（使用Sobel算子提取）
- `monroe_without_edges.jpg` - 梦露去除边缘后的图像（原图 - 边缘）
- `hybrid_edge_based.jpg` - 混合图像（爱因斯坦边缘 + 梦露无边缘）

**处理方法：**
1. 使用Sobel算子提取爱因斯坦的边缘信息（高频成分）
2. 从梦露图像中减去边缘信息，得到平滑的主体部分（低频成分）
3. 将爱因斯坦边缘与梦露无边缘部分叠加，形成混合图像

**Sobel算子特点：**
- 基于一阶梯度的边缘检测
- 使用 3×3 卷积核计算 X 和 Y 方向导数
- **频率响应：高通滤波特性**
  - 在频域中心（低频区）响应弱
  - 在频域边缘（高频区）响应强
  - X方向核对垂直边缘敏感（水平频率）
  - Y方向核对水平边缘敏感（垂直频率）
- 对噪声有一定抑制能力
- 计算效率高，适合实时处理

**视觉效果：**
- 近距离观看：能清晰看到爱因斯坦的边缘细节
- 远距离观看：能看到梦露的整体轮廓和形状
- 混合效果展示了高频（边缘）和低频（平滑区域）成分的分离与组合
