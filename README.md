# ELISA图像处理系统

![Platform](https://img.shields.io/badge/platform-windows-lightgrey)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

ELISA图像处理系统是一个基于深度学习的图像分割工具，专门用于处理ELISA（酶联免疫吸附测定）实验中的图像，自动识别和分析微孔板中的目标区域。

## 功能特点

- **深度学习模型支持**：支持U-Net和Attention U-Net两种模型架构
- **图形用户界面**：基于PySide6的直观易用的GUI界面
- **批量处理**：支持批量处理多个图像文件
- **后处理优化**：提供形态学处理和目标筛选功能
- **结果可视化**：实时显示分割结果和原图叠加效果
- **ROI提取**：自动提取感兴趣区域用于后续分类任务

## 界面预览

![系统主界面](D:\aGitHub\elisaSystem\mainwindow.png)

## 系统要求

- Windows 10/11 (24H2版本测试通过)
- Python 3.8+
- PyTorch
- OpenCV
- PySide6

## 安装指南

1. 克隆项目代码：
```bash
git clone https://github.com/yourusername/elisaSystem.git
cd elisaSystem
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载预训练模型权重文件并放置在`weight`目录下

## 使用说明

1. 运行主程序：
```bash
python ElisaSystem_main.py
```

2. 选择模型和权重文件：
   - 通过菜单栏选择模型类型（U-Net或Attention U-Net）
   - 选择对应的预训练权重文件

3. 单张图像处理：
   - 点击"打开单张图片"按钮选择图像
   - 调整参数（步长、置信度等）
   - 点击"分割"按钮开始处理

4. 批量处理：
   - 点击"打开多张图片文件夹"选择图像目录
   - 设置保存路径
   - 点击"开始批量推理"处理所有图像

## 项目结构

```
elisaSystem/
├── model/                  # 模型定义文件
│   ├── UNet.py            # U-Net模型实现
│   └── attention_unet.py  # Attention U-Net模型实现
├── weight/                 # 预训练模型权重文件
├── ElisaSystem_main.py     # 主程序入口
├── inferThread.py          # 推理线程处理
├── postProcess.py          # 后处理功能
├── prepareClassification.py# ROI区域提取
└── overlay_mask_on_image.py# 结果可视化
```

## 技术细节

### 模型架构

- **U-Net**：经典的图像分割架构，包含编码器-解码器结构和跳跃连接
- **Attention U-Net**：在U-Net基础上引入注意力机制，提高分割精度

### 图像处理流程

1. 图像预处理和滑动窗口分割
2. 深度模型推理
3. 结果融合和二值化
4. 后处理优化（可选）
5. ROI区域提取（可选）

### 后处理功能

- 形态学去噪
- 面积阈值筛选
- 圆形度筛选
- 目标数量限制

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目采用MIT许可证，详情请见[LICENSE](LICENSE)文件。

## 联系方式

如有问题，请提交Issue或联系项目维护者。