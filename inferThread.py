import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal
import torch
from sympy.strategies.core import switch
from torchvision import transforms

from model.UNet import Unet
from model.attention_unet import AttU_Net
from postProcess import postprocess_mask

#预处理
x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])#使用默认的imagenet的均值和方差
    ])
#中文路径读取
def imread_unicode(path):
    """支持中文路径的读取"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 或 cv2.IMREAD_GRAYSCALE
    return img

#加载模型
def load_model(model, checkpoint_path):
    """
    加载模型权重，并忽略训练过程中的元数据
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 强制加载到CPU

    model_state_dict = checkpoint.get('model_state_dict', checkpoint)  # 提取模型状态字典

    if isinstance(model_state_dict, dict):
        # 更新模型的参数
        model.load_state_dict(model_state_dict, strict=False)  # strict=False可以允许部分参数不匹配

    return model

def make_weight_mask(patch_size, enabled=True):
    """
    生成用于拼接的二维权重 mask。

    - enabled=True: 使用高斯形状的中心重权、边缘轻权，减轻重叠拼接痕迹。
    - enabled=False: 使用全1均匀权重。
    """
    if not enabled:
        return np.ones((patch_size, patch_size), dtype=np.float32)
    y = np.linspace(-1, 1, patch_size)
    x = np.linspace(-1, 1, patch_size)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx**2 + yy**2)
    sigma = 0.5
    weight = np.exp(-(d**2) / (2 * sigma**2))
    return weight.astype(np.float32)


def record_inference_time(model_name, inference_time, stride, device, image_filename=None,
                          csv_path="inference_times.csv"):
    """
    记录模型推理时间到CSV文件

    参数:
    - model_name: 模型名称
    - inference_time: 推理时间（秒）
    - stride: 步长
    - device: 使用的设备 (CPU/GPU)
    - image_filename: 推理的图像文件名
    - csv_path: CSV文件路径
    """
    # 检查文件是否存在
    file_exists = os.path.exists(csv_path)

    # 准备写入的数据
    data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'stride': stride,
        'inference_time': f"{inference_time:.4f}",
        'device': device
    }

    # 如果提供了图像文件名，则添加到记录中
    if image_filename:
        data['image_filename'] = image_filename

    # 写入CSV文件
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'model_name', 'stride', 'inference_time', 'device']
        # 如果提供了图像文件名，添加到字段列表中
        if image_filename:
            fieldnames.append('image_filename')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()

        # 写入数据行
        writer.writerow(data)
    print(f"推理时间已记录到 {csv_path}")


def process_large_image(model, image_path, patch_size=256, stride=192, use_weight_mask=True,
                        bin_thresh=0.5, use_gpu=False,progress_signal=None):
    """
    处理大尺寸图像（推荐）：滑窗裁切 -> 模型预测 -> 加权融合 -> 二值化保存

    参数:
    - model: 已加载的模型（eval 模式）
    - image_path: 输入大图路径
    - patch_size: 滑窗块尺寸（默认 256）
    - stride: 滑窗步长（默认 192，建议 3/4*patch，保证重叠）
    - use_weight_mask: 是否使用高斯权重融合（默认 True）。False 则均匀融合。
    - bin_thresh: 二值化阈值（默认 0.5）

    - use_gpu: 是否使用GPU进行推理（默认 False）
    """

    # 加载并预处理大图像（使用 OpenCV，保持与训练一致的 BGR 通道）
    # large_image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # shape: (H, W, C[BGR])
    large_image = imread_unicode(image_path)
    large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)
    if large_image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    height, width = large_image.shape[:2]
    # 输出预测图与累计权重
    full_pred = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    # 生成权重mask（可选）
    weight = make_weight_mask(patch_size, enabled=use_weight_mask)
    # 计算窗口与显示进度
    num_patches_h = (height - patch_size) // stride + 1
    num_patches_w = (width - patch_size) // stride + 1
    total_patches = num_patches_h * num_patches_w
    # 记录时间
    start_time = time.time()  # 记录开始时间
    # 检查是否使用GPU
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"使用设备: {device}")
    # 逐个处理每个图像块
    model.eval()
    patch_count = 0
    with torch.no_grad():
        for top in range(0, height - patch_size + 1, stride):
            for left in range(0, width - patch_size + 1, stride):
                bottom = top + patch_size
                right = left + patch_size
                # 裁切图像块（numpy 切片，保持 BGR）
                patch = large_image[top:bottom, left:right, :]
                # 归一化到 [0,1] 范围，与训练时保持一致
                patch = patch.astype('float32') / 255
                # 预处理（ToTensor 支持 numpy(HWC)，BGR 顺序将被原样转换为张量的通道顺序）
                patch_tensor = x_transforms(patch).unsqueeze(0)  # 添加batch维度
                patch_tensor = patch_tensor.to(device)  # 将数据移到指定设备
                output = model(patch_tensor)
                #推理完单个patch后计数加1
                patch_count += 1
                # 发出进度信号
                if progress_signal is not None:
                    progress_signal.emit(patch_count,total_patches)
                patch_pred = output.squeeze().cpu().numpy()

                # 添加到结果
                full_pred[top:bottom, left:right] += patch_pred * weight
                weight_map[top:bottom, left:right] += weight
    # 融合结果
    full_pred /= (weight_map + 1e-8)
    # 二值化
    binary_output = np.where(full_pred > bin_thresh, 255, 0).astype(np.uint8)

    # 结束时间
    end_time = time.time()  # 记录结束时间
    inference_time = end_time - start_time  # 计算推理时间
    print(f"步长：{stride}, 推理耗时: {inference_time:.4f} 秒")


    return binary_output, inference_time


class InferenceThread(QThread):
    # 推理完成信号，返回结果（比如 mask 或概率图）
    result_ready = Signal(object, float)
    # 进度信号
    progress_changed = Signal(int,int)#当前patch计数，总patch计数
    # GPU可用状态
    cuda_available = Signal(bool)

    def __init__(self, model,model_path, image_path, stride,thresh=0.5,cuda=False):
        super().__init__()
        self.model = model
        self.model_path = model_path
        self.image_path = image_path
        self.USING_GPU = cuda
        self.stride = stride
        self.patch_size = 256
        self.use_weight_mask = True
        self.bin_thresh = thresh

    def run(self):
        # 根据传入的字符来加载不同的模型
        if self.model == 'Unet':
            self.model = Unet(3,1)
            self.model = load_model(self.model, self.model_path)
        elif self.model == 'AttentionUnet':
            self.model = AttU_Net(3,1)
            self.model = load_model(self.model, self.model_path)

        # 检测设备
        if self.USING_GPU and torch.cuda.is_available():
            self.cuda_available.emit(True)
        else:
            self.cuda_available.emit(False)

        binary_output, inference_time = process_large_image(self.model, self.image_path,
                            patch_size=256, stride=self.stride,
                            use_weight_mask=True, bin_thresh=0.5,
                            use_gpu=self.USING_GPU,progress_signal=self.progress_changed)

        # 发射信号，将结果传回主线程
        self.result_ready.emit(binary_output, inference_time)

class BatchInferenceThread(QThread):
    # 推理完成信号，返回结果、文件名、推理时间
    finished_single = Signal(object,str, float)
    # 单个图片进度信号
    single_progress = Signal(int, int)  # 当前patch计数，总patch计数
    # 总进度信号
    batch_progress = Signal(int, int)
    # GPU可用状态
    cuda_available = Signal(bool)
    # 批次完成信号
    finished_batch = Signal()

    def __init__(self, model, model_path, image_list, stride, thresh=0.5, cuda=False,save_path=None,save_bool=False,post_process=False,max_objects=96,min_area=1500,min_circularity=0.5):
        super().__init__()
        self.model = model
        self.model_path = model_path
        self.image_list = image_list
        self.USING_GPU = cuda
        self.stride = stride
        self.patch_size = 256
        self.use_weight_mask = True
        self.bin_thresh = thresh
        self.save_path = save_path
        self.save_bool = save_bool
        self.post_process = post_process
        self.max_objects = max_objects
        self.min_area = min_area
        self.min_circularity = min_circularity

    def run(self):
        # 根据传入的字符来加载不同的模型
        if self.model == 'Unet':
            self.model = Unet(3, 1)
            self.model = load_model(self.model, self.model_path)
        elif self.model == 'AttentionUnet':
            self.model = AttU_Net(3, 1)
            self.model = load_model(self.model, self.model_path)

        # 检测设备
        if self.USING_GPU and torch.cuda.is_available():
            self.cuda_available.emit(True)
        else:
            self.cuda_available.emit(False)

        # 获取总长度
        total_images = len(self.image_list)

        # 逐个处理每个图像块
        for idx ,img_path in enumerate(self.image_list):
            binary_output, inference_time = process_large_image(self.model, img_path,
                                                                patch_size=256, stride=self.stride,
                                                                use_weight_mask=True, bin_thresh=0.5,
                                                                use_gpu=self.USING_GPU,
                                                                progress_signal=self.single_progress)
            # 发射单个推理完成信号
            self.finished_single.emit( binary_output, img_path, inference_time)
            # 批次进度
            self.batch_progress.emit(idx+1, total_images)

            # 如果保存结果
            if self.save_path is not None and self.save_bool:
                if self.post_process:
                    binary_output,_ = postprocess_mask(binary_output, 96,self.min_area,self.min_circularity)
                    os.makedirs(self.save_path, exist_ok=True)
                    save_file = os.path.join(self.save_path, os.path.basename(img_path))
                    Image.fromarray(binary_output).save(save_file)
                else:
                    os.makedirs(self.save_path, exist_ok=True)
                    save_file = os.path.join(self.save_path, os.path.basename(img_path))
                    Image.fromarray(binary_output).save(save_file)


        # 发射信号，将结果传回主线程
        self.finished_batch.emit()
