import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
import torch
from torchvision import transforms
from PIL import Image

class ClassificationThread(QThread):
    """用于对 self.roi_result 中的 patch 进行阳性分类的线程"""
    finished = Signal()  # 分类完成信号
    progress = Signal(int, int)  # 当前进度 (index, total)
    # GPU可用状态
    cuda_available = Signal(bool)

    def __init__(self, roi_result, model_path, classes_json, model_name='resnet18',
                 use_gpu = False, image_size=224, confidence_threshold=0.5, parent=None):
        super().__init__(parent)
        self.roi_result = roi_result  # 字典列表，每项包含 'patch' 和 'bbox'
        self.model_path = model_path
        self.classes_json = classes_json
        self.model_name = model_name
        self.useGPU = use_gpu
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        self._is_running = True

    def stop(self):
        """停止线程"""
        self._is_running = False

    def run(self):
        import json
        import timm
        import torch.nn as nn

        # 检测设备
        if self.useGPU and torch.cuda.is_available():
            self.cuda_available.emit(True)
        else:
            self.cuda_available.emit(False)

        # 加载类别
        with open(self.classes_json, 'r', encoding='utf-8') as f:
            classes_data = json.load(f)
        self.class_names = classes_data['classes']

        # 加载模型
        device = torch.device("cuda" if self.useGPU and torch.cuda.is_available() else "cpu")
        try:
            self.model = timm.create_model(self.model_name, pretrained=False, num_classes=len(self.class_names))
        except Exception as e:
            print(f"无法创建模型 {self.model_name}: {e}")
            return

        checkpoint = torch.load(self.model_path, map_location=device)
        model_state = checkpoint.get('model_state', checkpoint)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)
        self.model.to(device)
        self.model.eval()

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        # 遍历 roi_result
        total = len(self.roi_result) #96
        for idx, item in enumerate(self.roi_result):
            if not self._is_running:
                break

            patch = item.get('patch', None)
            if patch is None:
                item['class'] = None
                continue

            # 转为 PIL Image
            if isinstance(patch, np.ndarray):
                pil_img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            else:
                pil_img = patch

            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
                confidence_score = confidence.item()
                predicted_class = self.class_names[pred_idx.item()]

            # 根据置信度阈值决定最终类别
            if confidence_score < self.confidence_threshold:
                final_class = "Below Threshold"
            else:
                final_class = predicted_class

            item['class'] = final_class
            item['confidence'] = confidence_score
            item['probabilities'] = probs.cpu().numpy()[0]

            # 发射进度信号
            self.progress.emit(idx + 1, total)

        self.finished.emit()

class BatchClassificationThread(QThread):
    """用于对 self.roi_result 中的 patch 进行阳性分类的线程"""
    finished = Signal()  # 分类完成信号
    progress = Signal(int, int)  # 当前进度 (index, total)
    # GPU可用状态
    cuda_available = Signal(bool)

    def __init__(self, roi_result, model_path, classes_json, model_name='resnet18',
                 use_gpu = False, image_size=224, confidence_threshold=0.5, parent=None):
        super().__init__(parent)
        self.roi_result = roi_result  # 字典列表，每项包含 'patch' 和 'bbox'
        self.model_path = model_path
        self.classes_json = classes_json
        self.model_name = model_name
        self.useGPU = use_gpu
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        self._is_running = True

    def stop(self):
        """停止线程"""
        self._is_running = False

    def run(self):
        import json
        import timm
        import torch.nn as nn

        # 检测设备
        if self.useGPU and torch.cuda.is_available():
            self.cuda_available.emit(True)
        else:
            self.cuda_available.emit(False)

        # 加载类别
        with open(self.classes_json, 'r', encoding='utf-8') as f:
            classes_data = json.load(f)
        self.class_names = classes_data['classes']

        # 加载模型
        device = torch.device("cuda" if self.useGPU and torch.cuda.is_available() else "cpu")
        try:
            self.model = timm.create_model(self.model_name, pretrained=False, num_classes=len(self.class_names))
        except Exception as e:
            print(f"无法创建模型 {self.model_name}: {e}")
            return

        checkpoint = torch.load(self.model_path, map_location=device)
        model_state = checkpoint.get('model_state', checkpoint)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)
        self.model.to(device)
        self.model.eval()

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        # 遍历 roi_result
        total = len(self.roi_result) #96
        for idx, item in enumerate(self.roi_result):
            if not self._is_running:
                break

            patch = item.get('patch', None)
            if patch is None:
                item['class'] = None
                continue

            # 转为 PIL Image
            if isinstance(patch, np.ndarray):
                pil_img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            else:
                pil_img = patch

            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
                confidence_score = confidence.item()
                predicted_class = self.class_names[pred_idx.item()]

            # 根据置信度阈值决定最终类别
            if confidence_score < self.confidence_threshold:
                final_class = "Below Threshold"
            else:
                final_class = predicted_class

            item['class'] = final_class
            item['confidence'] = confidence_score
            item['probabilities'] = probs.cpu().numpy()[0]

            # 发射进度信号
            self.progress.emit(idx + 1, total)

        self.finished.emit()
