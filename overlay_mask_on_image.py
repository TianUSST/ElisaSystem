import cv2
import numpy as np
from PySide6.QtGui import QPixmap, QImage


def overlay_mask_on_image(image, mask, random_seed=42):
    """
    在原图上叠加掩码，每个连通区域用不同颜色。

    Args:
        image (np.ndarray): 原图，BGR
        mask (np.ndarray): 二值掩码，0/255
    Returns:
        overlaid_image (np.ndarray): 叠加后的彩色图
        num_regions (int): 连通区域数量
    """
    overlaid = image.copy()
    if mask.max() == 1:
        mask = (mask * 255).astype(np.uint8)

    # 查找连通区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_regions = len(contours)

    # 设置随机颜色
    rng = np.random.default_rng(random_seed)
    colors = rng.integers(0, 256, size=(num_regions, 3), dtype=np.uint8)

    # 绘制每个连通区域
    for i, cnt in enumerate(contours):
        cv2.drawContours(overlaid, [cnt], -1, color=colors[i].tolist(), thickness=2)

    return overlaid, num_regions

def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    """BGR -> RGB -> QPixmap"""
    if len(img.shape) == 2:
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_Grayscale8)
    else:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], rgb_img.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)