import os
import cv2
import numpy as np

def extract_positive_patches(mask: np.ndarray,
                             original_image: np.ndarray,
                             patch_size: int = 224,
                             save_folder: str = None):
    """
    根据分割掩码提取每个连通区域的外接矩形，并调整为指定大小
    Args:
        mask: np.ndarray, 分割结果二值图 (0/255 或 0/1)
        original_image: np.ndarray, 对应的原图 (H, W, 3)
        patch_size: int, 输出patch大小，默认224
        save_folder: str, 可选，保存裁切结果的文件夹

    Returns:
        patches: list of np.ndarray, 每个裁切后的区域 (patch_size, patch_size, 3)
        bboxes: list of tuple, 每个区域在原图上的坐标 (x, y, w, h)
    """
    # 1. 找到连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    patches = []
    bboxes = []

    # 从1开始，0是背景
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area <= 0:
            continue

        # 2. 裁切原图对应区域
        roi = original_image[y:y+h+20, x:x+w+20]

        # 3. 调整大小为patch_size x patch_size
        roi_resized = cv2.resize(roi, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

        patches.append(roi_resized)
        bboxes.append((x, y, w, h))

        # 4. 可选：保存到文件夹
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            filename = os.path.join(save_folder, f"patch_{x}_{y}.png")
            cv2.imwrite(filename, roi_resized)

    return patches, bboxes

# ----------------------
# 示例接口函数
# ----------------------
def preprocess_for_classification(mask, original_image, patch_size=224, save_folder=None):
    """
    这是对外接口，可直接在UI中调用
    """
    patches, bboxes = extract_positive_patches(mask, original_image, patch_size, save_folder)
    return patches, bboxes

# ----------------------
# 测试
# ----------------------
if __name__ == "__main__":
    mask_path = "empty/img1.jpg"
    image_path = "example/img1.jpg"

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)

    patches, bboxes = preprocess_for_classification(mask, image, 224, "patches_output")
    print(f"共提取 {len(patches)} 个patches")
