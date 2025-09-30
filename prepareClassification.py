import os
import cv2
import numpy as np

def extract_positive_patches(mask: np.ndarray,
                             original_image: np.ndarray,
                             patch_size: int = 224,
                             save_folder: str = None,
                             expand: int = 10):
    """
    根据分割掩码提取每个连通区域的外接矩形，并调整为指定大小
    Args:
        mask: np.ndarray, 分割结果二值图 (0/255 或 0/1)
        original_image: np.ndarray, 对应的原图 (H, W, 3)
        patch_size: int, 输出patch大小，默认224
        save_folder: str, 可选，保存裁切结果的文件夹
        expand: int, 外接矩形扩展像素，避免边界太紧

    Returns:
        rois: list of dict，每个ROI包含：
            {
                "id": int, 区域编号
                "patch": np.ndarray, 裁切patch (224x224x3)
                "bbox": (x, y, w, h), 外接矩形
                "centroid": (cx, cy), 区域中心
                "area": int, 区域面积
            }
    """
    # 保证mask是二值
    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    rois = []
    H, W = original_image.shape[:2]

    for i in range(1, num_labels):  # 从1开始，跳过背景
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        if area <= 0:
            continue

        # 扩展边界，防止裁切太紧
        x1 = max(0, x - expand)
        y1 = max(0, y - expand)
        x2 = min(W, x + w + expand)
        y2 = min(H, y + h + expand)

        roi = original_image[y1:y2, x1:x2]

        # resize到目标大小
        roi_resized = cv2.resize(roi, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

        roi_info = {
            "id": i,
            "patch": roi_resized,
            "bbox": (x, y, w, h),
            "centroid": (int(cx), int(cy)),
            "area": area
        }
        rois.append(roi_info)

        # 可选：保存到文件夹
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            filename = os.path.join(save_folder, f"patch_{i}_{x}_{y}.png")
            cv2.imwrite(filename, roi_resized)

    return rois


def preprocess_for_classification(mask, original_image, patch_size=224, save_folder=None):
    """
    这是对外接口，可直接在UI中调用
    Args:
        mask: np.ndarray, 分割掩码
        original_image: np.ndarray, 原图
        patch_size: int, 输出patch大小
        save_folder: str, 可选，保存patch路径
    Returns:
        rois: list[dict]，包含patch、坐标、中心等信息
    """
    return extract_positive_patches(mask, original_image, patch_size, save_folder)


# 预留接口：导出分类结果
def export_results(image_name: str, rois: list, predictions: list, save_path: str):
    """
    导出每张图片的详细监测信息（孔位置和分类结果）
    Args:
        image_name: str, 原图文件名
        rois: list[dict], preprocess_for_classification的输出
        predictions: list[int], 分类结果 (0/1)
        save_path: str, 导出txt/CSV路径
    """
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"{os.path.splitext(image_name)[0]}_results.txt")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("id,x,y,w,h,cx,cy,area,pred\n")
        for roi, pred in zip(rois, predictions):
            x, y, w, h = roi["bbox"]
            cx, cy = roi["centroid"]
            f.write(f"{roi['id']},{x},{y},{w},{h},{cx},{cy},{roi['area']},{pred}\n")

    print(f"结果已导出到: {out_file}")


# ----------------------
# 测试
# ----------------------
if __name__ == "__main__":
    mask_path = "empty/img1.jpg"   # 二值掩码
    image_path = "example/img1.jpg"  # 原图

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)

    rois = preprocess_for_classification(mask, image, 224, "patches_output")
    print(f"共提取 {len(rois)} 个patches")

    # 模拟预测结果（0阴性, 1阳性）
    preds = [1 if roi["id"] % 2 == 0 else 0 for roi in rois]

    # 导出测试
    export_results("img1.jpg", rois, preds, "results_output")
