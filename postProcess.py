import cv2
import numpy as np

def postprocess_mask(pred_array, max_objects=96, min_area=1500, min_circularity=0.5, threshold=0.5):
    """
    后处理预测结果：去噪、筛选圆形目标，保留最多 max_objects 个结果

    Args:
        pred_array (np.ndarray): 推理输出数组，可以是概率图（0~1/0~255）或二值 mask
        max_objects (int): 最多保留目标数量
        min_area (int): 面积阈值，小于该值的对象会被去除
        min_circularity (float): 圆形度阈值 [0,1]
        threshold (float): 概率图二值化阈值，0~1
        print_stats (bool): 是否打印筛选区域信息

    Returns:
        np.ndarray: 后处理后的二值 mask (0/255)
    """

    # 归一化到 0/255 并二值化
    if pred_array.max() <= 1.0:
        mask = (pred_array * 255).astype(np.uint8)
    else:
        mask = pred_array.astype(np.uint8)

    # 如果是概率图，应用阈值
    mask = np.where(mask > int(threshold * 255), 255, 0).astype(np.uint8)

    # 形态学开运算去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity >= min_circularity:
            filtered.append((cnt, area, circularity))

    # 按面积排序，取前 max_objects 个
    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:max_objects]


    # 新建空 mask
    new_mask = np.zeros_like(mask)
    for cnt, _, _ in filtered:
        cv2.drawContours(new_mask, [cnt], -1, 255, -1)

    return new_mask,len(filtered)#返回结果和目标数量
