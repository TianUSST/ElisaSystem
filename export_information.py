import csv
import os
import numpy as np
import cv2
from datetime import datetime


def export_detailed_csv(roi_result, save_path, current_image_path=None):
    """导出详细的检测信息到CSV文件"""
    if not roi_result or not save_path:
        return

    base_name = os.path.splitext(os.path.basename(current_image_path))[0] if current_image_path else "unknown"
    
    # 查找是否已存在相同基础名称的文件
    existing_files = os.listdir(save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 1
    csv_filename = f"detailed_analysis_{base_name}_{timestamp}.csv"
    
    # 检查是否有同名文件（不包括时间戳）
    for filename in existing_files:
        if filename.startswith(f"detailed_analysis_{base_name}_") and filename.endswith(".csv"):
            # 如果存在同名文件，我们不生成新文件
            print(f"详细分析结果已存在: {filename}")
            return os.path.join(save_path, filename)

    csv_path = os.path.join(save_path, csv_filename)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['index', 'centroid_x', 'centroid_y', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
                      'class', 'confidence', 'area']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, roi in enumerate(roi_result):
            x, y, w, h = roi['bbox']
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            class_result = roi.get('class', 'unknown')
            confidence = roi.get('confidence', 0.0)
            area = 0
            if 'patch' in roi and roi['patch'] is not None:
                patch = roi['patch']
                if len(patch.shape) == 3:
                    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                else:
                    patch_gray = patch
                _, binary = cv2.threshold(patch_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                area = cv2.countNonZero(binary)
            writer.writerow({
                'index': i,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_width': w,
                'bbox_height': h,
                'class': class_result,
                'confidence': f"{confidence:.4f}",
                'area': area
            })
    print(f"详细分析结果已导出到: {csv_path}")
    return csv_path


def export_plate_format_csv(roi_result, save_path, current_image_path=None):
    """导出板式分析CSV（支持行首缺失、行内连续缺失、行尾缺失）"""
    import csv
    import os
    from datetime import datetime
    import numpy as np

    if not roi_result or not save_path:
        print("没有ROI结果可导出")
        return

    base_name = os.path.splitext(os.path.basename(current_image_path))[0] if current_image_path else "unknown"
    
    # 查找是否已存在相同基础名称的文件
    existing_files = os.listdir(save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 1
    csv_filename = f"plate_analysis_{base_name}_{timestamp}.csv"
    
    # 检查是否有同名文件（不包括时间戳）
    for filename in existing_files:
        if filename.startswith(f"plate_analysis_{base_name}_") and filename.endswith(".csv"):
            # 如果存在同名文件，我们不生成新文件
            print(f"板式分析结果已存在: {filename}")
            return os.path.join(save_path, filename)

    csv_path = os.path.join(save_path, csv_filename)

    # 计算质心
    centroids = [(roi['bbox'][0] + roi['bbox'][2] // 2,
                  roi['bbox'][1] + roi['bbox'][3] // 2,
                  roi) for roi in roi_result]

    # 按行聚类
    # rows = group_centroids_by_y(centroids)
    rows = group_centroids_into_8_rows(centroids)
    if not rows:
        print("没有分出行")
        return

    # 计算全局最左侧X，用于行首缺失判断
    global_min_x = min(c[0] for c in centroids)

    # 构建8x12板式
    plate = [['?' for _ in range(12)] for _ in range(8)]

    for row_idx, row in enumerate(rows[:8]):
        row.sort(key=lambda c: c[0])
        xs = [c[0] for c in row]

        # --- 计算行内平均X间距 ---
        if len(xs) > 1:
            diffs = np.diff(xs)
            avg_dx = np.median(diffs)
        else:
            avg_dx = 1

        # --- 行首缺失孔数 ---
        first_gap = (xs[0] - global_min_x) / avg_dx
        col_idx = max(int(round(first_gap)), 0)  # 行首偏移

        prev_x = None
        for cx, cy, roi in row:
            if prev_x is not None:
                gap = cx - prev_x
                if gap > 1.5 * avg_dx:  # 异常间距
                    missing = int(round(gap / avg_dx)) - 1
                    for _ in range(missing):
                        if col_idx < 12:
                            plate[row_idx][col_idx] = '?'
                            col_idx += 1
            # 标记当前孔
            if col_idx < 12:
                class_result = roi.get('class', 'unknown')
                conf = roi.get('confidence', 0.0)
                plate[row_idx][col_idx] = '+' if class_result == 'positive' and conf >= 0.5 else '-'
                col_idx += 1
            prev_x = cx

        # --- 行尾缺失自动补? ---
        while col_idx < 12:
            plate[row_idx][col_idx] = '?'
            col_idx += 1

    # 写入CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Row/Col'] + [str(i + 1) for i in range(12)]
        writer.writerow(header)
        for i, row in enumerate(plate):
            writer.writerow([chr(65 + i)] + row)

    print(f"板式分析结果已导出到: {csv_path}")
    return csv_path


def group_centroids_into_8_rows(centroids):
    """确保聚类为正好8行"""
    if not centroids:
        return []

    # 按y坐标排序
    sorted_centroids = sorted(centroids, key=lambda c: c[1])
    y_coords = [c[1] for c in sorted_centroids]

    # 使用K-means聚类确保生成8行
    from sklearn.cluster import KMeans
    y_array = np.array(y_coords).reshape(-1, 1)
    kmeans = KMeans(n_clusters=8, random_state=42)
    labels = kmeans.fit_predict(y_array)

    # 按聚类结果分组
    rows = [[] for _ in range(8)]
    for i, label in enumerate(labels):
        rows[label].append(sorted_centroids[i])

    # 按行的y坐标排序
    row_means = [np.mean([c[1] for c in row]) if row else 0 for row in rows]
    rows = [row for _, row in sorted(zip(row_means, rows))]

    return rows