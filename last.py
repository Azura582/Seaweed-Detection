import os
import cv2
import numpy as np
import math
from ultralytics import YOLO
from pathlib import Path

# 权重
MODEL_PATH = 'best.pt'
# 测试图片路径
INPUT_DIR = 'images'
# 保存路径
OUTPUT_DIR = 'out/'

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_closest_edge_point(point, width, height):

    x, y = point

    dists = [x, width - x, y, height - y]
    min_dist = min(dists)

    edge_point = (0, 0)

    if min_dist == x:
        edge_point = (0, y)
    elif min_dist == width - x:
        edge_point = (width, y)
    elif min_dist == y:
        edge_point = (x, 0)
    else:
        edge_point = (x, height)

    return min_dist, edge_point


def plan_shortest_path(points, width, height):

    if not points:
        return [], 0

    best_total_path = []
    min_total_distance = float('inf')


    for start_node_idx in range(len(points)):

        first_point = points[start_node_idx]

        dist_to_wall, wall_start_point = get_closest_edge_point(first_point, width, height)

        current_path = [wall_start_point, first_point]
        current_distance = dist_to_wall
        unvisited = points[:start_node_idx] + points[start_node_idx + 1:]
        current_pos = first_point

        while unvisited:
            nearest_idx = -1
            min_d = float('inf')

            for i, p in enumerate(unvisited):
                d = calculate_distance(current_pos, p)
                if d < min_d:
                    min_d = d
                    nearest_idx = i

            next_point = unvisited[nearest_idx]
            current_distance += min_d
            current_path.append(next_point)
            current_pos = next_point

            unvisited.pop(nearest_idx)

            if current_distance > min_total_distance:
                break

        if current_distance < min_total_distance and len(current_path) == len(points) + 1:
            min_total_distance = current_distance
            best_total_path = current_path

    return best_total_path, min_total_distance



def process_images():

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

  
    print(f"正在加载模型: {MODEL_PATH} ...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"错误: 无法加载模型，请检查路径。详细信息: {e}")
        return

    # 3. 遍历图片目录
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not image_files:
        print(f"在 {INPUT_DIR} 中未找到图片。")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)

        # 读取图片
        original_img = cv2.imread(img_path)
        if original_img is None:
            continue

        h, w = original_img.shape[:2]

        # --- A. 推理 ---
        results = model.predict(img_path, conf=0.25, verbose=False)
        result = results[0]

        # 提取中心点
        points = []
        if result.boxes:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                points.append((cx, cy))

        # --- B. 生成掩码 (Mask) ---
        mask = np.zeros((h, w), dtype=np.uint8)
        for pt in points:
            # 半径设为5，白色实心
            cv2.circle(mask, pt, 5, 255, -1)

        # --- C. 路径规划 ---
        path_img = original_img.copy()

        if len(points) > 0:
            # 计算路径
            shortest_path, total_dist = plan_shortest_path(points, w, h)

            # 绘制路径
            # 1. 绘制连线
            for i in range(len(shortest_path) - 1):
                pt1 = shortest_path[i]
                pt2 = shortest_path[i + 1]
                # 绿色线条，宽度2
                cv2.line(path_img, pt1, pt2, (0, 255, 0), 2)

            # 2. 绘制关键点
            # 起点 (边界上的点) - 蓝色大圆点
            if shortest_path:
                cv2.circle(path_img, shortest_path[0], 10, (255, 0, 0), -1)
                cv2.putText(path_img, "Start", (shortest_path[0][0] + 10, shortest_path[0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 水藻目标点 - 红色圆点，并标序号
            for i, pt in enumerate(shortest_path[1:]):  # 跳过第一个点(因为是墙壁点)
                cv2.circle(path_img, pt, 5, (0, 0, 255), -1)
                # 标出访问顺序
                cv2.putText(path_img, str(i + 1), (pt[0] - 5, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        else:
            print(f"图片 {img_name} 未检测到目标。")

        # --- D. 保存结果 ---
        # 保存掩码图
        mask_filename = f"mask_{os.path.splitext(img_name)[0]}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, mask_filename), mask)

        # 保存路径规划图
        path_filename = f"path_{os.path.splitext(img_name)[0]}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, path_filename), path_img)

        print(f"已处理: {img_name} -> 保存至 {OUTPUT_DIR}")

    print("所有任务完成！")


if __name__ == "__main__":
    process_images()