from ultralytics import YOLO
import os


def main():
    # ================= 配置区域 =================
    # 1. 模型路径
    weights_path = 'best.pt'

    # 2. 测试集图片文件夹路径
    # 根据你之前的描述，转换后的数据集在 data/data/1，所以 test 图片在这里：
    source_path = 'images'

    # 3. 结果保存的根目录
    save_root = 'out/'
    # 子文件夹名字 (最终图片会保存在 save_root/predictions 里面)
    sub_name = 'pr'
    # ===========================================

    # 检查路径
    if not os.path.exists(weights_path):
        print(f"错误: 模型文件不存在 {weights_path}")
        return
    if not os.path.exists(source_path):
        print(f"错误: 图片路径不存在 {source_path}")
        return

    print(f"加载模型: {weights_path}")
    model = YOLO(weights_path).to("cpu")

    print(f"开始处理图片，来源: {source_path}")
    print(f"这是一个耗时操作，取决于测试集图片数量...")

    # 执行预测
    # stream=False: 一次性处理所有并保存（如果内存不够，可以设为True并手动迭代，但这里一般没问题）
    results = model.predict(
        source=source_path,
        save=True,  # 必须为 True 才能保存画框图片
        imgsz=256,  # 保持和训练一致
        conf=0.25,  # 置信度阈值 (低于这个分数的框不画)
        iou=0.45,  # NMS 阈值 (去重)
        project=save_root,  # 保存的根路径
        name=sub_name,  # 保存的子文件夹名
        exist_ok=True,  # 如果文件夹存在，覆盖写入，不创建 predictions2, predictions3...

        # === 可视化微调 ===
        line_width=1,  # 线条宽度：SAR图像较小，建议设细一点 (1 或 2)
        show_labels=True,  # 是否显示类别名
        show_conf=True,  # 是否显示置信度
        # max_det=300,          # 每张图最大检测数量
    )

    full_save_path = os.path.join(save_root, sub_name)
    print("\n" + "=" * 40)
    print("全部完成！")
    print(f"画好框的图片已保存在: {full_save_path}")
    print("=" * 40)


if __name__ == '__main__':
    main()