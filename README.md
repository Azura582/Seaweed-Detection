# Seaweed Detection Web

这是一个基于 Flask 的海藻检测与路径规划演示。上传 1-9 张图片后，系统会在原图上绘制检测框、中心点和最短路径，并返回结果图片。

## 功能
- YOLO 检测海藻目标
- 在原图上绘制检测框与中心点
- 计算并绘制最短访问路径（从最近边界进入）
- 批量上传 1-9 张图片

## 目录结构
- `app.py`：Flask 后端入口（合并检测 + 绘制逻辑）
- `path_planning.py`：最短路径规划算法
- `templates/index.html`：前端页面
- `static/css/style.css`：页面样式

## 运行
1. 安装依赖：
   - `pip install -r requirements.txt`
2. 启动服务：
   - `python app.py`
3. 浏览器访问：
   - `http://localhost:5000`

## 注意事项
- 权重文件默认读取 `best.pt`，请确保在项目根目录。
- 默认使用 CPU 推理（避免 CUDA 兼容问题）。
- 结果图片保存在 `static/results/<批次ID>/`。
