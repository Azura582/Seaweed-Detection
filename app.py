import os
import uuid
from datetime import datetime
from typing import List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flask import Flask, render_template, request
import cv2
import numpy as np
from ultralytics import YOLO

from path_planning import plan_shortest_path

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(APP_ROOT, "best.pt")
UPLOAD_DIR = os.path.join(APP_ROOT, "static", "uploads")
RESULT_DIR = os.path.join(APP_ROOT, "static", "results")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILES = 9

MODEL = None


def ensure_dirs() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def get_model() -> YOLO:
    global MODEL
    if MODEL is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"模型文件不存在: {WEIGHTS_PATH}")
        MODEL = YOLO(WEIGHTS_PATH).to("cpu")
    return MODEL


def draw_detections_and_path(image: np.ndarray, boxes: List[Tuple[float, float, float, float]]) -> np.ndarray:
    output = image.copy()
    points: List[Tuple[int, int]] = []

    for x1, y1, x2, y2 in boxes:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(output, p1, p2, (0, 255, 0), 2)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        points.append((cx, cy))
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

    if points:
        h, w = output.shape[:2]
        shortest_path, _ = plan_shortest_path(points, w, h)

        for i in range(len(shortest_path) - 1):
            pt1 = shortest_path[i]
            pt2 = shortest_path[i + 1]
            cv2.line(output, pt1, pt2, (0, 255, 255), 2)

        if shortest_path:
            cv2.circle(output, shortest_path[0], 8, (255, 0, 0), -1)
            cv2.putText(
                output,
                "Start",
                (shortest_path[0][0] + 8, shortest_path[0][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        for i, pt in enumerate(shortest_path[1:]):
            cv2.putText(
                output,
                str(i + 1),
                (pt[0] + 4, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

    return output


def process_image(image_path: str, output_path: str) -> int:
    model = get_model()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图片")

    results = model.predict(image_path, conf=0.25, imgsz=256, verbose=False, device="cpu")
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        boxes = []
    else:
        boxes = result.boxes.xyxy.cpu().numpy().tolist()

    output_img = draw_detections_and_path(image, boxes)
    cv2.imwrite(output_path, output_img)

    return len(boxes)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


@app.route("/", methods=["GET", "POST"])
def index():
    ensure_dirs()

    if request.method == "POST":
        files = request.files.getlist("images")
        files = [f for f in files if f and f.filename]

        if not files:
            return render_template("index.html", error="请至少上传 1 张图片。")

        if len(files) > MAX_FILES:
            return render_template("index.html", error="最多只能上传 9 张图片。")

        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        batch_upload_dir = os.path.join(UPLOAD_DIR, batch_id)
        batch_result_dir = os.path.join(RESULT_DIR, batch_id)
        os.makedirs(batch_upload_dir, exist_ok=True)
        os.makedirs(batch_result_dir, exist_ok=True)

        results_payload = []
        for file in files:
            if not allowed_file(file.filename):
                return render_template("index.html", error=f"不支持的文件格式: {file.filename}")

            safe_name = os.path.basename(file.filename)
            upload_path = os.path.join(batch_upload_dir, safe_name)
            result_path = os.path.join(batch_result_dir, safe_name)

            file.save(upload_path)

            try:
                count = process_image(upload_path, result_path)
            except Exception as exc:
                return render_template("index.html", error=f"处理 {safe_name} 失败: {exc}")

            results_payload.append(
                {
                    "filename": safe_name,
                    "detections": count,
                    "url": f"/static/results/{batch_id}/{safe_name}",
                }
            )

        return render_template("index.html", results=results_payload, batch_id=batch_id)

    return render_template("index.html")


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=5000, debug=True)
