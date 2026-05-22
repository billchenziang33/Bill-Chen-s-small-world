import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = DATA_DIR / "object_runs" / "detect_objects" / "weights" / "best.pt"
CLASSES = ["phone", "cup", "ipad", "mouse", "paper", "tissue"]
CLASS_LABELS = {
    "phone": "手机",
    "cup": "杯子",
    "ipad": "iPad",
    "mouse": "鼠标",
    "paper": "纸",
    "tissue": "纸巾",
}


def main():
    try:
        from ultralytics import YOLO
    except Exception as error:
        raise RuntimeError(
            "ultralytics is not installed. Install it with: python -m pip install ultralytics"
        ) from error

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise RuntimeError("image does not exist")
    if not MODEL_PATH.exists():
        raise RuntimeError("YOLO object model is not trained yet.")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(str(image_path), imgsz=640, conf=0.05, verbose=False)
    detections = []
    probabilities = {class_name: 0.0 for class_name in CLASSES}

    for result in results:
        height, width = result.orig_shape
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
            confidence = round(float(box.conf[0]) * 100, 1)
            probabilities[class_name] = max(probabilities.get(class_name, 0), confidence)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "label": class_name,
                    "name": CLASS_LABELS.get(class_name, class_name),
                    "confidence": confidence,
                    "box": {
                        "x": x1 / width,
                        "y": y1 / height,
                        "width": (x2 - x1) / width,
                        "height": (y2 - y1) / height,
                    },
                }
            )

    probability_list = [
        {
            "label": class_name,
            "name": CLASS_LABELS[class_name],
            "probability": round(probabilities[class_name], 1),
        }
        for class_name in CLASSES
    ]
    probability_list.sort(key=lambda item: item["probability"], reverse=True)

    print(json.dumps({"detections": detections, "probabilities": probability_list}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        sys.exit(1)
