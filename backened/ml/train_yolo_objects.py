import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_YAML = DATA_DIR / "object_dataset" / "data.yaml"
LABELS_DIR = DATA_DIR / "object_dataset" / "labels" / "train"
RUNS_DIR = DATA_DIR / "object_runs"
DEFAULT_EPOCHS = 5


def main():
    try:
        from ultralytics import YOLO
    except Exception as error:
        raise RuntimeError(
            "ultralytics is not installed. Install it with: python -m pip install ultralytics"
        ) from error

    if not DATA_YAML.exists():
        raise RuntimeError("No YOLO object dataset found yet.")
    sample_count = len(list(LABELS_DIR.glob("*.txt"))) if LABELS_DIR.exists() else 0
    if sample_count < 6:
        raise RuntimeError(
            f"Not enough object samples to train YOLO yet. Saved samples: {sample_count}. "
            "Please collect several samples for each object class first."
        )

    model = YOLO("yolov8n.pt")
    epochs = int(os.environ.get("YOLO_EPOCHS", DEFAULT_EPOCHS))
    result = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=640,
        batch=8,
        project=str(RUNS_DIR),
        name="detect_objects",
        exist_ok=True,
        verbose=False,
    )
    best_path = RUNS_DIR / "detect_objects" / "weights" / "best.pt"
    print(json.dumps({"trained": True, "epochs": epochs, "model": str(best_path)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        sys.exit(1)
