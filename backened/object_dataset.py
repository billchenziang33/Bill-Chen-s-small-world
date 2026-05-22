import base64
import json
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OBJECT_DIR = DATA_DIR / "object_dataset"
IMAGES_DIR = OBJECT_DIR / "images" / "train"
LABELS_DIR = OBJECT_DIR / "labels" / "train"
DATA_YAML = OBJECT_DIR / "data.yaml"
METADATA_FILE = OBJECT_DIR / "metadata.json"

CLASSES = ["phone", "cup", "ipad", "mouse", "paper", "tissue"]
CLASS_LABELS = {
    "phone": "手机",
    "cup": "杯子",
    "ipad": "iPad",
    "mouse": "鼠标",
    "paper": "纸",
    "tissue": "纸巾",
}


def ensure_dataset():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text(
        "\n".join(
            [
                f"path: {OBJECT_DIR.as_posix()}",
                "train: images/train",
                "val: images/train",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    if not METADATA_FILE.exists():
        METADATA_FILE.write_text("[]", encoding="utf-8")


def read_metadata():
    ensure_dataset()
    try:
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def write_metadata(items):
    METADATA_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def decode_data_url(data_url):
    if "," not in data_url:
        raise ValueError("image must be a data URL")
    return base64.b64decode(data_url.split(",", 1)[1])


def normalize_box(box):
    x = float(box.get("x", 0.2))
    y = float(box.get("y", 0.18))
    width = float(box.get("width", 0.6))
    height = float(box.get("height", 0.64))
    x = max(0.0, min(0.99, x))
    y = max(0.0, min(0.99, y))
    width = max(0.04, min(1.0 - x, width))
    height = max(0.04, min(1.0 - y, height))
    x_center = max(0.0, min(1.0, x + width / 2))
    y_center = max(0.0, min(1.0, y + height / 2))
    return x_center, y_center, width, height, x, y


def add_sample(payload):
    ensure_dataset()
    label = payload.get("label")
    image = payload.get("image")
    box = payload.get("box") or {}
    note = str(payload.get("note") or "").strip()

    if label not in CLASSES:
        raise ValueError("label is not supported")
    if not image:
        raise ValueError("image is required")

    x_center, y_center, width, height, x, y = normalize_box(box)
    sample_id = f"{label}_{int(time.time() * 1000)}"
    image_path = IMAGES_DIR / f"{sample_id}.jpg"
    label_path = LABELS_DIR / f"{sample_id}.txt"
    image_path.write_bytes(decode_data_url(image))
    label_path.write_text(
        f"{CLASSES.index(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
        encoding="utf-8",
    )

    metadata = read_metadata()
    metadata.append(
        {
            "id": sample_id,
            "label": label,
            "name": CLASS_LABELS[label],
            "note": note,
            "image": str(image_path.relative_to(OBJECT_DIR)),
            "box": {"x": x, "y": y, "width": width, "height": height},
            "createdAt": int(time.time() * 1000),
        }
    )
    write_metadata(metadata)
    return {"saved": True, "sampleId": sample_id, "summary": get_summary()}


def get_summary():
    ensure_dataset()
    counts = {name: 0 for name in CLASSES}
    for label_file in LABELS_DIR.glob("*.txt"):
        content = label_file.read_text(encoding="utf-8").strip().split()
        if content:
            class_id = int(content[0])
            if 0 <= class_id < len(CLASSES):
                counts[CLASSES[class_id]] += 1
    recent = sorted(read_metadata(), key=lambda item: item.get("createdAt", 0), reverse=True)[:12]
    return {"counts": counts, "labels": CLASS_LABELS, "classes": CLASSES, "recent": recent}


def clear_samples(payload):
    ensure_dataset()
    label = payload.get("label")
    labels_to_clear = [label] if label in CLASSES else CLASSES
    metadata = read_metadata()
    metadata = [item for item in metadata if item.get("label") not in labels_to_clear]
    write_metadata(metadata)

    for target_label in labels_to_clear:
        prefix = f"{target_label}_"
        for file_path in IMAGES_DIR.glob(f"{prefix}*.jpg"):
            file_path.unlink(missing_ok=True)
        for file_path in LABELS_DIR.glob(f"{prefix}*.txt"):
            file_path.unlink(missing_ok=True)
    return {"cleared": True, "summary": get_summary()}


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if len(sys.argv) > 2 and sys.argv[2] == "-":
        payload = json.loads(sys.stdin.read() or "{}")
    else:
        payload = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

    if command == "add":
        result = add_sample(payload)
    elif command == "clear":
        result = clear_samples(payload)
    elif command == "summary":
        result = get_summary()
    else:
        raise ValueError(f"Unknown command: {command}")

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        sys.exit(1)
