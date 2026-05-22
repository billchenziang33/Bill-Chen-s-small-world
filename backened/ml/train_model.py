import json
import os
import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "gesture_training.sqlite"
MODEL_JSON_PATH = DATA_DIR / "gesture_model.json"
MODEL_PT_PATH = DATA_DIR / "gesture_model.pt"
DEFAULT_EPOCHS = 60


def load_samples():
    if not DB_PATH.exists():
        raise RuntimeError("No training database found yet.")

    connection = sqlite3.connect(DB_PATH)
    rows = connection.execute(
        "SELECT digit, vector_json FROM training_samples ORDER BY created_at ASC"
    ).fetchall()
    connection.close()

    samples = []
    labels = []
    for digit, vector_json in rows:
        samples.append(json.loads(vector_json))
        labels.append(int(digit) - 1)
    return samples, labels


def main():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as error:
        raise RuntimeError(
            "PyTorch is not installed. Install it with: python -m pip install torch"
        ) from error

    samples, labels = load_samples()
    if len(samples) < 50:
        raise RuntimeError("At least 50 total samples are required before training.")

    feature_count = len(samples[0])
    class_count = 10
    x = torch.tensor(samples, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(feature_count, 64),
        nn.ReLU(),
        nn.Dropout(0.08),
        nn.Linear(64, class_count),
    )

    loader = DataLoader(TensorDataset(x, y), batch_size=48, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    epochs = int(os.environ.get("GESTURE_EPOCHS", DEFAULT_EPOCHS))
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()
        average_confidence = probabilities.max(dim=1).values.mean().item()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PT_PATH)

    layers = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            layers.append(
                {
                    "weight": layer.weight.detach().tolist(),
                    "bias": layer.bias.detach().tolist(),
                    "activation": "relu" if len(layers) < 2 else "linear",
                }
            )

    payload = {
        "type": "mlp",
        "featureCount": feature_count,
        "classes": [str(index) for index in range(1, 11)],
        "output": "softmax",
        "accuracy": round(accuracy, 4),
        "averageConfidence": round(average_confidence, 4),
        "sampleCount": len(samples),
        "epochs": epochs,
        "layers": layers,
    }
    MODEL_JSON_PATH.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({"trained": True, **payload}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        sys.exit(1)
