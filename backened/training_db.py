import json
import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "gesture_training.sqlite"


def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS training_samples (
            id TEXT PRIMARY KEY,
            digit TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return connection


def get_summary(connection):
    rows = connection.execute(
        "SELECT digit, COUNT(*) AS count FROM training_samples GROUP BY digit"
    ).fetchall()
    counts = {str(index): 0 for index in range(1, 11)}
    for row in rows:
        counts[str(row["digit"])] = row["count"]
    return counts


def list_samples(connection):
    rows = connection.execute(
        "SELECT id, digit, vector_json, created_at FROM training_samples ORDER BY created_at ASC"
    ).fetchall()
    samples = []
    for row in rows:
        samples.append(
            {
                "id": row["id"],
                "digit": row["digit"],
                "vector": json.loads(row["vector_json"]),
                "createdAt": row["created_at"],
            }
        )
    return {"items": samples, "summary": get_summary(connection)}


def add_sample(connection, payload):
    sample_id = str(payload.get("id") or "")
    digit = str(payload.get("digit") or "")
    vector = payload.get("vector")

    if not sample_id:
        raise ValueError("id is required")
    if digit not in {str(index) for index in range(1, 11)}:
        raise ValueError("digit must be 1-10")
    if not isinstance(vector, list) or not vector:
        raise ValueError("vector must be a non-empty list")

    connection.execute(
        "INSERT OR REPLACE INTO training_samples (id, digit, vector_json) VALUES (?, ?, ?)",
        (sample_id, digit, json.dumps(vector)),
    )
    connection.commit()
    return {"saved": True, "summary": get_summary(connection)}


def clear_samples(connection, payload):
    digit = payload.get("digit")
    if digit:
        connection.execute("DELETE FROM training_samples WHERE digit = ?", (str(digit),))
    else:
        connection.execute("DELETE FROM training_samples")
    connection.commit()
    return {"cleared": True, "summary": get_summary(connection)}


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "list"
    payload = {}
    if len(sys.argv) > 2:
        payload = json.loads(sys.argv[2])

    with connect() as connection:
        if command == "list":
            result = list_samples(connection)
        elif command == "add":
            result = add_sample(connection, payload)
        elif command == "clear":
            result = clear_samples(connection, payload)
        elif command == "summary":
            result = {"summary": get_summary(connection)}
        else:
            raise ValueError(f"Unknown command: {command}")

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        sys.exit(1)
