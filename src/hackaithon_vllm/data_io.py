from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def find_data_path(input_dir: Path = Path("/data")) -> Path:
    patterns = (
        "private_test.csv",
        "private-test.csv",
        "public_test.csv",
        "public-test.csv",
        "private_test.json",
        "private-test.json",
        "public_test.json",
        "public-test.json",
        "**/private-test*.json",
        "**/private_test*.json",
        "**/public-test*.json",
        "**/public_test*.json",
        "**/private-test*.csv",
        "**/public-test*.csv",
    )
    for pattern in patterns:
        hits = sorted(input_dir.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No public/private test file found under {input_dir}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError("JSON test file must contain a list of rows.")
        return rows

    if suffix == ".csv":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row.get("qid") or row.get("id") or row.get("question_id")
                question = row.get("question") or row.get("prompt") or row.get("content")
                choices = []
                for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    value = (
                        row.get(label)
                        or row.get(label.lower())
                        or row.get(f"choice_{label}")
                        or row.get(f"option_{label}")
                    )
                    if value not in (None, ""):
                        choices.append(value)
                if not choices and row.get("choices"):
                    try:
                        parsed = json.loads(row["choices"])
                        if isinstance(parsed, list):
                            choices = parsed
                    except Exception:
                        choices = [x.strip() for x in str(row["choices"]).split("||") if x.strip()]
                if not qid or not question or not choices:
                    raise ValueError(f"Cannot parse CSV row into qid/question/choices: {row}")
                rows.append({"qid": qid, "question": question, "choices": choices})
        return rows

    raise ValueError(f"Unsupported data file type: {path}")
