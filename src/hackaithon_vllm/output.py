from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path


def validate_prediction_file(path: Path, expected_qids: set[str] | None = None) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty.")
    if set(rows[0].keys()) != {"qid", "answer"}:
        raise ValueError(f"{path} must have exactly qid,answer columns; got {list(rows[0].keys())}")
    qids = [str(row["qid"]) for row in rows]
    duplicate_qids = sorted(qid for qid, count in Counter(qids).items() if count > 1)
    if duplicate_qids:
        raise ValueError(f"Duplicate qids: {duplicate_qids[:20]}")
    invalid_answers = [row for row in rows if not str(row["answer"]).strip()]
    if invalid_answers:
        raise ValueError(f"Blank answers found: {invalid_answers[:5]}")
    missing_qids = sorted(expected_qids - set(qids)) if expected_qids else []
    extra_qids = sorted(set(qids) - expected_qids) if expected_qids else []
    if missing_qids or extra_qids:
        raise ValueError({"missing_qids": missing_qids[:20], "extra_qids": extra_qids[:20]})
    return {"rows": len(rows), "answer_distribution": dict(Counter(row["answer"] for row in rows))}


def copy_prediction(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
