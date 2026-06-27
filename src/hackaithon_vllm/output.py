from __future__ import annotations

import csv
import shutil
import warnings
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


def _runtime_by_qid_from_log(log_path: Path | None) -> dict[str, float]:
    if log_path is None or not Path(log_path).exists():
        return {}
    runtimes: dict[str, float] = {}
    with Path(log_path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("qid", ""))
            if not qid:
                continue
            try:
                runtimes[qid] = max(float(row.get("runtime_sec", 0.0)), 0.0)
            except (TypeError, ValueError):
                runtimes[qid] = 0.0
    return runtimes


def write_submission_aliases(
    pred_path: Path,
    elapsed_seconds: float | None = None,
    log_path: Path | None = None,
) -> list[Path]:
    """Write BTC-compatible aliases next to pred.csv and under /code when available."""
    pred_path = Path(pred_path)
    with pred_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{pred_path} is empty.")

    output_dirs = [pred_path.parent]
    code_dir = Path("/code")
    if code_dir.exists() and code_dir not in output_dirs:
        output_dirs.append(code_dir)
    output_dir = Path("/output")
    if output_dir.exists() and output_dir not in output_dirs:
        output_dirs.append(output_dir)

    runtime_by_qid = _runtime_by_qid_from_log(log_path)
    avg_time = 0.0
    if elapsed_seconds is not None and rows:
        avg_time = max(float(elapsed_seconds), 0.0) / len(rows)

    written: list[Path] = []
    for out_dir in output_dirs:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if out_dir == code_dir:
                warnings.warn(f"Could not create optional /code submission aliases: {exc}", RuntimeWarning)
                continue
            raise
        submission_path = out_dir / "submission.csv"
        time_path = out_dir / "submission_time.csv"

        try:
            with submission_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({"qid": row["qid"], "answer": row["answer"]})

            with time_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["qid", "answer", "time"])
                writer.writeheader()
                for row in rows:
                    runtime_sec = runtime_by_qid.get(str(row["qid"]), avg_time)
                    writer.writerow({"qid": row["qid"], "answer": row["answer"], "time": f"{runtime_sec:.6f}"})
        except OSError as exc:
            if out_dir == code_dir:
                warnings.warn(f"Could not write optional /code submission aliases: {exc}", RuntimeWarning)
                continue
            raise

        written.extend([submission_path, time_path])

    return written
