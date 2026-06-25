#!/usr/bin/env bash
set -euo pipefail

mkdir -p /output /data

if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ -f /data/private_test.csv ]]; then
    export DATA_PATH=/data/private_test.csv
  elif [[ -f /data/private-test.csv ]]; then
    export DATA_PATH=/data/private-test.csv
  elif [[ -f /data/public_test.csv ]]; then
    export DATA_PATH=/data/public_test.csv
  elif [[ -f /data/public-test.csv ]]; then
    export DATA_PATH=/data/public-test.csv
  elif [[ -f /data/private_test.json ]]; then
    export DATA_PATH=/data/private_test.json
  elif [[ -f /data/private-test.json ]]; then
    export DATA_PATH=/data/private-test.json
  elif [[ -f /data/public_test.json ]]; then
    export DATA_PATH=/data/public_test.json
  elif [[ -f /data/public-test.json ]]; then
    export DATA_PATH=/data/public-test.json
  else
    echo "No test file found in /data. Expected public_test.csv/private_test.csv or .json." >&2
    exit 2
  fi
fi

export WORK_DIR="${WORK_DIR:-/output}"
export KAGGLE_INPUT_DIR="${KAGGLE_INPUT_DIR:-/data}"
export MODEL_ROOT="${MODEL_ROOT:-/models}"
export BGE_MODEL_DIR="${BGE_MODEL_DIR:-/bge/bge-m3}"
export ADAPTER_ROOT="${ADAPTER_ROOT:-/adapters}"
export PRED_PATH="${PRED_PATH:-/output/pred.csv}"
export CHECKPOINT_TO_RUN="${CHECKPOINT_TO_RUN:-ckpt12_internal_rag}"

echo "DATA_PATH=${DATA_PATH}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "BGE_MODEL_DIR=${BGE_MODEL_DIR}"
echo "ADAPTER_ROOT=${ADAPTER_ROOT}"
echo "PRED_PATH=${PRED_PATH}"

PYTHONPATH=/app/src python -m hackaithon_vllm.run

test -f "${PRED_PATH}"
python - <<'PY'
import csv
import os
from pathlib import Path

path = Path(os.environ.get("PRED_PATH", "/output/pred.csv"))
with path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("pred.csv is empty")
if set(rows[0]) != {"qid", "answer"}:
    raise SystemExit(f"pred.csv must contain exactly qid,answer columns; got {list(rows[0])}")
print({"pred_path": str(path), "rows": len(rows)})
PY
