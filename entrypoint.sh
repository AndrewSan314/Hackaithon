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

BUILTIN_RAG_ZIP="${BUILTIN_RAG_ZIP:-/app/outputs/rag_vector_db_final.zip}"
if [[ -z "${LAW_ADMIN_VECTOR_DB_DIR:-}" && ! -f /rag/rag_vector_db_final/metadata.json && -f "${BUILTIN_RAG_ZIP}" ]]; then
  export LAW_ADMIN_VECTOR_DB_DIR="${BUILTIN_RAG_DIR:-/tmp/hackaithon_rag/rag_vector_db_final}"
  if [[ ! -f "${LAW_ADMIN_VECTOR_DB_DIR}/metadata.json" ]]; then
    mkdir -p "$(dirname "${LAW_ADMIN_VECTOR_DB_DIR}")"
    python - "${BUILTIN_RAG_ZIP}" "$(dirname "${LAW_ADMIN_VECTOR_DB_DIR}")" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile

zip_path = Path(sys.argv[1])
out_parent = Path(sys.argv[2])
with ZipFile(zip_path) as zf:
    zf.extractall(out_parent)
rag_dir = out_parent / "rag_vector_db_final"
if not (rag_dir / "metadata.json").exists():
    raise SystemExit(f"Built-in RAG zip did not extract metadata.json into {rag_dir}")
print({"rag_dir": str(rag_dir), "source_zip": str(zip_path)})
PY
  fi
else
  export LAW_ADMIN_VECTOR_DB_DIR="${LAW_ADMIN_VECTOR_DB_DIR:-/rag/rag_vector_db_final}"
fi

BUILTIN_ADAPTER_ZIP="${BUILTIN_ADAPTER_ZIP:-/app/outputs/qwen35_qlora_mcq_mixed_resume_noeval.zip}"
if [[ -z "${ADAPTER_DIR:-}" && -f "${BUILTIN_ADAPTER_ZIP}" ]]; then
  export ADAPTER_ROOT="${BUILTIN_ADAPTER_ROOT:-/tmp/hackaithon_adapters}"
  export ADAPTER_DIR="${ADAPTER_ROOT}/qwen35_qlora_mcq_mixed_resume_noeval"
  if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
    mkdir -p "${ADAPTER_DIR}"
    python - "${BUILTIN_ADAPTER_ZIP}" "${ADAPTER_DIR}" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
if not (out_dir / "adapter_config.json").exists():
    raise SystemExit(f"Built-in adapter zip did not extract adapter_config.json into {out_dir}")
print({"adapter_dir": str(out_dir), "source_zip": str(zip_path)})
PY
  fi
fi

echo "DATA_PATH=${DATA_PATH}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "BGE_MODEL_DIR=${BGE_MODEL_DIR}"
echo "ADAPTER_ROOT=${ADAPTER_ROOT}"
echo "ADAPTER_DIR=${ADAPTER_DIR:-auto}"
echo "LAW_ADMIN_VECTOR_DB_DIR=${LAW_ADMIN_VECTOR_DB_DIR}"
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
