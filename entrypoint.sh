#!/usr/bin/env bash
set -euo pipefail

mkdir -p /output /data /code

if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ -f /code/private_test.json ]]; then
    export DATA_PATH=/code/private_test.json
  elif [[ -f /code/private_test.csv ]]; then
    export DATA_PATH=/code/private_test.csv
  elif [[ -f /app/data/private_test.json ]]; then
    export DATA_PATH=/app/data/private_test.json
  elif [[ -f /app/data/private_test.csv ]]; then
    export DATA_PATH=/app/data/private_test.csv
  elif [[ -f /data/private_test.csv ]]; then
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
    echo "No test file found. Expected /code/private_test.json or public/private CSV/JSON under /data." >&2
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
export MODEL_SELECT="${MODEL_SELECT:-4b}"

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

BUILTIN_ADAPTER_ZIP="${BUILTIN_ADAPTER_ZIP:-/app/outputs/qwen35_4b_qlora_mcq_mixed.zip}"
if [[ -z "${ADAPTER_DIR:-}" && -f "${BUILTIN_ADAPTER_ZIP}" ]]; then
  export ADAPTER_ROOT="${BUILTIN_ADAPTER_ROOT:-/tmp/hackaithon_adapters}"
  export ADAPTER_DIR="${ADAPTER_ROOT}/qwen35_4b_qlora_mcq_mixed"
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
elif [[ -z "${ADAPTER_DIR:-}" && -f /app/outputs/qwen35_qlora_mcq_mixed_resume_noeval.zip ]]; then
  echo "Found legacy 9B adapter zip but MODEL_SELECT defaults to 4b; not loading incompatible adapter." >&2
  echo "Provide a Qwen3.5-4B adapter via ADAPTER_DIR or /app/outputs/qwen35_4b_qlora_mcq_mixed.zip." >&2
fi

echo "DATA_PATH=${DATA_PATH}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "MODEL_SELECT=${MODEL_SELECT}"
echo "BGE_MODEL_DIR=${BGE_MODEL_DIR}"
echo "ADAPTER_ROOT=${ADAPTER_ROOT}"
echo "ADAPTER_DIR=${ADAPTER_DIR:-auto}"
echo "LAW_ADMIN_VECTOR_DB_DIR=${LAW_ADMIN_VECTOR_DB_DIR}"
echo "PRED_PATH=${PRED_PATH}"

PYTHONPATH=/app/src python -m hackaithon_vllm.run

if [[ ! -f "${PRED_PATH}" ]]; then
  echo "Expected prediction file was not created: ${PRED_PATH}" >&2
  exit 3
fi

if [[ ! -f "/output/submission.csv" || ! -f "/output/submission_time.csv" ]]; then
  echo "Expected submission.csv and submission_time.csv aliases were not created in /output." >&2
  exit 3
fi

if [[ -w /code && ( ! -f "/code/submission.csv" || ! -f "/code/submission_time.csv" ) ]]; then
  echo "Expected submission.csv and submission_time.csv aliases were not created in writable /code." >&2
  exit 3
fi

python - <<'PY'
import csv
import os
from pathlib import Path

path = Path(os.environ.get("PRED_PATH", "/output/pred.csv"))
required_files = [path, Path("/output/submission.csv"), Path("/output/submission_time.csv")]
if Path("/code/submission.csv").exists() and Path("/code/submission_time.csv").exists():
    required_files.extend([Path("/code/submission.csv"), Path("/code/submission_time.csv")])
for required in required_files:
    with required.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{required} is empty")
    expected = {"qid", "answer", "time"} if required.name == "submission_time.csv" else {"qid", "answer"}
    if set(rows[0]) != expected:
        raise SystemExit(f"{required} must contain exactly {sorted(expected)} columns; got {list(rows[0])}")
print({"pred_path": str(path), "rows": len(rows), "submission": "/output/submission.csv", "submission_time": "/output/submission_time.csv"})
PY
