# Hackaithon MCQ vLLM Submission

This repository root is the reproducible inference package derived from
`hackaithon-vllm (1).ipynb`. It runs the production checkpoint
`ckpt12_internal_rag` only and writes the required submission file
`/output/pred.csv` with columns `qid,answer`.

## Contents

- `Dockerfile`: container definition for Docker Hub.
- `entrypoint.sh`: reads `/data/private_test.csv` or `/data/public_test.csv` and writes `/output/pred.csv`.
- `src/hackaithon-vllm.ipynb`: thin notebook runner for the Python package.
- `src/hackaithon_vllm_pipeline.py`: compatibility wrapper.
- `src/hackaithon_vllm/`: refactored Python package; runtime supports `ckpt12_internal_rag` only.
- `outputs/pred.csv`: latest public-test prediction snapshot.
- `outputs/rag_vector_db_final.zip`: final Law/Admin RAG vector DB artifact.
- `outputs/qwen35_qlora_mcq_mixed_resume_noeval.zip`: production QLoRA adapter artifact.
- `docs/method_report.tex`: LaTeX method document.

## Expected Runtime Mounts

The image does not bake in the 9B model or BGE model. Mount them:

- `/models`: Qwen3.5-9B local Hugging Face model folder, or a parent containing it.
- `/bge/bge-m3`: BGE-M3 local Hugging Face model folder.
- `/data`: contains `private_test.csv` or `public_test.csv`.
- `/output`: output directory for `pred.csv`.

The QLoRA adapter is included in this repository as
`outputs/qwen35_qlora_mcq_mixed_resume_noeval.zip`. During container startup,
`entrypoint.sh` automatically extracts it to
`/tmp/hackaithon_adapters/qwen35_qlora_mcq_mixed_resume_noeval` and sets
`ADAPTER_DIR` when no adapter is explicitly provided. You can still override it
by mounting your own adapter and setting `ADAPTER_DIR` or `ADAPTER_ROOT`.

The final Law/Admin RAG DB is included as `outputs/rag_vector_db_final.zip`.
During container startup, `entrypoint.sh` automatically extracts it to
`/tmp/hackaithon_rag/rag_vector_db_final` and sets `LAW_ADMIN_VECTOR_DB_DIR`
when `/rag/rag_vector_db_final` is not mounted. You can still override it by
mounting your own RAG DB and setting `LAW_ADMIN_VECTOR_DB_DIR`.

## Build

```bash
docker build -t hackaithon-vllm-submission:latest .
```

## Run

Input can be either `/data/private_test.csv` or `/data/public_test.csv`.
The container auto-detects the file unless `DATA_PATH` is set.

```bash
docker run --rm --gpus all \
  -v /path/to/test_data:/data:ro \
  -v /path/to/qwen_models:/models:ro \
  -v /path/to/bge-m3:/bge/bge-m3:ro \
  -v /path/to/output:/output \
  hackaithon-vllm-submission:latest
```

Optional explicit env form:

```bash
docker run --rm --gpus all \
  -e DATA_PATH=/data/private_test.csv \
  -e PRED_PATH=/output/pred.csv \
  -e CHECKPOINT_TO_RUN=ckpt12_internal_rag \
  -v /path/to/test_data:/data:ro \
  -v /path/to/qwen_models:/models:ro \
  -v /path/to/bge-m3:/bge/bge-m3:ro \
  -v /path/to/output:/output \
  hackaithon-vllm-submission:latest
```

For local debugging without Docker, first extract the two zip artifacts under
`outputs/`, then use the same package entrypoint:

```bash
mkdir -p outputs/qwen35_qlora_mcq_mixed_resume_noeval
python -m zipfile -e outputs/qwen35_qlora_mcq_mixed_resume_noeval.zip outputs/qwen35_qlora_mcq_mixed_resume_noeval
python -m zipfile -e outputs/rag_vector_db_final.zip outputs

PYTHONPATH=src python -m hackaithon_vllm.run \
  --data /data/private_test.csv \
  --output /output/pred.csv \
  --model-root /models \
  --bge-model-dir /bge/bge-m3 \
  --adapter-dir outputs/qwen35_qlora_mcq_mixed_resume_noeval \
  --rag-db-dir outputs/rag_vector_db_final
```

Important env/CLI knobs:

- `DATA_PATH` / `--data`: input CSV or JSON file.
- `PRED_PATH` / `--output`: output CSV path, default `/output/pred.csv`.
- `MODEL_ROOT` / `--model-root`: parent folder containing the Qwen model.
- `BGE_MODEL_DIR` / `--bge-model-dir`: BGE-M3 folder.
- `ADAPTER_ROOT` / `--adapter-root`: parent folder containing the QLoRA adapter.
- `ADAPTER_DIR` / `--adapter-dir`: exact adapter folder if known.
- `LAW_ADMIN_VECTOR_DB_DIR` / `--rag-db-dir`: final Law/Admin vector DB folder.
- `LIMIT` / `--limit`: optional smoke-test row limit.

The entrypoint always writes:

```text
/output/pred.csv
```

with exactly:

```text
qid,answer
```

## Notes

The runtime intentionally keeps only the production checkpoint
`ckpt12_internal_rag`; older development checkpoints were removed from the
container path. The final public-test snapshot in `outputs/pred.csv` scored
`429/463` against the current internal round-4 reference used during local
review. This reference is not part of the container contract; it is only included
here as development context.
