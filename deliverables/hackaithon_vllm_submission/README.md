# Hackaithon MCQ vLLM Submission Bundle

This folder contains the reproducible inference package derived from
`hackaithon-vllm (1).ipynb`.

## Contents

- `Dockerfile`: container definition for Docker Hub.
- `entrypoint.sh`: reads `/data/private_test.csv` or `/data/public_test.csv` and writes `/output/pred.csv`.
- `src/hackaithon-vllm.ipynb`: thin notebook runner for the Python package.
- `src/hackaithon_vllm_pipeline.py`: compatibility wrapper.
- `src/hackaithon_vllm/`: refactored Python package; runtime supports `ckpt12_internal_rag` only.
- `outputs/pred.csv`: latest public-test prediction snapshot.
- `outputs/rag_vector_db_final.zip`: final Law/Admin RAG vector DB artifact.
- `docs/method_report.tex`: LaTeX method document.

## Expected Runtime Mounts

The image does not bake in the 9B model, BGE model, or LoRA adapter. Mount them:

- `/models`: Qwen3.5-9B local Hugging Face model folder, or a parent containing it.
- `/bge/bge-m3`: BGE-M3 local Hugging Face model folder.
- `/adapters`: QLoRA adapter folder, or a parent containing `adapter_config.json`.
- `/rag/rag_vector_db_final`: extracted final RAG DB folder.
- `/data`: contains `private_test.csv` or `public_test.csv`.
- `/output`: output directory for `pred.csv`.

## Build

```bash
docker build -t hackaithon-vllm-submission:latest .
```

## Run

```bash
docker run --rm --gpus all \
  -v /path/to/test_data:/data:ro \
  -v /path/to/qwen_models:/models:ro \
  -v /path/to/bge-m3:/bge/bge-m3:ro \
  -v /path/to/adapter_parent:/adapters:ro \
  -v /path/to/rag_vector_db_final:/rag/rag_vector_db_final:ro \
  -v /path/to/output:/output \
  hackaithon-vllm-submission:latest
```

The equivalent module entrypoint is:

```bash
PYTHONPATH=src python -m hackaithon_vllm.run \
  --data /data/private_test.csv \
  --output /output/pred.csv \
  --model-root /models \
  --bge-model-dir /bge/bge-m3 \
  --adapter-root /adapters \
  --rag-db-dir /rag/rag_vector_db_final
```

The entrypoint writes:

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
