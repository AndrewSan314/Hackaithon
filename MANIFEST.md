# Manifest

Generated from local workspace on 2026-06-25.

| Path | Purpose |
| --- | --- |
| `Dockerfile` | Docker Hub build file. |
| `predict.py` | BTC-compatible Python entrypoint. |
| `inference.sh` | Shell wrapper for predict.py. |
| `entrypoint.sh` | Container entrypoint. |
| `requirements.txt` | Additional Python dependencies on top of vLLM image. |
| `src/hackaithon-vllm.ipynb` | Thin notebook runner. |
| `src/hackaithon_vllm_pipeline.py` | Compatibility wrapper calling the package entrypoint. |
| `src/hackaithon_vllm/` | Refactored Python package; ckpt12-only runtime. |
| `outputs/pred.csv` | Latest prediction CSV snapshot. |
| `outputs/rag_vector_db_final.zip` | Final RAG vector DB artifact. |
| `outputs/qwen35_4b_qlora_mcq_mixed.zip` | Production QLoRA adapter artifact for Qwen3.5-4B, if bundled. |
| `docs/method_report.tex` | LaTeX method report. |
| `docs/method_report.pdf` | Compiled PDF preview of the LaTeX report. |

Large artifacts expected as external mounts:

- Qwen3.5-4B model.
- BGE-M3 embedding model.
