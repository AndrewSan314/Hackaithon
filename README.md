# Bộ nộp bài Hackaithon MCQ vLLM

Repository này chứa gói suy luận có thể tái tạo từ notebook
`hackaithon-vllm (1).ipynb`. Runtime chỉ giữ checkpoint production
`ckpt12_internal_rag` và ghi file nộp bài bắt buộc tại `/output/pred.csv`
với đúng hai cột:

```text
qid,answer
```

## Nội dung

- `Dockerfile`: định nghĩa image để build/push Docker Hub.
- `entrypoint.sh`: tự tìm file test trong `/data`, chạy pipeline và ghi `/output/pred.csv`.
- `src/hackaithon-vllm.ipynb`: notebook runner mỏng gọi package Python.
- `src/hackaithon_vllm_pipeline.py`: compatibility wrapper.
- `src/hackaithon_vllm/`: package Python đã tách module, runtime ckpt12-only.
- `outputs/pred.csv`: prediction snapshot mới nhất trên public test.
- `outputs/rag_vector_db_final.zip`: artifact Law/Admin RAG vector DB cuối cùng.
- `outputs/qwen35_qlora_mcq_mixed_resume_noeval.zip`: artifact QLoRA adapter production.
- `docs/method_report.tex`: tài liệu thuyết minh phương pháp bằng LaTeX.
- `docs/method_report.pdf`: bản PDF preview của tài liệu thuyết minh.

## Yêu cầu mount khi chạy

Image không bake sẵn model Qwen3.5-9B và BGE-M3, nên cần mount:

- `/models`: thư mục model Qwen3.5-9B định dạng Hugging Face, hoặc thư mục cha chứa model.
- `/bge/bge-m3`: thư mục model BGE-M3 định dạng Hugging Face.
- `/data`: chứa `private_test.csv`, `public_test.csv` hoặc biến thể `.json`.
- `/output`: thư mục nhận file `/output/pred.csv`.

QLoRA adapter đã có sẵn trong repo tại:

```text
outputs/qwen35_qlora_mcq_mixed_resume_noeval.zip
```

Khi container khởi động, `entrypoint.sh` tự giải nén adapter này vào:

```text
/tmp/hackaithon_adapters/qwen35_qlora_mcq_mixed_resume_noeval
```

và tự set `ADAPTER_DIR` nếu người chạy chưa truyền adapter riêng.
Nếu muốn dùng adapter khác, mount adapter đó và set `ADAPTER_DIR` hoặc `ADAPTER_ROOT`.

Law/Admin RAG DB cũng đã có sẵn tại:

```text
outputs/rag_vector_db_final.zip
```

Khi container khởi động, `entrypoint.sh` tự giải nén RAG DB vào:

```text
/tmp/hackaithon_rag/rag_vector_db_final
```

và tự set `LAW_ADMIN_VECTOR_DB_DIR` nếu chưa mount `/rag/rag_vector_db_final`.
Nếu muốn dùng RAG DB khác, mount DB đó và set `LAW_ADMIN_VECTOR_DB_DIR`.

## Build Docker image

```bash
docker build -t hackaithon-vllm-submission:latest .
```

## Chạy bằng Docker

Container tự ưu tiên tìm `private_test.csv` rồi `public_test.csv` trong `/data`.
Nếu muốn chỉ định rõ file input, set `DATA_PATH`.

```bash
docker run --rm --gpus all \
  -v /path/to/test_data:/data:ro \
  -v /path/to/qwen_models:/models:ro \
  -v /path/to/bge-m3:/bge/bge-m3:ro \
  -v /path/to/output:/output \
  hackaithon-vllm-submission:latest
```

Ví dụ với env rõ ràng:

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

Sau khi chạy xong, file kết quả nằm tại:

```text
/output/pred.csv
```

## Chạy local không qua Docker

Trước tiên giải nén hai artifact trong `outputs/`:

```bash
mkdir -p outputs/qwen35_qlora_mcq_mixed_resume_noeval
python -m zipfile -e outputs/qwen35_qlora_mcq_mixed_resume_noeval.zip outputs/qwen35_qlora_mcq_mixed_resume_noeval
python -m zipfile -e outputs/rag_vector_db_final.zip outputs
```

Sau đó chạy module entrypoint:

```bash
PYTHONPATH=src python -m hackaithon_vllm.run \
  --data /data/private_test.csv \
  --output /output/pred.csv \
  --model-root /models \
  --bge-model-dir /bge/bge-m3 \
  --adapter-dir outputs/qwen35_qlora_mcq_mixed_resume_noeval \
  --rag-db-dir outputs/rag_vector_db_final
```

## Biến môi trường / CLI quan trọng

| Env / CLI | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `DATA_PATH` / `--data` | tự tìm trong `/data` | File input CSV/JSON. |
| `PRED_PATH` / `--output` | `/output/pred.csv` | File prediction đầu ra. |
| `MODEL_ROOT` / `--model-root` | `/models` | Thư mục cha chứa model Qwen. |
| `MODEL_SELECT` / `--model-select` | `auto` | Cách chọn model local. |
| `BGE_MODEL_DIR` / `--bge-model-dir` | `/bge/bge-m3` | Thư mục BGE-M3. |
| `ADAPTER_DIR` / `--adapter-dir` | adapter tự giải nén | Thư mục adapter cụ thể. |
| `ADAPTER_ROOT` / `--adapter-root` | `/adapters` | Thư mục cha để tìm adapter nếu không có `ADAPTER_DIR`. |
| `LAW_ADMIN_VECTOR_DB_DIR` / `--rag-db-dir` | RAG DB tự giải nén | Thư mục vector DB cuối cùng. |
| `CHECKPOINT_TO_RUN` / `--checkpoint` | `ckpt12_internal_rag` | Checkpoint production duy nhất được hỗ trợ. |
| `LIMIT` / `--limit` | không giới hạn | Giới hạn số dòng để smoke test. |

## Ghi chú vận hành

- Runtime cố ý chỉ giữ `ckpt12_internal_rag`; các checkpoint thử nghiệm cũ đã bị loại khỏi đường chạy container.
- `outputs/pred.csv` là snapshot public-test đã review nội bộ.
- Trong quá trình review local, snapshot này đạt `429/463` theo reference round 4 nội bộ.
- Reference đó không thuộc contract container, chỉ là ngữ cảnh phát triển.
