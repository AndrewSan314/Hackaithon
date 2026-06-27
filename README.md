# Bộ nộp bài Hackaithon MCQ vLLM

Repository này chứa gói suy luận có thể tái tạo từ notebook
`hackaithon-vllm (1).ipynb`. Runtime chỉ giữ checkpoint production
`ckpt12_internal_rag` và ghi các file nộp bài:

```text
/output/pred.csv
/output/submission.csv
/output/submission_time.csv
```

## Môi trường Docker

Dockerfile dùng base image `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` để khớp guideline CUDA 12.8+ của BTC. Dependency Python được cài bằng `uv` với flag:

```bash
uv pip install --no-cache --torch-backend=cu128 -r /app/requirements.txt
```

Cách cài này buộc PyTorch/vLLM dùng CUDA 12.8 wheel, giảm rủi ro mismatch giữa CUDA runtime, PyTorch và vLLM khi BTC chạy trên RTX 5060 Ti.

## Nội dung

- `Dockerfile`: định nghĩa image để build/push Docker Hub.
- `predict.py`: entry-point Python theo guideline BTC.
- `inference.sh`: wrapper shell gọi `predict.py`.
- `entrypoint.sh`: tự tìm file test trong `/code`, `/app/data` hoặc `/data`, chạy pipeline và ghi output.
- `src/hackaithon-vllm.ipynb`: notebook runner mỏng gọi package Python.
- `src/hackaithon_vllm_pipeline.py`: compatibility wrapper.
- `src/hackaithon_vllm/`: package Python đã tách module, runtime ckpt12-only.
- `outputs/pred.csv`: prediction snapshot mới nhất trên public test.
- `outputs/rag_vector_db_final.zip`: artifact Law/Admin RAG vector DB cuối cùng.
- `outputs/qwen35_4b_qlora_mcq_mixed.zip`: artifact QLoRA adapter production cho Qwen3.5-4B nếu được bundle vào image.
- `docs/method_report.tex`: tài liệu thuyết minh phương pháp bằng LaTeX.
- `docs/method_report.pdf`: bản PDF preview của tài liệu thuyết minh.

## Yêu cầu mount khi chạy

Image không bake sẵn model Qwen3.5-4B và BGE-M3, nên cần mount:

- `/models`: thư mục model Qwen3.5-4B định dạng Hugging Face, hoặc thư mục cha chứa model.
- `/bge/bge-m3`: thư mục model BGE-M3 định dạng Hugging Face.
- `/code`: thư mục input/output chính theo guideline BTC; chứa `private_test.json` và nhận `submission.csv`, `submission_time.csv`.
- `/data`: có thể chứa `private_test.csv`, `public_test.csv` hoặc biến thể `.json`.
- `/output`: thư mục phụ để dễ lấy artifact local; cũng nhận `pred.csv`, `submission.csv`, `submission_time.csv`.

Nếu bundle QLoRA adapter 4B, đặt zip tại:

```text
outputs/qwen35_4b_qlora_mcq_mixed.zip
```

Khi file zip này tồn tại, `entrypoint.sh` tự giải nén adapter 4B vào:

```text
/tmp/hackaithon_adapters/qwen35_4b_qlora_mcq_mixed
```

và tự set `ADAPTER_DIR` nếu người chạy chưa truyền adapter riêng.
Adapter 9B cũ không được tự nạp cho model 4B vì khác shape layer. Nếu chưa có adapter 4B zip trong `outputs/` và không truyền `ADAPTER_DIR`, runtime sẽ chạy base 4B do `REQUIRE_LORA=0` mặc định. Production nên truyền `ADAPTER_DIR`, hoặc bundle đúng `qwen35_4b_qlora_mcq_mixed.zip`, hoặc set `REQUIRE_LORA=1` để fail sớm khi thiếu adapter.

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

Container tự ưu tiên tìm đúng input chính thức `/code/private_test.json`, sau đó mới tới các fallback `private_test.csv`/`public_test.csv` trong `/app/data` hoặc `/data`.
Nếu muốn chỉ định rõ file input, set `DATA_PATH`.

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/test_data:/data:ro \
  -v /path/to/private_test_dir:/code \
  -v /path/to/qwen_models:/models:ro \
  -v /path/to/bge-m3:/bge/bge-m3:ro \
  -v /path/to/output:/output \
  hackaithon-vllm-submission:latest
```

Ví dụ với env rõ ràng:

```bash
docker run --rm --gpus all --ipc=host \
  -e DATA_PATH=/code/private_test.json \
  -e PRED_PATH=/output/pred.csv \
  -e MODEL_SELECT=4b \
  -e CHECKPOINT_TO_RUN=ckpt12_internal_rag \
  -v /path/to/private_test_dir:/code \
  -v /path/to/qwen_models:/models:ro \
  -v /path/to/bge-m3:/bge/bge-m3:ro \
  -v /path/to/output:/output \
  hackaithon-vllm-submission:latest
```

Sau khi chạy xong, file kết quả nằm tại:

```text
/code/submission.csv
/code/submission_time.csv
/output/pred.csv
/output/submission.csv
/output/submission_time.csv
```

`submission_time.csv` ghi thời gian cho từng `qid`. Với nhóm câu chạy qua vLLM batch, runtime được quy đổi theo thời gian batch chia đều cho số job trong batch; với rule solver/repair, runtime lấy trực tiếp từ từng result trong log.

## Chạy local không qua Docker

Trước tiên giải nén hai artifact trong `outputs/`:

```bash
mkdir -p outputs/qwen35_4b_qlora_mcq_mixed
python -m zipfile -e outputs/qwen35_4b_qlora_mcq_mixed.zip outputs/qwen35_4b_qlora_mcq_mixed
python -m zipfile -e outputs/rag_vector_db_final.zip outputs
```

Sau đó chạy module entrypoint:

```bash
PYTHONPATH=src python -m hackaithon_vllm.run \
  --data /data/private_test.csv \
  --output /output/pred.csv \
  --model-root /models \
  --model-select 4b \
  --bge-model-dir /bge/bge-m3 \
  --adapter-dir outputs/qwen35_4b_qlora_mcq_mixed \
  --rag-db-dir outputs/rag_vector_db_final
```

## Biến môi trường / CLI quan trọng

| Env / CLI | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `DATA_PATH` / `--data` | tự tìm trong `/data` | File input CSV/JSON. |
| `PRED_PATH` / `--output` | `/output/pred.csv` | File prediction đầu ra. |
| `MODEL_ROOT` / `--model-root` | `/models` | Thư mục cha chứa model Qwen. |
| `MODEL_SELECT` / `--model-select` | `4b` | Cách chọn model local; mặc định bắt Qwen3.5-4B. |
| `BGE_MODEL_DIR` / `--bge-model-dir` | `/bge/bge-m3` | Thư mục BGE-M3. |
| `ADAPTER_DIR` / `--adapter-dir` | adapter tự giải nén | Thư mục adapter cụ thể. |
| `ADAPTER_ROOT` / `--adapter-root` | `/adapters` | Thư mục cha để tìm adapter nếu không có `ADAPTER_DIR`. |
| `LAW_ADMIN_VECTOR_DB_DIR` / `--rag-db-dir` | RAG DB tự giải nén | Thư mục vector DB cuối cùng. |
| `CHECKPOINT_TO_RUN` / `--checkpoint` | `ckpt12_internal_rag` | Checkpoint production duy nhất được hỗ trợ. |
| `LIMIT` / `--limit` | không giới hạn | Giới hạn số dòng để smoke test. |

## Ghi chú vận hành

- Runtime cố ý chỉ giữ `ckpt12_internal_rag`; các checkpoint thử nghiệm cũ đã bị loại khỏi đường chạy container.
- BTC xác nhận môi trường chấm dùng NVIDIA RTX 5060 Ti 16GB VRAM và RAM 32GB. README vì vậy ghi rõ `--ipc=host` để vLLM có shared memory đủ ổn định khi reproduce.
- `outputs/pred.csv` là snapshot public-test đã review nội bộ.
- Trong quá trình review local, snapshot này đạt `429/463` theo reference round 4 nội bộ.
- Reference đó không thuộc contract container, chỉ là ngữ cảnh phát triển.
