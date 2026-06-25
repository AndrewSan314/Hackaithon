# Bộ nộp bài Hackaithon MCQ vLLM

Thư mục này chứa gói suy luận có thể tái tạo (reproducible inference package) được trích xuất từ
`hackaithon-vllm (1).ipynb`.

## Nội dung

- `Dockerfile`: định nghĩa container cho Docker Hub.
- `entrypoint.sh`: đọc `/data/private_test.csv` hoặc `/data/public_test.csv` và ghi ra `/output/pred.csv`.
- `src/hackaithon-vllm.ipynb`: notebook runner mỏng cho gói Python.
- `src/hackaithon_vllm_pipeline.py`: wrapper tương thích.
- `src/hackaithon_vllm/`: gói Python đã được cấu trúc lại (refactored); runtime chỉ hỗ trợ `ckpt12_internal_rag`.
- `outputs/pred.csv`: bản chụp dự đoán (prediction snapshot) mới nhất cho public-test.
- `outputs/rag_vector_db_final.zip`: artifact cơ sở dữ liệu vector RAG cuối cùng về Luật/Hành chính.
- `docs/method_report.tex`: tài liệu phương pháp bằng LaTeX.

## Các mount điểm dự kiến khi chạy (Expected Runtime Mounts)

Image này không tích hợp sẵn mô hình 9B, mô hình BGE, hoặc adapter LoRA. Vui lòng mount chúng:

- `/models`: Thư mục mô hình Hugging Face cục bộ của Qwen3.5-9B, hoặc thư mục cha chứa nó.
- `/bge/bge-m3`: Thư mục mô hình Hugging Face cục bộ của BGE-M3.
- `/adapters`: Thư mục adapter QLoRA, hoặc thư mục cha chứa `adapter_config.json`.
- `/rag/rag_vector_db_final`: Thư mục RAG DB cuối cùng đã được giải nén.
- `/data`: chứa `private_test.csv` hoặc `public_test.csv`.
- `/output`: thư mục đầu ra cho `pred.csv`.

## Build

```bash
docker build -t hackaithon-vllm-submission:latest .
```

## Chạy (Run)

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

Entrypoint tương đương cho module là:

```bash
PYTHONPATH=src python -m hackaithon_vllm.run \
  --data /data/private_test.csv \
  --output /output/pred.csv \
  --model-root /models \
  --bge-model-dir /bge/bge-m3 \
  --adapter-root /adapters \
  --rag-db-dir /rag/rag_vector_db_final
```

Entrypoint sẽ ghi ra:

```text
/output/pred.csv
```

với định dạng chính xác là:

```text
qid,answer
```

## Ghi chú

Runtime cố ý chỉ giữ lại checkpoint chạy thực tế (production)
`ckpt12_internal_rag`; các checkpoint phát triển cũ đã bị loại bỏ khỏi
đường dẫn của container. Bản chụp public-test cuối cùng ở `outputs/pred.csv` đạt
điểm `429/463` dựa trên tham chiếu nội bộ vòng 4 (round-4 reference) hiện tại được sử dụng trong quá trình
đánh giá cục bộ. Tham chiếu này không thuộc phạm vi hợp đồng container; nó chỉ được đưa vào
đây như một ngữ cảnh phát triển.
