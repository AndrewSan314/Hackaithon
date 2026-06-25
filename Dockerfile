FROM vllm/vllm-openai:latest

ENV PYTHONUNBUFFERED=1 \
    USE_TORCH=1 \
    USE_TF=0 \
    USE_FLAX=0 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY outputs /app/outputs
COPY src /app/src
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
