FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    USE_TORCH=1 \
    USE_TF=0 \
    USE_FLAX=0 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && python -m pip install --no-cache-dir --upgrade pip uv

COPY requirements.txt /app/requirements.txt
RUN uv pip install --no-cache --torch-backend=cu128 -r /app/requirements.txt

COPY outputs /app/outputs
COPY src /app/src
COPY predict.py /app/predict.py
COPY inference.sh /app/inference.sh
COPY entrypoint.sh /app/entrypoint.sh
RUN python -c "from pathlib import Path; p=Path('/app/entrypoint.sh'); p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))" \
    && python -c "from pathlib import Path; p=Path('/app/inference.sh'); p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))" \
    && chmod +x /app/entrypoint.sh /app/inference.sh

ENTRYPOINT ["/app/entrypoint.sh"]
