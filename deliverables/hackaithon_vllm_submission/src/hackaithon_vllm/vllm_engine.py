from __future__ import annotations

import os


def configure_vllm_environment() -> None:
    os.environ.update(
        {
            "USE_TORCH": "1",
            "USE_TF": "0",
            "USE_FLAX": "0",
            "USE_TORCH_XLA": "0",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
        }
    )
    for key in ("VLLM_USE_V1", "VLLM_QUANTIZATION", "VLLM_ATTENTION_BACKEND"):
        os.environ.pop(key, None)
