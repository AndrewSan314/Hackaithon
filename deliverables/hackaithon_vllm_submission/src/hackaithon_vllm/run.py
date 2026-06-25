from __future__ import annotations

import runpy
from pathlib import Path

from .config import PipelineConfig
from .vllm_engine import configure_vllm_environment


def main(argv: list[str] | None = None) -> None:
    config = PipelineConfig.from_args(argv)
    if config.checkpoint_to_run != "ckpt12_internal_rag":
        raise ValueError("This refactored runtime only supports ckpt12_internal_rag.")
    configure_vllm_environment()
    config.apply_to_env()
    legacy_path = Path(__file__).with_name("_legacy_pipeline.py")
    runpy.run_path(str(legacy_path), run_name="__main__")


if __name__ == "__main__":
    main()
