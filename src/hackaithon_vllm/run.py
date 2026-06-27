from __future__ import annotations

import runpy
import time
from pathlib import Path

from .config import PipelineConfig
from .output import write_submission_aliases
from .vllm_engine import configure_vllm_environment


def main(argv: list[str] | None = None) -> None:
    config = PipelineConfig.from_args(argv)
    if config.checkpoint_to_run != "ckpt12_internal_rag":
        raise ValueError("This refactored runtime only supports ckpt12_internal_rag.")
    configure_vllm_environment()
    config.apply_to_env()
    legacy_path = Path(__file__).with_name("_legacy_pipeline.py")
    start = time.perf_counter()
    namespace = runpy.run_path(str(legacy_path), run_name="__main__")
    elapsed = time.perf_counter() - start
    if config.pred_path.exists():
        metrics = namespace.get("metrics") if isinstance(namespace, dict) else None
        log_path = Path(metrics["log_path"]) if isinstance(metrics, dict) and metrics.get("log_path") else None
        aliases = write_submission_aliases(config.pred_path, elapsed_seconds=elapsed, log_path=log_path)
        print({"submission_aliases": [str(path) for path in aliases]})


if __name__ == "__main__":
    main()
