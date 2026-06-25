from __future__ import annotations

from pathlib import Path


def lora_adapter_ready(path: Path) -> bool:
    return (
        path.exists()
        and path.is_dir()
        and (path / "adapter_config.json").exists()
        and any((path / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    )
