from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BGEState:
    model_dir: Path
    tokenizer: Any = None
    model: Any = None
    device: str | None = None
    disabled_reason: str | None = None
