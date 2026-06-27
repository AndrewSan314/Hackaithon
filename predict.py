#!/usr/bin/env python
"""BTC-compatible prediction entrypoint.

Reads DATA_PATH or the default mounted test file, runs ckpt12_internal_rag, and
writes pred.csv plus submission.csv/submission_time.csv aliases.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hackaithon_vllm.run import main


if __name__ == "__main__":
    main()
