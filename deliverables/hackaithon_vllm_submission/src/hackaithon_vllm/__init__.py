"""Hackaithon vLLM MCQ inference package.

The production entrypoint is :mod:`hackaithon_vllm.run`.  The current package
keeps the ckpt12 runtime behavior stable while exposing smaller modules for
configuration, I/O, output validation, and final Law/Admin RAG.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
