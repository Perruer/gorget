"""Gorget package"""

import importlib.util as _importlib_util
import os as _os

if _importlib_util.find_spec("torch") is None:
    # Without PyTorch Gorget runs on ONNX Runtime and uses transformers only for
    # tokenizers, so its "PyTorch was not found" advice would be noise.
    _os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from .evaluate import scan_output, scan_prompt  # noqa: E402

__all__ = ["scan_output", "scan_prompt"]
