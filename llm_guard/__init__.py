"""Compatibility package for code written against LLM Guard.

Gorget continues LLM Guard, which Protect AI archived in July 2026. Every
``llm_guard`` module name points at the same module object in ``gorget``, so
``from llm_guard.input_scanners import PromptInjection`` keeps working and
``isinstance`` checks see one set of classes.
"""

import importlib
import pkgutil
import sys

import gorget

__all__ = ["scan_output", "scan_prompt"]

_OLD, _NEW = __name__, gorget.__name__


def _alias(module_name: str):
    module = importlib.import_module(module_name)
    alias = _OLD + module_name[len(_NEW) :]
    sys.modules[alias] = module
    parent, _, child = alias.rpartition(".")
    if parent == _OLD:
        globals()[child] = module
    return module


for _info in pkgutil.walk_packages(gorget.__path__, _NEW + "."):
    _alias(_info.name)

from gorget import scan_output, scan_prompt  # noqa: E402
