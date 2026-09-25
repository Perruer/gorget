"""Interfaces for plugging your own models into Gorget.

Two kinds of models can replace or extend the built-in ones:

- a **text classifier** for `PromptInjection`: any callable that takes a list of texts and
  returns labels with scores, for example a scikit-learn model, an internal HTTP service
  or a model that sorts injections into categories;
- an **entity detector** for `Anonymize` and `Sensitive`: any callable that finds spans of
  sensitive data in a text, including your own data types such as contract numbers.

Configuration files refer to such objects by import path (``package.module:attribute``) or
by a name that an installed package registers in the ``gorget.plugins`` entry point group.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable, Iterable, Sequence
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

from .exception import GorgetValidationError

ENTRY_POINT_GROUP = "gorget.plugins"

if TYPE_CHECKING:
    from .input_scanners.anonymize_helpers.custom_recognizer import CallableRecognizer

__all__ = [
    "CallableRecognizer",
    "Entity",
    "EntityDetector",
    "TextClassifier",
    "build_object",
    "load_object",
]

LabelScore = Union[dict, tuple]
"""A ``{"label": ..., "score": ...}`` dict or a ``(label, score)`` tuple."""


@runtime_checkable
class TextClassifier(Protocol):
    """
    A classifier that `PromptInjection` can use instead of its built-in model.

    It gets a batch of texts and returns, for each text, either one ``(label, score)``
    pair (the top label) or all labels with their scores. Returning all labels lets
    Gorget add up the scores of several injection categories.
    """

    def __call__(self, texts: list[str]) -> Sequence[LabelScore | Sequence[LabelScore]]: ...


@dataclasses.dataclass(frozen=True)
class Entity:
    """A span of sensitive data found by an entity detector."""

    entity_type: str
    start: int
    end: int
    score: float = 1.0


EntityLike = Union[Entity, dict, tuple]
"""An `Entity`, a dict with the same keys, or an ``(entity_type, start, end, score)`` tuple."""

EntityDetector = Callable[[str, str], Iterable[EntityLike]]
"""A callable ``(text, language) -> entities``."""


def normalize_label_scores(result: Any) -> list[tuple[str, float]]:
    """Turn one classifier result into a list of ``(label, score)`` pairs."""
    if isinstance(result, dict) or (
        isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str)
    ):
        result = [result]

    pairs = []
    for item in result:
        if isinstance(item, dict):
            label, score = item["label"], item["score"]
        else:
            label, score = item
        pairs.append((str(label), float(score)))
    return pairs


def load_object(reference: str) -> Any:
    """
    Load an object from ``package.module:attribute`` or from a name registered in the
    ``gorget.plugins`` entry point group of an installed package.
    """
    if ":" in reference:
        module_name, _, attribute = reference.partition(":")
        try:
            obj: Any = importlib.import_module(module_name)
        except ImportError as exc:
            raise GorgetValidationError(f"Cannot import {module_name!r}: {exc}") from exc
        missing = object()
        for part in attribute.split("."):
            obj = getattr(obj, part, missing)
            if obj is missing:
                raise GorgetValidationError(f"{module_name!r} has no attribute {attribute!r}")
        return obj

    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name == reference:
            return entry_point.load()

    raise GorgetValidationError(
        f"Unknown plugin {reference!r}: use 'package.module:attribute' or install a package "
        f"that registers it in the {ENTRY_POINT_GROUP!r} entry point group"
    )


def build_object(spec: Any) -> Any:
    """
    Build an object from configuration:

    - ``"package.module:attribute"`` gives the object it points to (a class is instantiated
      without arguments);
    - ``{"class": ref, "params": {...}}`` calls the class or factory with the parameters;
    - ``{"ref": ref}`` gives the object without calling it.

    Parameter values may themselves be ``{"class": ...}`` or ``{"ref": ...}`` mappings, so a
    YAML file can pass a function or another object to a constructor. Anything else is
    returned unchanged.
    """
    if isinstance(spec, str):
        obj = load_object(spec)
        return obj() if isinstance(obj, type) else obj
    if isinstance(spec, dict) and set(spec) == {"ref"}:
        return load_object(spec["ref"])
    if isinstance(spec, dict) and "class" in spec:
        factory = load_object(spec["class"])
        params = {key: _build_param(value) for key, value in (spec.get("params") or {}).items()}
        return factory(**params)
    return spec


def _build_param(value: Any) -> Any:
    if isinstance(value, dict) and ("class" in value or set(value) == {"ref"}):
        return build_object(value)
    if isinstance(value, list):
        return [_build_param(item) for item in value]
    return value


def __getattr__(name: str) -> Any:
    # CallableRecognizer needs Presidio, which is slow to import, so it is loaded on use.
    if name == "CallableRecognizer":
        from .input_scanners.anonymize_helpers.custom_recognizer import CallableRecognizer

        return CallableRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
