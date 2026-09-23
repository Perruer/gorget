from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from typing import Literal, get_args

from . import runtime
from .exception import GorgetValidationError
from .model import Model
from .util import device, get_logger, lazy_load_dep

LOGGER = get_logger()

Backend = Literal["torch", "onnx"]


@lru_cache(maxsize=None)
def is_torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


@lru_cache(maxsize=None)  # Unbounded cache
def is_onnx_supported() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


@lru_cache(maxsize=None)
def _log_backend_once(message: str) -> None:
    LOGGER.info(message)


def resolve_backend(use_onnx: bool = False) -> Backend:
    """
    Pick the inference backend for a scanner model.

    `use_onnx=True` asks for ONNX Runtime. Otherwise PyTorch is used when it is installed and
    ONNX Runtime when it is not, so `pip install gorget` works without PyTorch. The
    `GORGET_BACKEND` environment variable (`torch` or `onnx`) overrides both.
    """
    forced = os.environ.get("GORGET_BACKEND", "").strip().lower()
    if forced in get_args(Backend):
        return forced  # type: ignore[return-value]

    if use_onnx:
        if is_onnx_supported():
            return "onnx"
        LOGGER.warning(
            "ONNX Runtime is not installed, using PyTorch. Install it with `pip install onnxruntime`."
        )

    if is_torch_available():
        return "torch"

    if is_onnx_supported():
        _log_backend_once("PyTorch is not installed, running models with ONNX Runtime")
        return "onnx"

    raise GorgetValidationError(
        "No inference backend found. Install ONNX Runtime (`pip install onnxruntime`) "
        "or PyTorch (`pip install gorget[torch]`)."
    )


def _causes(exc: BaseException):
    """The exception and everything it was raised from (transformers wraps Hub errors)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _hub_error(exc: Exception) -> bool:
    """True for Hub errors that mean the repository itself is gone or needs access."""
    errors = lazy_load_dep("huggingface_hub.errors", "huggingface_hub")
    wanted = (errors.RepositoryNotFoundError, errors.GatedRepoError)
    return any(isinstance(e, wanted) for e in _causes(exc))


def _gated_hint(model: Model, exc: Exception) -> GorgetValidationError | None:
    errors = lazy_load_dep("huggingface_hub.errors", "huggingface_hub")
    if any(isinstance(e, errors.GatedRepoError) for e in _causes(exc)):
        repo = model.onnx_path or model.path
        return GorgetValidationError(
            f"{repo} is a gated model on Hugging Face. Accept its terms at "
            f"https://huggingface.co/{repo} and set the HF_TOKEN environment variable."
        )
    return None


def _load_onnx(model: Model):
    try:
        return runtime.load(model)
    except Exception as exc:
        hint = _gated_hint(model, exc)
        if hint is not None:
            raise hint from exc
        raise


def get_tokenizer(model: Model):
    """
    This function loads a tokenizer given a model identifier and caches it.
    Subsequent calls with the same model_identifier will return the cached tokenizer.

    Args:
        model (Model): The model to load the tokenizer for.
    """
    transformers = lazy_load_dep("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.path, revision=model.revision, **model.tokenizer_kwargs
    )
    return tokenizer


def _load_torch(model: Model, auto_class: str):
    transformers = lazy_load_dep("transformers")
    tf_tokenizer = get_tokenizer(model)
    tf_model = getattr(transformers, auto_class).from_pretrained(
        model.path,
        subfolder=model.subfolder,
        revision=model.revision,
        **model.kwargs,
    )
    LOGGER.debug("Initialized PyTorch model", model=model, device=device())
    return tf_tokenizer, tf_model


@lru_cache(maxsize=None)
def _warn_non_commercial(path: str, license_name: str) -> None:
    LOGGER.warning(
        "Model license does not allow commercial use. For commercial products pick a model "
        "with a permissive license, e.g. recognizer_conf=BERT_SMALL_GRAVITEE_PII_CONF for PII.",
        model=path,
        license=license_name,
    )


def _load(model: Model, use_onnx: bool, auto_class: str):
    if model.license and "-nc" in model.license.lower():
        _warn_non_commercial(model.path, model.license)

    if resolve_backend(use_onnx) == "onnx":
        return _load_onnx(model)

    try:
        return _load_torch(model, auto_class)
    except Exception as exc:
        # A deleted or gated PyTorch repository should not break a scanner whose
        # ONNX export is still available.
        if model.onnx_path and is_onnx_supported() and _hub_error(exc):
            LOGGER.warning(
                "PyTorch model is unavailable, using its ONNX export",
                model=model.path,
                onnx_model=model.onnx_path,
            )
            return _load_onnx(model)
        hint = _gated_hint(model, exc)
        if hint is not None:
            raise hint from exc
        raise


def get_tokenizer_and_model_for_classification(
    model: Model,
    use_onnx: bool = False,
):
    """
    This function loads a tokenizer and model given a model identifier.

    Args:
        model (Model): The model to load the tokenizer and model for.
        use_onnx (bool): Whether to use the ONNX version of the model. Defaults to False.
    """
    return _load(model, use_onnx, "AutoModelForSequenceClassification")


def get_tokenizer_and_model_for_ner(
    model: Model,
    use_onnx: bool = False,
):
    """
    This function loads a tokenizer and model given a model identifier.

    Args:
        model (Model): The model to load the tokenizer and model for.
        use_onnx (bool): Whether to use the ONNX version of the model. Defaults to False.
    """
    return _load(model, use_onnx, "AutoModelForTokenClassification")


def get_tokenizer_and_model_for_embeddings(
    model: Model,
    use_onnx: bool = False,
):
    """Load a tokenizer and a bare encoder that returns hidden states."""
    return _load(model, use_onnx, "AutoModel")


ClassificationTask = Literal["text-classification", "zero-shot-classification"]
NERTask = Literal["ner", "token-classification"]


def pipeline(
    task: str,
    model,
    tokenizer,
    **kwargs,
):
    if task not in get_args(ClassificationTask) + get_args(NERTask):
        raise GorgetValidationError(
            f"Invalid task. Must be one of {get_args(ClassificationTask) + get_args(NERTask)}"
        )

    if isinstance(model, runtime.OnnxModel):
        return runtime.pipeline(task, model, tokenizer, **kwargs)

    if task in get_args(ClassificationTask) and kwargs.get("max_length", None) is None:
        kwargs["max_length"] = tokenizer.model_max_length
    kwargs.setdefault("device", device())

    transformers = lazy_load_dep("transformers")
    return transformers.pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        **kwargs,
    )
