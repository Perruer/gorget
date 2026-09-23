"""ONNX Runtime inference without PyTorch.

Every scanner model runs either through PyTorch (transformers pipelines) or through ONNX
Runtime. This module is the ONNX side: it loads exported models from the Hugging Face Hub
and reimplements the few transformers pipelines the scanners use, returning results in the
same shape, so a scanner does not need to know which backend is active.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .exception import GorgetValidationError
from .model import Model
from .util import get_logger, lazy_load_dep

LOGGER = get_logger()

# Files needed next to the ONNX graph: tokenizer, model config and label maps.
_SIDE_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "spm.model",
    "sentencepiece.bpe.model",
    "spiece.model",
)

# Some tokenizers report a huge sentinel as their maximum length.
_FALLBACK_MAX_LENGTH = 512
_BATCH_SIZE = 8

_UNSET: Any = object()


@dataclass
class OnnxModel:
    """An ONNX Runtime session plus the transformers config that describes its labels."""

    session: Any
    config: Any
    name_or_path: str
    tokenizer: Any = None
    _input_types: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._input_types = {i.name: i.type for i in self.session.get_inputs()}

    @property
    def input_names(self) -> list[str]:
        return list(self._input_types)

    def run(self, encoded: dict[str, Any]) -> dict[str, np.ndarray]:
        feed = {}
        for name, type_name in self._input_types.items():
            dtype = np.int32 if "int32" in type_name else np.int64
            if name in encoded:
                feed[name] = np.asarray(encoded[name], dtype=dtype)
            elif name == "token_type_ids":
                # Models exported with token type ids get zeros when the tokenizer
                # does not return them, which matches what PyTorch does by default.
                feed[name] = np.zeros_like(np.asarray(encoded["input_ids"]), dtype=dtype)
            else:
                raise GorgetValidationError(f"ONNX model {self.name_or_path} expects input {name}")

        outputs = self.session.run(None, feed)
        return {o.name: value for o, value in zip(self.session.get_outputs(), outputs)}

    def __str__(self) -> str:
        return self.name_or_path


def providers() -> list[str]:
    """Execution providers to use, CUDA first when onnxruntime-gpu can see a GPU."""
    ort = lazy_load_dep("onnxruntime")
    available = ort.get_available_providers()

    wanted = os.environ.get("GORGET_ONNX_PROVIDERS", "")
    if wanted:
        chosen = [p.strip() for p in wanted.split(",") if p.strip() in available]
        return chosen or ["CPUExecutionProvider"]

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def _download(model: Model) -> tuple[Path, Path]:
    """Fetch the ONNX graph and its side files; return (graph path, directory with config)."""
    huggingface_hub = lazy_load_dep("huggingface_hub")

    repo = model.onnx_path or model.path
    subfolder = model.onnx_subfolder.strip("/") if model.onnx_path else ""
    prefix = f"{subfolder}/" if subfolder else ""
    graph = f"{prefix}{model.onnx_filename}"

    if Path(repo).is_dir():
        # A model already on disk, e.g. baked into a container image.
        local_dir = Path(repo)
        graph_path = local_dir / graph
        if not graph_path.exists():
            raise GorgetValidationError(f"{repo} has no ONNX model at {graph}")
        side_dir = graph_path.parent if (graph_path.parent / "config.json").exists() else local_dir
        return graph_path, side_dir

    patterns = [graph, f"{graph}_data", f"{graph}.data"]
    patterns += [f"{prefix}{name}" for name in _SIDE_FILES]
    patterns += list(_SIDE_FILES)

    params = {
        "repo_id": repo,
        "revision": model.onnx_revision if model.onnx_path else model.revision,
        "allow_patterns": patterns,
        "token": model.kwargs.get("token"),
    }
    try:
        # A cached snapshot needs no network: faster startup and offline containers.
        local_dir = Path(huggingface_hub.snapshot_download(local_files_only=True, **params))
        if not (local_dir / graph).exists():
            raise FileNotFoundError(graph)
    except Exception:
        local_dir = Path(huggingface_hub.snapshot_download(**params))

    graph_path = local_dir / graph
    if not graph_path.exists():
        raise GorgetValidationError(f"{repo} has no ONNX model at {graph}")

    side_dir = graph_path.parent
    if not (side_dir / "config.json").exists():
        side_dir = local_dir

    return graph_path, side_dir


# Weak references: a session is freed once no scanner holds it, but scanners created at
# the same time (Anonymize and Sensitive, Language and LanguageSame) share one.
_CACHE: weakref.WeakValueDictionary[tuple, OnnxModel] = weakref.WeakValueDictionary()
_CACHE_LOCK = threading.Lock()


def load(model: Model) -> tuple[Any, OnnxModel]:
    """Load (tokenizer, OnnxModel) for a scanner model; identical models share a session."""
    key = (
        model.onnx_path or model.path,
        model.onnx_revision,
        model.onnx_subfolder,
        model.onnx_filename,
        repr(sorted(model.tokenizer_kwargs.items())),
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached.tokenizer, cached

        ort = lazy_load_dep("onnxruntime")
        if importlib.util.find_spec("torch") is None:
            # Only tokenizers and configs are used here, so transformers' advice that
            # PyTorch is missing is noise.
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        transformers = lazy_load_dep("transformers")

        graph_path, side_dir = _download(model)

        tokenizer_kwargs = {k: v for k, v in model.tokenizer_kwargs.items() if k != "token"}
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(side_dir), **tokenizer_kwargs)
        config = transformers.AutoConfig.from_pretrained(str(side_dir))

        options = ort.SessionOptions()
        options.log_severity_level = 3
        threads = os.environ.get("GORGET_ONNX_THREADS", "")
        if threads.isdigit():
            options.intra_op_num_threads = int(threads)
        session = ort.InferenceSession(str(graph_path), options, providers=providers())

        onnx_model = OnnxModel(
            session=session, config=config, name_or_path=str(model), tokenizer=tokenizer
        )
        LOGGER.debug("Initialized ONNX model", model=model, providers=session.get_providers())

        _CACHE[key] = onnx_model
        return tokenizer, onnx_model


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = np.exp(x - x.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _max_length(tokenizer, max_length: int | None) -> int:
    if max_length:
        return max_length
    limit = getattr(tokenizer, "model_max_length", None) or _FALLBACK_MAX_LENGTH
    return limit if limit < 100_000 else _FALLBACK_MAX_LENGTH


def _batches(items: list, size: int = _BATCH_SIZE):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class _Pipeline:
    def __init__(
        self,
        model: OnnxModel,
        tokenizer: Any,
        *,
        max_length: int | None = None,
        truncation: bool | str = True,
        return_token_type_ids: bool | None = None,
        **_ignored: Any,
    ) -> None:
        # batch_size, device and padding only matter for PyTorch; ONNX batches with
        # dynamic padding, which gives the same scores faster.
        self.model = model
        self.tokenizer = tokenizer
        self._max_length = _max_length(tokenizer, max_length)
        self._truncation = truncation
        self._return_token_type_ids = return_token_type_ids

    def _encode(self, first: list[str], second: list[str] | None = None, **kwargs: Any):
        params: dict[str, Any] = {
            "padding": True,
            "truncation": self._truncation,
            "max_length": self._max_length,
            "return_tensors": "np",
        }
        if self._return_token_type_ids is not None:
            params["return_token_type_ids"] = self._return_token_type_ids
        params.update(kwargs)
        if second is None:
            return self.tokenizer(first, **params)
        return self.tokenizer(first, second, **params)

    def _logits(self, first: list[str], second: list[str] | None = None) -> np.ndarray:
        chunks = []
        for start in range(0, len(first), _BATCH_SIZE):
            end = start + _BATCH_SIZE
            encoded = self._encode(first[start:end], second[start:end] if second else None)
            chunks.append(self.model.run(dict(encoded))["logits"])
        return np.concatenate(chunks, axis=0).astype(np.float64)


class TextClassificationPipeline(_Pipeline):
    """Mirrors transformers' text-classification pipeline outputs."""

    def __init__(self, model, tokenizer, *, top_k=_UNSET, function_to_apply=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self._top_k = top_k
        self._function_to_apply = function_to_apply

    def _activation(self, function_to_apply: str | None) -> str:
        if function_to_apply:
            return str(function_to_apply).lower()
        config = self.model.config
        problem_type = getattr(config, "problem_type", None)
        if problem_type == "regression":
            return "none"
        if problem_type == "multi_label_classification" or config.num_labels == 1:
            return "sigmoid"
        if problem_type == "single_label_classification" or config.num_labels > 1:
            return "softmax"
        return str(getattr(config, "function_to_apply", "none")).lower()

    def __call__(self, inputs, top_k=_UNSET, function_to_apply=None):
        single = isinstance(inputs, str)
        texts = [inputs] if single else list(inputs)
        effective_top_k = self._top_k if top_k is _UNSET else top_k
        legacy = effective_top_k is _UNSET
        activation = self._activation(function_to_apply or self._function_to_apply)

        results = []
        if texts:
            logits = self._logits(texts)
            if activation == "sigmoid":
                scores = _sigmoid(logits)
            elif activation == "softmax":
                scores = _softmax(logits)
            else:
                scores = logits

            id2label = self.model.config.id2label
            for row in scores:
                if legacy:
                    best = int(row.argmax())
                    results.append({"label": id2label[best], "score": float(row[best])})
                    continue
                item = [{"label": id2label[i], "score": float(s)} for i, s in enumerate(row)]
                item.sort(key=lambda x: x["score"], reverse=True)
                if effective_top_k is not None:
                    item = item[:effective_top_k]
                results.append(item)

        if single:
            # transformers wraps a single string in a list unless top_k is passed per call.
            return [results[0]] if top_k is _UNSET else results[0]
        return results


class ZeroShotClassificationPipeline(_Pipeline):
    """Mirrors transformers' zero-shot-classification pipeline (NLI based)."""

    def __init__(self, model, tokenizer, **kwargs):
        # transformers ignores tokenizer settings for zero-shot and truncates only the premise.
        kwargs.pop("max_length", None)
        kwargs.pop("truncation", None)
        super().__init__(model, tokenizer, truncation="only_first", **kwargs)
        self._entailment_id = -1
        for label, index in model.config.label2id.items():
            if str(label).lower().startswith("entail"):
                self._entailment_id = int(index)

    def __call__(
        self,
        sequences,
        candidate_labels,
        hypothesis_template: str = "This example is {}.",
        multi_label: bool = False,
    ):
        if isinstance(candidate_labels, str):
            candidate_labels = [x.strip() for x in candidate_labels.split(",") if x.strip()]
        if not candidate_labels:
            raise GorgetValidationError("You must include at least one label.")

        single = isinstance(sequences, str)
        results = []
        for sequence in [sequences] if single else list(sequences):
            hypotheses = [hypothesis_template.format(label) for label in candidate_labels]
            logits = self._logits([sequence] * len(hypotheses), hypotheses)

            if multi_label or len(candidate_labels) == 1:
                contradiction_id = -1 if self._entailment_id == 0 else 0
                pair = logits[:, [contradiction_id, self._entailment_id]]
                scores = _softmax(pair)[:, 1]
            else:
                scores = _softmax(logits[:, self._entailment_id])

            order = list(reversed(scores.argsort()))
            results.append(
                {
                    "sequence": sequence,
                    "labels": [candidate_labels[i] for i in order],
                    "scores": scores[order].tolist(),
                }
            )

        return results[0] if single else results


class TokenClassificationPipeline(_Pipeline):
    """Mirrors transformers' token-classification pipeline with the "simple" aggregation."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        aggregation_strategy: str = "simple",
        ignore_labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(model, tokenizer, **kwargs)
        strategy = str(getattr(aggregation_strategy, "value", aggregation_strategy)).lower()
        if strategy not in ("simple", "none"):
            raise GorgetValidationError(
                f"ONNX runtime supports aggregation_strategy 'simple' and 'none', got {strategy}"
            )
        self._aggregate = strategy == "simple"
        self._ignore_labels = ignore_labels if ignore_labels is not None else ["O"]

    def __call__(self, inputs):
        single = isinstance(inputs, str)
        results = [self._predict(text) for text in ([inputs] if single else list(inputs))]
        return results[0] if single else results

    def _predict(self, text: str) -> list[dict[str, Any]]:
        encoded = self.tokenizer(
            [text],
            truncation=True,
            max_length=self._max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="np",
        )
        offsets = encoded.pop("offset_mapping")[0]
        special = encoded.pop("special_tokens_mask")[0]
        input_ids = encoded["input_ids"][0]
        logits = self.model.run(dict(encoded))["logits"][0].astype(np.float64)
        scores = _softmax(logits)
        id2label = self.model.config.id2label

        entities = []
        for idx, token_scores in enumerate(scores):
            if special[idx]:
                continue
            best = int(token_scores.argmax())
            start, end = int(offsets[idx][0]), int(offsets[idx][1])
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                word = text[start:end]
            entities.append(
                {
                    "entity": id2label[best],
                    "score": np.float32(token_scores[best]),
                    "index": idx,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )

        if self._aggregate:
            entities = self._group_entities(entities)
            return [e for e in entities if e["entity_group"] not in self._ignore_labels]

        return [e for e in entities if e["entity"] not in self._ignore_labels]

    @staticmethod
    def _tag(entity_name: str) -> tuple[str, str]:
        if entity_name.startswith("B-"):
            return "B", entity_name[2:]
        if entity_name.startswith("I-"):
            return "I", entity_name[2:]
        return "I", entity_name

    def _group(self, tokens: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "entity_group": tokens[0]["entity"].split("-", 1)[-1],
            "score": np.float32(np.nanmean([t["score"] for t in tokens])),
            "word": self.tokenizer.convert_tokens_to_string([t["word"] for t in tokens]),
            "start": tokens[0]["start"],
            "end": tokens[-1]["end"],
        }

    def _group_entities(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        for entity in entities:
            if not current:
                current.append(entity)
                continue
            bi, tag = self._tag(entity["entity"])
            _, last_tag = self._tag(current[-1]["entity"])
            if tag == last_tag and bi != "B":
                current.append(entity)
            else:
                groups.append(self._group(current))
                current = [entity]
        if current:
            groups.append(self._group(current))
        return groups


class FeatureExtractionPipeline(_Pipeline):
    """Sentence embeddings with CLS or mean pooling."""

    def embed(self, texts: list[str], *, pooling: str = "cls", normalize: bool = True):
        vectors = []
        for batch in _batches(texts):
            encoded = self._encode(batch)
            hidden = self.model.run(dict(encoded))["last_hidden_state"].astype(np.float64)
            if pooling == "mean":
                mask = np.asarray(encoded["attention_mask"], dtype=np.float64)[..., None]
                pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            else:
                pooled = hidden[:, 0]
            if normalize:
                pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12)
            vectors.append(pooled)
        return np.concatenate(vectors, axis=0)


class SequencePairClassifier(_Pipeline):
    """Raw probabilities for (text, text_pair) inputs, used by FactualConsistency."""

    def probabilities(self, text: str, text_pair: str) -> np.ndarray:
        return _softmax(self._logits([text], [text_pair]))[0]


_TASKS = {
    "text-classification": TextClassificationPipeline,
    "zero-shot-classification": ZeroShotClassificationPipeline,
    "ner": TokenClassificationPipeline,
    "token-classification": TokenClassificationPipeline,
    "feature-extraction": FeatureExtractionPipeline,
}


def pipeline(task: str, model: OnnxModel, tokenizer: Any, **kwargs: Any):
    if task not in _TASKS:
        raise GorgetValidationError(f"Task {task} is not supported by the ONNX runtime")
    return _TASKS[task](model, tokenizer, **kwargs)
