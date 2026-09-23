"""ONNX Runtime pipelines must return what the transformers pipelines return."""

from __future__ import annotations

import gc
import importlib.util

import pytest

from gorget import runtime, transformers_helpers
from gorget.input_scanners.anonymize_helpers.ner_mapping import BERT_BASE_NER_CONF
from gorget.input_scanners.ban_topics import MODEL_ROBERTA_BASE_C_V2
from gorget.input_scanners.toxicity import DEFAULT_MODEL as TOXICITY_MODEL
from gorget.output_scanners.no_refusal import DEFAULT_MODEL as NO_REFUSAL_MODEL
from gorget.output_scanners.relevance import MODEL_EN_BGE_SMALL, Relevance

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="PyTorch is needed for the comparison"
)

TEXTS = [
    "I'm sorry, but I can't help with that request.",
    "Sure! Here is a poem about the sea. The waves are calm tonight.",
    "You are a stupid idiot and nobody likes you.",
]


def _pair(model, auto_class: str):
    onnx_tokenizer, onnx_model = runtime.load(model)
    torch_tokenizer, torch_model = transformers_helpers._load_torch(model, auto_class)
    return (onnx_tokenizer, onnx_model), (torch_tokenizer, torch_model)


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 2e-3


@pytest.fixture(autouse=True)
def _free_models():
    yield
    gc.collect()


@needs_torch
@pytest.mark.parametrize(
    "model, kwargs",
    [
        (NO_REFUSAL_MODEL, {}),
        (TOXICITY_MODEL, {}),
    ],
    ids=["legacy-top-1", "top-k-none-sigmoid"],
)
def test_text_classification_matches_transformers(model, kwargs):
    (ot, om), (tt, tm) = _pair(model, "AutoModelForSequenceClassification")
    onnx_pipe = transformers_helpers.pipeline(
        "text-classification", om, ot, **model.pipeline_kwargs, **kwargs
    )
    torch_pipe = transformers_helpers.pipeline(
        "text-classification", tm, tt, **model.pipeline_kwargs, **kwargs
    )

    for inputs in (TEXTS, TEXTS[0]):
        got, want = onnx_pipe(inputs), torch_pipe(inputs)
        assert type(got) is type(want)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            g_items = g if isinstance(g, list) else [g]
            w_items = w if isinstance(w, list) else [w]
            assert [x["label"] for x in g_items] == [x["label"] for x in w_items]
            assert all(_close(x["score"], y["score"]) for x, y in zip(g_items, w_items))


@needs_torch
def test_zero_shot_matches_transformers():
    model = MODEL_ROBERTA_BASE_C_V2
    (ot, om), (tt, tm) = _pair(model, "AutoModelForSequenceClassification")
    onnx_pipe = transformers_helpers.pipeline(
        "zero-shot-classification", om, ot, **model.pipeline_kwargs
    )
    torch_pipe = transformers_helpers.pipeline(
        "zero-shot-classification", tm, tt, **model.pipeline_kwargs
    )
    topics = ["violence", "cooking", "politics"]

    for multi_label in (False, True):
        got = onnx_pipe(TEXTS[2], topics, multi_label=multi_label)
        want = torch_pipe(TEXTS[2], topics, multi_label=multi_label)
        assert got["labels"] == want["labels"]
        assert all(_close(a, b) for a, b in zip(got["scores"], want["scores"]))


@needs_torch
def test_token_classification_matches_transformers():
    model = BERT_BASE_NER_CONF["DEFAULT_MODEL"]
    (ot, om), (tt, tm) = _pair(model, "AutoModelForTokenClassification")
    kwargs = {**model.pipeline_kwargs, "ignore_labels": ["O"]}
    onnx_pipe = transformers_helpers.pipeline("ner", om, ot, **kwargs)
    torch_pipe = transformers_helpers.pipeline("ner", tm, tt, **kwargs)
    text = "My name is Johnathan Doe and I work at Microsoft in Seattle with Angela Merkel."

    got, want = onnx_pipe(text), torch_pipe(text)
    assert [(e["entity_group"], e["start"], e["end"], e["word"]) for e in got] == [
        (e["entity_group"], e["start"], e["end"], e["word"]) for e in want
    ]
    assert all(_close(float(a["score"]), float(b["score"])) for a, b in zip(got, want))


@needs_torch
def test_embeddings_match_transformers():
    onnx_scanner = Relevance(model=MODEL_EN_BGE_SMALL, use_onnx=True)
    torch_scanner = Relevance(model=MODEL_EN_BGE_SMALL, use_onnx=False)
    prompt, output = "How do I bake bread?", "Mix flour, water, yeast and salt, then bake."
    got = onnx_scanner._encode(prompt).dot(onnx_scanner._encode(output))
    want = torch_scanner._encode(prompt).dot(torch_scanner._encode(output))
    assert _close(float(got), float(want))


def test_backend_falls_back_to_onnx_without_torch(monkeypatch):
    monkeypatch.delenv("GORGET_BACKEND", raising=False)
    monkeypatch.setattr(transformers_helpers, "is_torch_available", lambda: False)
    assert transformers_helpers.resolve_backend(use_onnx=False) == "onnx"


def test_backend_can_be_forced(monkeypatch):
    monkeypatch.setenv("GORGET_BACKEND", "onnx")
    assert transformers_helpers.resolve_backend(use_onnx=False) == "onnx"
    monkeypatch.setenv("GORGET_BACKEND", "torch")
    assert transformers_helpers.resolve_backend(use_onnx=True) == "torch"


def test_sessions_are_shared_between_scanners():
    first = runtime.load(MODEL_EN_BGE_SMALL)
    second = runtime.load(MODEL_EN_BGE_SMALL)
    assert first[1] is second[1]


def test_rate_limited_download_is_retried(monkeypatch):
    httpx = pytest.importorskip("httpx")
    errors = pytest.importorskip("huggingface_hub.errors")
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    request = httpx.Request("GET", "https://huggingface.co/api/models/org/model")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            response = httpx.Response(429, headers={"Retry-After": "5"}, request=request)
            raise errors.HfHubHTTPError("429 Too Many Requests", response=response)
        return "cached"

    hub = type("Hub", (), {"snapshot_download": staticmethod(snapshot_download)})
    assert str(runtime._snapshot_with_retry(hub, {"repo_id": "org/model"})) == "cached"
    assert len(calls) == 2
