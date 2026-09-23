The first release of Gorget, a maintained continuation of [LLM Guard](https://github.com/protectai/llm-guard), which Protect AI archived on July 8, 2026.

## Install

```bash
pip install gorget            # ONNX Runtime, no PyTorch
pip install "gorget[torch]"   # with PyTorch, e.g. for GPUs
```

Code written for LLM Guard keeps working: `from llm_guard.input_scanners import PromptInjection` returns the same class as `from gorget.input_scanners import PromptInjection`.

## Highlights

- **No PyTorch needed.** Every scanner runs on ONNX Runtime with Gorget's own pipelines, which return the same results as the transformers pipelines (compared in the test suite).
- **Python 3.10–3.14** and current dependencies: the vulnerable `transformers==4.51.3` and `presidio==2.2.358` pins are gone.
- **Nothing is installed while you scan.** spaCy models are no longer pip-installed at runtime, NLTK data is no longer downloaded, cached models load offline (`HF_HUB_OFFLINE=1`).
- **Russian personal data** with `language="ru"`: names and addresses, INN, SNILS and OGRN with control sums, passports, phone numbers.
- **Model licenses in the open.** [docs/models.md](https://github.com/Perruer/gorget/blob/main/docs/models.md) lists every model; the default `Anonymize` model inherited from LLM Guard is non-commercial, so Gorget warns about it and adds an Apache-2.0 alternative (`BERT_SMALL_GRAVITEE_PII_CONF`).
- **Fixes:** `MaliciousURLs` (its PyTorch model was deleted), `EmotionDetection` with ONNX, `Sensitive` ignoring `language`, `FactualConsistency` and `Relevance` importing PyTorch on import.
- **API image** `ghcr.io/perruer/gorget-api` runs on CPU without PyTorch; the `-cuda` tag uses PyTorch.

Full list: [changelog](https://github.com/Perruer/gorget/blob/main/docs/changelog.md).

## Support

Gorget is free. If it protects your product, you can support it on [Boosty](https://boosty.to/mikio_kuroki/donate) or with crypto (addresses in the [README](https://github.com/Perruer/gorget#support-the-project)).
