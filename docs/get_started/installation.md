# Installing Gorget

## Prerequisites

Supported Python versions: 3.10 to 3.14.

## Using `pip`

!!! note

    Consider installing Gorget in a virtual environment like `venv` or `conda`.

```bash
pip install gorget
```

This installs Gorget with [ONNX Runtime](https://onnxruntime.ai) and without PyTorch: every
scanner runs on CPU, and the install is several times smaller than one that pulls PyTorch.

To run models with PyTorch instead, for example on a GPU, add the `torch` extra:

```bash
pip install "gorget[torch]"
```

With PyTorch installed, scanners use it unless you pass `use_onnx=True`. The `GORGET_BACKEND`
environment variable (`onnx` or `torch`) overrides the choice for every scanner.

## Moving from LLM Guard

Gorget continues LLM Guard, so existing code keeps working:

```bash
pip uninstall llm-guard
pip install gorget
```

```python
from llm_guard.input_scanners import PromptInjection  # still works
from gorget.input_scanners import PromptInjection  # the new name
```

Both imports return the same classes. `LLMGuardValidationError` is still available as an alias of
`GorgetValidationError`.

## No downloads at runtime

Gorget does not install anything while it runs. Models come from the Hugging Face Hub on first use
and are cached; spaCy models and NLTK data are no longer downloaded. To run offline, fill the cache
once and set `HF_HUB_OFFLINE=1`.

## Install from source

```bash
git clone https://github.com/Perruer/gorget.git
cd gorget
python -m venv venv
source venv/bin/activate
python -m pip install -e ".[dev,torch]"
```
