# Optimization Strategies

## ONNX Runtime

Gorget ships its own [ONNX Runtime](https://onnxruntime.ai) backend. It needs neither PyTorch nor
`optimum`, loads the ONNX exports of every scanner model and returns the same results as the
PyTorch pipelines (the test suite compares both). Without PyTorch installed it is used
automatically; with PyTorch installed, ask for it per scanner:

```python
scanner = Code(languages=["PHP"], use_onnx=True)
```

or for every scanner with `GORGET_BACKEND=onnx`. Scanners that use the same model share one
ONNX session.

Settings:

- `GORGET_ONNX_THREADS` limits the threads each session uses (for example, `1` per API worker).
- `GORGET_ONNX_PROVIDERS` picks execution providers, e.g. `CUDAExecutionProvider,CPUExecutionProvider`
  after replacing `onnxruntime` with `onnxruntime-gpu`.

## ONNX Runtime with Quantization

Although not built-in in the library, you can use quantized or optimized versions of the models.
However, that doesn't always lead to better latency but can reduce the model size.

## Enabling Low CPU/Memory Usage

To minimize CPU and memory usage:

```python
from gorget.input_scanners.code import Code, DEFAULT_MODEL

DEFAULT_MODEL.kwargs["low_cpu_mem_usage"] = True
scanner = Code(languages=["PHP"], model=DEFAULT_MODEL)
```

For an in-depth understanding of this feature and its impact on large model handling, refer to the detailed [Large Model Loading Documentation](https://huggingface.co/docs/transformers/main_classes/model#large-model-loading).

Alternatively, quantization can be used to reduce the model size and memory usage.

## Use smaller models

For certain scanners, smaller model variants are available e.g. distilbert, bert-small, bert-tiny versions.
These models are designed for enhanced performance, offering reduced latency without significantly compromising accuracy or effectiveness.

## PyTorch hacks

To speed up warm compile times:

```python
import torch
torch.set_float32_matmul_precision('high')

import torch._inductor.config
torch._inductor.config.fx_graph_cache = True
```

## Streaming mode

To optimize the output scanning, you can analyze the output in chunks. In [OpenAI](./openai.md) guide, we demonstrate how to use Gorget to protect OpenAI client with streaming.
