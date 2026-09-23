from __future__ import annotations

import numpy as np

from gorget import runtime
from gorget.model import Model
from gorget.transformers_helpers import get_tokenizer_and_model_for_embeddings
from gorget.util import calculate_risk_score, device, get_logger, lazy_load_dep

from .base import Scanner

LOGGER = get_logger()

MODEL_EN_BGE_BASE = Model(
    path="BAAI/bge-base-en-v1.5",
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    onnx_path="BAAI/bge-base-en-v1.5",
    onnx_subfolder="onnx",
    onnx_filename="model.onnx",
    onnx_revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
)
MODEL_EN_BGE_LARGE = Model(
    path="BAAI/bge-large-en-v1.5",
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    onnx_path="BAAI/bge-large-en-v1.5",
    onnx_subfolder="onnx",
    onnx_revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
)
MODEL_EN_BGE_SMALL = Model(
    path="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    onnx_path="BAAI/bge-small-en-v1.5",
    onnx_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    onnx_subfolder="onnx",
)


class Relevance(Scanner):
    """
    A class used to scan the relevance of the output of a language model to the input prompt.

    This class encodes the prompt and output into vector embeddings, then computes
    the cosine similarity between them. If the similarity is below a given threshold, the output is considered
    not relevant to the prompt.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        model: Model | None = None,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes an instance of the Relevance class.

        Parameters:
            threshold: The minimum similarity score to compare prompt and output.
            model: Model for calculating embeddings. Default is `BAAI/bge-base-en-v1.5`.
            use_onnx: Whether to use the ONNX version of the model. Defaults to False.
        """

        self._threshold = threshold

        if model is None:
            model = MODEL_EN_BGE_BASE

        self.pooling_method = "cls"
        self.normalize_embeddings = True

        self._tokenizer, self._model = get_tokenizer_and_model_for_embeddings(
            model=model,
            use_onnx=use_onnx,
        )
        if isinstance(self._model, runtime.OnnxModel):
            self._embedder = runtime.FeatureExtractionPipeline(
                self._model, self._tokenizer, max_length=512
            )
        else:
            self._embedder = None
            self._model = self._model.to(device())
            self._model.eval()

    def pooling(self, last_hidden_state, attention_mask):
        if self.pooling_method == "cls":
            return last_hidden_state[:, 0]
        elif self.pooling_method == "mean":
            torch = lazy_load_dep("torch")
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        return None

    def _encode(self, sentence: str, max_length: int = 512) -> np.ndarray:
        if self._embedder is not None:
            return self._embedder.embed(
                [sentence], pooling=self.pooling_method, normalize=self.normalize_embeddings
            )[0]

        torch = lazy_load_dep("torch")
        inputs = self._tokenizer(
            [sentence],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        inputs = {key: val.to(device()) for key, val in inputs.items()}

        with torch.no_grad():
            last_hidden_state = self._model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs["attention_mask"])
            assert embeddings is not None
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            embeddings = embeddings.cpu().numpy()

        return embeddings[0]

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if output.strip() == "":
            return output, True, -1.0

        prompt_embedding = self._encode(prompt)
        output_embedding = self._encode(output)
        similarity = prompt_embedding.dot(output_embedding.T)

        if similarity < self._threshold:
            LOGGER.warning("Result is not similar to the prompt", similarity_score=similarity)

            return output, False, calculate_risk_score(1 - similarity, self._threshold)

        LOGGER.debug("Result is similar to the prompt", similarity_score=similarity)

        return output, True, calculate_risk_score(1 - similarity, self._threshold)
