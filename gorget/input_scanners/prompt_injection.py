from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING

from gorget.exception import GorgetValidationError
from gorget.model import Model
from gorget.plugins import TextClassifier, normalize_label_scores
from gorget.transformers_helpers import get_tokenizer_and_model_for_classification, pipeline
from gorget.util import (
    calculate_risk_score,
    get_logger,
    split_text_by_sentences,
    split_text_to_word_chunks,
    truncate_tokens_head_tail,
)

from .base import Scanner

LOGGER = get_logger()

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

PROMPT_CHARACTERS_LIMIT = 256

# This model is proprietary but open source.
V1_MODEL = Model(
    path="protectai/deberta-v3-base-prompt-injection",
    revision="f51c3b2a5216ae1af467b511bc7e3b78dc4a99c9",
    onnx_path="ProtectAI/deberta-v3-base-prompt-injection",
    onnx_revision="f51c3b2a5216ae1af467b511bc7e3b78dc4a99c9",
    onnx_subfolder="onnx",
    onnx_filename="model.onnx",
    pipeline_kwargs={
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
    },
)

V2_MODEL = Model(
    path="protectai/deberta-v3-base-prompt-injection-v2",
    revision="89b085cd330414d3e7d9dd787870f315957e1e9f",
    onnx_path="ProtectAI/deberta-v3-base-prompt-injection-v2",
    onnx_revision="89b085cd330414d3e7d9dd787870f315957e1e9f",
    onnx_subfolder="onnx",
    onnx_filename="model.onnx",
    pipeline_kwargs={
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
    },
)

# This is gated model, which requires our approval.
V2_SMALL_MODEL = Model(
    path="protectai/deberta-v3-small-prompt-injection-v2",
    revision="3897fe66d47d2e1649e669c3470bf2ba3ddecc22",
    onnx_path="protectai/deberta-v3-small-prompt-injection-v2",
    onnx_revision="3897fe66d47d2e1649e669c3470bf2ba3ddecc22",
    onnx_subfolder="onnx",
    onnx_filename="model.onnx",
    pipeline_kwargs={
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
    },
    tokenizer_kwargs={
        "use_fast": False,
        "token": True,
    },
    kwargs={"token": True},  # You can also configure with your token.
)


class MatchType(Enum):
    SENTENCE = "sentence"
    FULL = "full"
    # TRUNCATE_TOKEN_HEAD_TAIL is used to split the prompt into two parts (126 head and 382 tail) and check them.
    TRUNCATE_TOKEN_HEAD_TAIL = "truncate_token_head_tail"
    # TRUNCATE_SIDES is used to split the prompt into two parts (256 head and 256 tail) and check them.
    TRUNCATE_HEAD_TAIL = "truncate_head_tail"
    CHUNKS = "chunks"

    _tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def get_inputs(self, prompt: str) -> list[str]:
        if self == MatchType.SENTENCE:
            return split_text_by_sentences(prompt)

        if self == MatchType.CHUNKS:
            chunks = []
            for chunk_start, chunk_end in split_text_to_word_chunks(
                len(prompt), chunk_length=PROMPT_CHARACTERS_LIMIT, overlap_length=25
            ):
                chunks.append(prompt[chunk_start:chunk_end])

            return chunks

        tokenizer = getattr(self, "_tokenizer", None)
        if self == MatchType.TRUNCATE_TOKEN_HEAD_TAIL and tokenizer is not None:
            tokenized_input = tokenizer.tokenize(prompt)

            return [tokenizer.convert_tokens_to_string(truncate_tokens_head_tail(tokenized_input))]

        if self == MatchType.TRUNCATE_HEAD_TAIL and len(prompt) > PROMPT_CHARACTERS_LIMIT:
            part_length = (PROMPT_CHARACTERS_LIMIT - 3) // 2

            start = prompt[:part_length]
            end = prompt[-part_length:]

            return [f"{start}...{end}"]

        return [prompt]


DEFAULT_INJECTION_LABELS = ("INJECTION",)


@dataclasses.dataclass(frozen=True)
class InjectionDetection:
    """
    Result of `PromptInjection.detect`.

    Attributes:
        score: Injection probability of the most suspicious part of the text, 0 to 1.
        label: Injection category with the highest score, or None when the classifier
            only said the text is safe.
        is_injection: Whether the score is above the scanner threshold.
    """

    score: float
    label: str | None
    is_injection: bool


class PromptInjection(Scanner):
    """
    A prompt injection scanner based on HuggingFace model. It is used to
    detect if a prompt is attempting to perform an injection attack.

    The built-in model can be replaced with any Hugging Face or local model (`model`),
    or with your own classifier (`classifier`), and `injection_labels` tells which of its
    labels mean an attack. With several injection labels the scanner adds up their scores
    and reports the most likely category.
    """

    def __init__(
        self,
        *,
        model: Model | None = None,
        threshold: float = 0.92,
        match_type: MatchType | str = MatchType.FULL,
        use_onnx: bool = False,
        injection_labels: Sequence[str] | None = None,
        classifier: TextClassifier | None = None,
    ) -> None:
        """
        Initializes PromptInjection with a threshold.

        Parameters:
            model (Model, optional): Chosen model to classify prompt. Default is the Protect AI DeBERTa v3 (v2 weights) model.
            threshold (float): Threshold for the injection score. Default is 0.92.
            match_type (MatchType): Whether to match the full text or individual sentences. Default is MatchType.FULL.
            use_onnx (bool): Whether to use ONNX for inference. Defaults to False.
            injection_labels (Sequence[str], optional): Labels of the classifier that mean an injection,
                compared case-insensitively. Default is ["INJECTION"].
            classifier (TextClassifier, optional): Your own classifier, used instead of `model`: a callable
                that takes a list of texts and returns labels with scores for each (see `gorget.plugins`).

        Raises:
            ValueError: If non-existent models were provided.
        """
        if isinstance(match_type, str):
            match_type = MatchType(match_type)

        if model is not None and classifier is not None:
            raise GorgetValidationError("Pass either model or classifier, not both")

        self._threshold = threshold
        self._injection_labels = {
            label.upper() for label in (injection_labels or DEFAULT_INJECTION_LABELS)
        }
        if not self._injection_labels:
            raise GorgetValidationError("injection_labels must not be empty")

        if classifier is not None:
            self._model = None
            self._classify = classifier
        else:
            if model is None:
                model = V2_MODEL
            self._model = model

            tf_tokenizer, tf_model = get_tokenizer_and_model_for_classification(
                model=model,
                use_onnx=use_onnx,
            )

            text_pipeline = pipeline(
                task="text-classification",
                model=tf_model,
                tokenizer=tf_tokenizer,
                **model.pipeline_kwargs,
            )
            self._classify = lambda texts: text_pipeline(texts, top_k=None)
            match_type.set_tokenizer(tf_tokenizer)

        self._match_type = match_type

    def _injection_score(self, result) -> tuple[float, str | None]:
        pairs = normalize_label_scores(result)
        injections = [
            (label, score) for label, score in pairs if label.upper() in self._injection_labels
        ]

        if len(pairs) == 1:
            # Only the top label is known: a safe label with score s means injection 1 - s.
            label, score = pairs[0]
            if injections:
                return score, label
            return 1 - score, None

        if not injections:
            return 0.0, None

        category = max(injections, key=lambda pair: pair[1])[0]
        total = sum(score for _, score in pairs)
        if abs(total - 1) <= 0.02:
            # Softmax over classes: the chance of any injection category is their sum.
            return min(1.0, sum(score for _, score in injections)), category
        # Independent (sigmoid) labels: the strongest one decides.
        return max(score for _, score in injections), category

    def detect(self, prompt: str) -> InjectionDetection:
        """Classify a prompt and return the injection score and category."""
        if prompt.strip() == "":
            return InjectionDetection(score=0.0, label=None, is_injection=False)

        inputs = self._match_type.get_inputs(prompt)
        best_score, best_label = 0.0, None
        for result in self._classify(inputs):
            score, label = self._injection_score(result)
            score = round(score, 2)
            if score > best_score or (
                score == best_score and best_label is None and label is not None
            ):
                best_score, best_label = score, label
            if score > self._threshold:
                break

        return InjectionDetection(
            score=best_score,
            label=best_label,
            is_injection=best_score > self._threshold,
        )

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        detection = self.detect(prompt)
        if detection.is_injection:
            LOGGER.warning(
                "Detected prompt injection",
                injection_score=detection.score,
                category=detection.label,
            )
            return prompt, False, calculate_risk_score(detection.score, self._threshold)

        LOGGER.debug("No prompt injection detected", highest_score=detection.score)

        return prompt, True, calculate_risk_score(detection.score, self._threshold)
