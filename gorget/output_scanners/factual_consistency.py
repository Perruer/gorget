from __future__ import annotations

from gorget import runtime
from gorget.input_scanners.ban_topics import MODEL_DEBERTA_BASE_V2
from gorget.model import Model
from gorget.transformers_helpers import get_tokenizer_and_model_for_classification
from gorget.util import calculate_risk_score, device, get_logger, lazy_load_dep

from .base import Scanner

LOGGER = get_logger()


class FactualConsistency(Scanner):
    """
    FactualConsistency Class:

    This class checks for entailment between a given prompt and output using a pretrained NLI model.
    """

    def __init__(
        self,
        *,
        model: Model | None = None,
        minimum_score=0.75,
        use_onnx=False,
    ) -> None:
        """
        Initializes an instance of the Refutation class.

        Parameters:
            model (Model, optional): The model to use for entailment checking. Defaults to None.
            minimum_score (float): The minimum entailment score for the output to be considered valid. Defaults to 0.75.
            use_onnx (bool): Whether to use the ONNX version of the model. Defaults to False.
        """

        self._minimum_score = minimum_score

        if model is None:
            model = MODEL_DEBERTA_BASE_V2

        self._tokenizer, self._model = get_tokenizer_and_model_for_classification(
            model=model,
            use_onnx=use_onnx,
        )
        if isinstance(self._model, runtime.OnnxModel):
            self._classifier = runtime.SequencePairClassifier(self._model, self._tokenizer)
        else:
            self._classifier = None
            self._model = self._model.to(device())
            self._model.eval()

    def _probabilities(self, output: str, prompt: str) -> list[float]:
        if self._classifier is not None:
            return self._classifier.probabilities(output, prompt).tolist()

        torch = lazy_load_dep("torch")
        tokenized_input_seq_pair = self._tokenizer(
            output, prompt, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized_input_seq_pair = {
            key: val.to(device()) for key, val in tokenized_input_seq_pair.items()
        }
        with torch.no_grad():
            model_output = self._model(**tokenized_input_seq_pair)
            return torch.softmax(model_output["logits"][0], -1).tolist()

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return output, True, -1.0

        model_prediction = self._probabilities(output, prompt)

        label_names = ["entailment", "not_entailment"]
        prediction = {
            name: round(float(pred), 2) for pred, name in zip(model_prediction, label_names)
        }

        entailment_score = prediction["entailment"]
        if entailment_score < self._minimum_score:
            LOGGER.warning("Entailment score is below the threshold", prediction=prediction)

            return (
                output,
                False,
                calculate_risk_score(prediction["not_entailment"], self._minimum_score),
            )

        LOGGER.debug("The output is factually consistent", prediction=prediction)

        return (
            output,
            True,
            calculate_risk_score(prediction["not_entailment"], self._minimum_score),
        )
