from __future__ import annotations

from collections.abc import Sequence

from presidio_analyzer import EntityRecognizer
from presidio_anonymizer import AnonymizerEngine

from gorget.exception import GorgetValidationError
from gorget.input_scanners.anonymize import (
    ALL_SUPPORTED_LANGUAGES,
    LANGUAGE_DEFAULT_RECOGNIZER_CONF,
    Anonymize,
    _entity_types,
)
from gorget.input_scanners.anonymize_helpers import (
    DEBERTA_AI4PRIVACY_v2_CONF,
    get_analyzer,
    get_recognizers,
    get_regex_patterns,
)
from gorget.input_scanners.anonymize_helpers.ner_mapping import NERConfig
from gorget.util import calculate_risk_score, get_logger

from ..input_scanners.anonymize_helpers.regex_patterns import (
    DefaultRegexPatterns,
    RegexPatternsReuse,
)
from .base import Scanner

LOGGER = get_logger()


class Sensitive(Scanner):
    """
    A class used to detect sensitive (PII) data in the output of a language model.

    This class uses the Presidio Analyzer Engine and predefined internally patterns (patterns.py) to analyze the output for specified entity types.
    If no entity types are specified, it defaults to checking for all entity types.
    """

    def __init__(
        self,
        *,
        entity_types: list[str] | None = None,
        regex_patterns: list[DefaultRegexPatterns | RegexPatternsReuse] | None = None,
        redact: bool = False,
        recognizer_conf: NERConfig | Sequence[NERConfig] | None = None,
        threshold: float = 0.5,
        use_onnx: bool = False,
        language: str = "en",
        recognizers: Sequence[EntityRecognizer] | None = None,
    ) -> None:
        """
        Initializes an instance of the Sensitive class.

        Parameters:
           entity_types (Optional[Sequence[str]]): The entity types to look for in the output. Defaults to the
                                               built-in types plus the custom types of your recognizers,
                                               NER models and regex patterns.
           regex_patterns (Optional[List[Dict]]): List of regex patterns to use for detection. Default is None.
           redact (bool): Redact found sensitive entities. Default to False.
           recognizer_conf (Optional[Dict]): Configuration of the NER model, or a list of them. Default is Ai4Privacy DeBERTa;
                                               an empty list runs no NER model.
           threshold (float): Acceptance threshold. Default is 0.
           use_onnx (bool): Use ONNX model for inference. Default is False.
           language (str): Language of the output. Default is "en".
           recognizers (Optional[Sequence]): Your own Presidio recognizers or `gorget.plugins.CallableRecognizer`s.
        """
        if language not in ALL_SUPPORTED_LANGUAGES:
            raise GorgetValidationError(
                f"Language must be in the list of allowed: {ALL_SUPPORTED_LANGUAGES}"
            )

        regex_groups = get_regex_patterns(regex_patterns)
        self._entity_types = _entity_types(
            entity_types,
            recognizer_conf=recognizer_conf,
            recognizers=recognizers,
            regex_groups=regex_groups,
        )
        self._redact = redact
        self._threshold = threshold
        self._language = language

        if recognizer_conf is None:
            recognizer_conf = LANGUAGE_DEFAULT_RECOGNIZER_CONF.get(
                language, DEBERTA_AI4PRIVACY_v2_CONF
            )

        self._analyzer = get_analyzer(
            get_recognizers(
                recognizer_conf=recognizer_conf,
                recognizers=recognizers,
                use_onnx=use_onnx,
                language=language,
            ),
            regex_groups,
            [],
            list(set(["en", language])),
        )
        self._anonymizer = AnonymizerEngine()

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if output.strip() == "":
            return prompt, True, -1.0

        analyzer_results = self._analyzer.analyze(
            text=Anonymize.remove_single_quotes(output),
            language=self._language,
            entities=self._entity_types,
            score_threshold=self._threshold,
        )

        if analyzer_results:
            if self._redact:
                LOGGER.debug("Redacting sensitive entities")
                result = self._anonymizer.anonymize(text=output, analyzer_results=analyzer_results)  # type: ignore
                output = result.text

            risk_score = round(
                max(analyzer_result.score for analyzer_result in analyzer_results), 2
            )
            LOGGER.warning("Found sensitive data in the output", results=analyzer_results)
            return output, False, calculate_risk_score(risk_score, self._threshold)

        LOGGER.debug("No sensitive data found in the output")
        return output, True, -1.0
