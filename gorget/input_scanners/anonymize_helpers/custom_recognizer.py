"""Presidio recognizer around a plain entity detector (see `gorget.plugins`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from presidio_analyzer import AnalysisExplanation, EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from gorget.exception import GorgetValidationError
from gorget.plugins import Entity, EntityDetector, EntityLike


def _to_entity(item: EntityLike) -> Entity:
    if isinstance(item, Entity):
        return item
    if isinstance(item, dict):
        return Entity(
            entity_type=item["entity_type"],
            start=int(item["start"]),
            end=int(item["end"]),
            score=float(item.get("score", 1.0)),
        )
    entity_type, start, end, *rest = item
    return Entity(entity_type, int(start), int(end), float(rest[0]) if rest else 1.0)


class CallableRecognizer(EntityRecognizer):
    """
    Presidio recognizer around an entity detector, so that a plain function or a client
    of an internal NER service can be passed to `Anonymize` and `Sensitive`.

    Example:
        ```python
        import re
        from gorget.plugins import CallableRecognizer, Entity

        def contracts(text, language):
            for m in re.finditer(r"\\bCTR-\\d{6}\\b", text):
                yield Entity("CONTRACT_NUMBER", m.start(), m.end(), 0.9)

        recognizer = CallableRecognizer(contracts, entities=["CONTRACT_NUMBER"])
        ```
    """

    def __init__(
        self,
        detector: EntityDetector,
        *,
        entities: Sequence[str],
        name: str | None = None,
        supported_language: str = "en",
    ) -> None:
        if not entities:
            raise GorgetValidationError("CallableRecognizer needs the entity types it can find")
        self._detector = detector
        super().__init__(
            supported_entities=list(entities),
            name=name or getattr(detector, "__name__", type(detector).__name__),
            supported_language=supported_language,
        )

    def load(self) -> None:
        pass

    def for_language(self, language: str) -> CallableRecognizer:
        """Return the same detector registered for another language."""
        return CallableRecognizer(
            self._detector,
            entities=self.supported_entities,
            name=self.name,
            supported_language=language,
        )

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        results = []
        for item in self._detector(text, self.supported_language) or []:
            entity = _to_entity(item)
            if entities and entity.entity_type not in entities:
                continue
            if not 0 <= entity.start < entity.end <= len(text):
                continue
            results.append(
                RecognizerResult(
                    entity_type=entity.entity_type,
                    start=entity.start,
                    end=entity.end,
                    score=entity.score,
                    analysis_explanation=AnalysisExplanation(
                        recognizer=self.name,
                        original_score=entity.score,
                        textual_explanation=f"Found by {self.name}",
                    ),
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                )
            )
        return results


def check_recognizer(obj: Any) -> EntityRecognizer:
    """Fail early with a clear message when something other than a recognizer is passed."""
    if not isinstance(obj, EntityRecognizer):
        raise GorgetValidationError(
            "Recognizers must be Presidio EntityRecognizer objects or CallableRecognizer; "
            f"got {type(obj).__name__}. Wrap plain functions with CallableRecognizer."
        )
    return obj


def as_recognizer(obj: Any, language: str) -> EntityRecognizer:
    """
    Accept a Presidio recognizer or a `CallableRecognizer` and make sure it works for
    ``language`` (Presidio only runs recognizers registered for the analysed language).
    """
    check_recognizer(obj)
    if isinstance(obj, CallableRecognizer):
        return obj if obj.supported_language == language else obj.for_language(language)
    if obj.supported_language != language:
        raise GorgetValidationError(
            f"Recognizer {obj.name} supports language {obj.supported_language!r}, "
            f"but the scanner analyses {language!r}"
        )
    return obj
