from typing import Callable

from presidio_analyzer import EntityRecognizer


def _get_predefined_recognizers(language: str) -> list[Callable[..., EntityRecognizer]]:
    if language == "zh":
        from .zh import CryptoRecognizer, EmailRecognizer, IpRecognizer, PhoneRecognizer

        return [
            CryptoRecognizer,
            PhoneRecognizer,
            EmailRecognizer,
            IpRecognizer,
        ]
    if language == "ru":
        from .ru import RU_RECOGNIZERS

        return list(RU_RECOGNIZERS)

    return []


__all__ = [
    "_get_predefined_recognizers",
]
