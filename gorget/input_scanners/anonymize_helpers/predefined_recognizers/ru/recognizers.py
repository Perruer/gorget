"""Recognizers for Russian personal data: INN, SNILS, OGRN, passport and phone numbers.

Numbers with a control sum are accepted only when the sum matches, which keeps random
digit strings from being taken for documents.
"""

from __future__ import annotations

import datetime
import re

from presidio_analyzer import Pattern, PatternRecognizer
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    CryptoRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    IpRecognizer,
)

from ..phone_recognizer import PhoneRecognizer as _BasePhoneRecognizer

_DIGITS = re.compile(r"\D")


def _digits(text: str) -> list[int]:
    return [int(c) for c in _DIGITS.sub("", text)]


def _weighted(digits: list[int], weights: list[int]) -> int:
    return sum(d * w for d, w in zip(digits, weights))


def is_valid_inn(text: str) -> bool:
    """Taxpayer number: 10 digits for companies, 12 for people, with control digits."""
    d = _digits(text)
    if len(d) == 10:
        return (_weighted(d, [2, 4, 10, 3, 5, 9, 4, 6, 8]) % 11) % 10 == d[9]
    if len(d) == 12:
        n11 = (_weighted(d, [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]) % 11) % 10
        n12 = (_weighted(d, [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]) % 11) % 10
        return n11 == d[10] and n12 == d[11]
    return False


def is_valid_snils(text: str) -> bool:
    """Insurance number: 9 digits and a 2-digit control sum (checked above 001-001-998)."""
    d = _digits(text)
    if len(d) != 11:
        return False
    if int("".join(map(str, d[:9]))) <= 1001998:
        return False
    total = _weighted(d[:9], [9, 8, 7, 6, 5, 4, 3, 2, 1])
    if total < 100:
        control = total
    elif total in (100, 101):
        control = 0
    else:
        control = total % 101
        if control == 100:
            control = 0
    return control == d[9] * 10 + d[10]


def is_valid_ogrn(text: str) -> bool:
    """Registration number: 13 digits (OGRN) or 15 digits (OGRNIP)."""
    d = _digits(text)
    number = "".join(map(str, d))
    if len(d) == 13:
        return (int(number[:12]) % 11) % 10 == d[12]
    if len(d) == 15:
        return (int(number[:14]) % 13) % 10 == d[14]
    return False


def is_plausible_passport(text: str) -> bool:
    """Passport series is a 2-digit region code and the last 2 digits of the issue year."""
    d = _digits(text)
    if len(d) != 10:
        return False
    region, year = d[0] * 10 + d[1], d[2] * 10 + d[3]
    this_year = datetime.date.today().year % 100
    return region > 0 and (year >= 97 or year <= this_year)


class RuInnRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern("INN (12 digits)", r"\b\d{12}\b", 0.3),
        Pattern("INN (10 digits)", r"\b\d{10}\b", 0.2),
    ]
    CONTEXT = ["инн", "налогоплательщика", "inn"]

    def __init__(self, supported_language: str = "ru", supported_entity: str = "RU_INN"):
        super().__init__(
            supported_entity=supported_entity,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        return is_valid_inn(pattern_text)


class RuSnilsRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern("SNILS (formatted)", r"\b\d{3}-\d{3}-\d{3}[ -]\d{2}\b", 0.5),
        Pattern("SNILS (digits)", r"\b\d{11}\b", 0.2),
    ]
    CONTEXT = ["снилс", "страховой", "пенсионного", "snils"]

    def __init__(self, supported_language: str = "ru", supported_entity: str = "RU_SNILS"):
        super().__init__(
            supported_entity=supported_entity,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        return is_valid_snils(pattern_text)


class RuOgrnRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern("OGRNIP", r"\b\d{15}\b", 0.2),
        Pattern("OGRN", r"\b\d{13}\b", 0.2),
    ]
    CONTEXT = ["огрн", "огрнип", "регистрационный", "ogrn"]

    def __init__(self, supported_language: str = "ru", supported_entity: str = "RU_OGRN"):
        super().__init__(
            supported_entity=supported_entity,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        return is_valid_ogrn(pattern_text)


class RuPassportRecognizer(PatternRecognizer):
    """Passport has no control sum, so it relies on context words to score high."""

    PATTERNS = [
        Pattern("Passport (series and number)", r"\b\d{2}\s?\d{2}\s?(?:№\s?)?\d{6}\b", 0.3),
    ]
    CONTEXT = ["паспорт", "паспорта", "паспортные", "серия", "серии", "выдан", "passport"]

    def __init__(self, supported_language: str = "ru", supported_entity: str = "RU_PASSPORT"):
        super().__init__(
            supported_entity=supported_entity,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        # None keeps the pattern score; context words raise it.
        return None if is_plausible_passport(pattern_text) else False


class RuPhoneRecognizer(_BasePhoneRecognizer):
    DEFAULT_SUPPORTED_REGIONS = ("RU", "BY", "KZ", "UA", "US", "DE", "GB")

    def __init__(self, supported_language: str = "ru"):
        super().__init__(
            context=["телефон", "тел", "моб", "мобильный", "позвонить", "звонить", "phone"],
            supported_language=supported_language,
            supported_regions=self.DEFAULT_SUPPORTED_REGIONS,
        )


def _multilingual(recognizer_class, context: list[str]):
    """Presidio's language-neutral recognizers, registered for Russian text."""

    def build(supported_language: str = "ru"):
        return recognizer_class(supported_language=supported_language, context=context)

    return build


RU_RECOGNIZERS = [
    RuInnRecognizer,
    RuSnilsRecognizer,
    RuOgrnRecognizer,
    RuPassportRecognizer,
    RuPhoneRecognizer,
    _multilingual(EmailRecognizer, ["почта", "email", "e-mail", "адрес"]),
    _multilingual(IpRecognizer, ["ip", "адрес", "сервер"]),
    _multilingual(CreditCardRecognizer, ["карта", "карты", "картой", "visa", "mastercard", "мир"]),
    _multilingual(IbanRecognizer, ["iban", "счёт", "счет"]),
    _multilingual(CryptoRecognizer, ["кошелёк", "кошелек", "биткоин", "btc"]),
]
