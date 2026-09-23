"""Russian personal data: control sums and recognition in text (no NER model needed)."""

import pytest
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry

from gorget.input_scanners.anonymize_helpers.analyzer import _get_nlp_engine
from gorget.input_scanners.anonymize_helpers.predefined_recognizers.ru import (
    RU_RECOGNIZERS,
    is_plausible_passport,
    is_valid_inn,
    is_valid_ogrn,
    is_valid_snils,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("7707083893", True),  # Sberbank
        ("7707083894", False),
        ("500100732259", True),
        ("500100732258", False),
        ("12345", False),
    ],
)
def test_inn_control_sum(value, expected):
    assert is_valid_inn(value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("112-233-445 95", True),
        ("11223344595", True),
        ("112-233-445 94", False),
        ("001-001-998 00", False),
    ],
)
def test_snils_control_sum(value, expected):
    assert is_valid_snils(value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1027700132195", True),  # Sberbank OGRN
        ("1027700132196", False),
        ("304500116000157", True),
        ("304500116000158", False),
    ],
)
def test_ogrn_control_sum(value, expected):
    assert is_valid_ogrn(value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("45 06 123456", True),
        ("4506 123456", True),
        ("0006 123456", False),
        ("4550 123456", False),
    ],
)
def test_passport_plausibility(value, expected):
    assert is_plausible_passport(value) is expected


@pytest.fixture(scope="module")
def analyzer():
    registry = RecognizerRegistry(supported_languages=["ru"])
    for recognizer in RU_RECOGNIZERS:
        registry.add_recognizer(recognizer(supported_language="ru"))
    return AnalyzerEngine(
        nlp_engine=_get_nlp_engine(["ru"]), registry=registry, supported_languages=["ru"]
    )


def _found(analyzer, text: str) -> dict[str, str]:
    results = analyzer.analyze(text=text, language="ru", score_threshold=0.5)
    return {r.entity_type: text[r.start : r.end] for r in results}


def test_documents_are_found_in_russian_text(analyzer):
    text = (
        "Меня зовут Иван. Мой ИНН 500100732259, СНИЛС 112-233-445 95, "
        "паспорт серии 45 06 номер 123456, телефон +7 912 345-67-89, почта ivan@example.ru."
    )
    found = _found(analyzer, text)
    assert found["RU_INN"] == "500100732259"
    assert found["RU_SNILS"] == "112-233-445 95"
    assert found["PHONE_NUMBER"] == "+7 912 345-67-89"
    assert found["EMAIL_ADDRESS"] == "ivan@example.ru"


def test_passport_needs_context(analyzer):
    assert "RU_PASSPORT" in _found(analyzer, "Паспорт 4506 123456 выдан в 2006 году.")
    assert "RU_PASSPORT" not in _found(analyzer, "Заказ 4506 123456 отправлен.")


def test_random_numbers_are_not_documents(analyzer):
    found = _found(analyzer, "Номер заказа 1234567890, трек 12345678901.")
    assert not {"RU_INN", "RU_SNILS", "RU_OGRN"} & set(found)
