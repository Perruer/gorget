import re

import pytest

from gorget.exception import GorgetValidationError
from gorget.input_scanners import Anonymize, PromptInjection
from gorget.input_scanners.anonymize_helpers import make_ner_config
from gorget.input_scanners.anonymize_helpers.analyzer import get_custom_entity_types
from gorget.model import Model
from gorget.output_scanners import Sensitive
from gorget.plugins import CallableRecognizer, Entity, build_object, load_object
from gorget.vault import Vault


def contracts(text, language):
    for match in re.finditer(r"\bCTR-\d{6}\b", text):
        yield Entity("CONTRACT_NUMBER", match.start(), match.end(), 0.9)


CONTRACTS = CallableRecognizer(contracts, entities=["CONTRACT_NUMBER"])


class CategoryClassifier:
    """Softmax over three classes, like a model that sorts injections into categories."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return [[{"label": k, "score": v} for k, v in self.scores.items()] for _ in texts]


# PromptInjection with your own classifier


def test_injection_categories_add_up():
    classifier = CategoryClassifier({"benign": 0.1, "jailbreak": 0.5, "prompt_leak": 0.4})
    scanner = PromptInjection(
        classifier=classifier, injection_labels=["jailbreak", "PROMPT_LEAK"], threshold=0.8
    )
    detection = scanner.detect("anything")
    assert detection.score == 0.9
    assert detection.label == "jailbreak"
    assert detection.is_injection
    assert scanner.scan("anything")[1] is False


def test_top_label_only_classifier_inverts_safe_label():
    scanner = PromptInjection(
        classifier=lambda texts: [("LABEL_0", 0.97) for _ in texts],
        injection_labels=["LABEL_1"],
    )
    detection = scanner.detect("hello")
    assert detection.score == 0.03
    assert detection.label is None
    assert scanner.scan("hello")[1] is True


def test_independent_labels_take_the_strongest():
    scanner = PromptInjection(
        classifier=lambda texts: [
            [("jailbreak", 0.7), ("leak", 0.6), ("safe", 0.9)] for _ in texts
        ],
        injection_labels=["jailbreak", "leak"],
        threshold=0.65,
    )
    assert scanner.detect("x") == scanner.detect("x")
    assert scanner.detect("x").score == 0.7


def test_non_injection_labels_give_zero():
    scanner = PromptInjection(classifier=CategoryClassifier({"safe": 0.6, "spam": 0.4}))
    assert scanner.detect("x").score == 0.0
    assert scanner.scan("x") == ("x", True, -1.0)


def test_model_and_classifier_are_exclusive():
    with pytest.raises(GorgetValidationError):
        PromptInjection(model=Model(path="x"), classifier=CategoryClassifier({}))


def test_empty_prompt_skips_classifier():
    classifier = CategoryClassifier({"INJECTION": 1.0})
    scanner = PromptInjection(classifier=classifier)
    assert scanner.scan("  ") == ("  ", True, -1.0)
    assert classifier.calls == 0


# Your own recognizers in Anonymize and Sensitive


def test_anonymize_finds_custom_type_by_default():
    scanner = Anonymize(Vault(), recognizer_conf=[], recognizers=[CONTRACTS])
    sanitized, valid, _ = scanner.scan("Contract CTR-123456 for john@example.com")
    assert sanitized == "Contract [REDACTED_CONTRACT_NUMBER_1] for [REDACTED_EMAIL_ADDRESS_1]"
    assert not valid


def test_anonymize_respects_explicit_entity_types_and_keeps_the_list():
    entity_types = ["EMAIL_ADDRESS"]
    scanner = Anonymize(
        Vault(), recognizer_conf=[], recognizers=[CONTRACTS], entity_types=entity_types
    )
    sanitized, _, _ = scanner.scan("Contract CTR-123456 for john@example.com")
    assert sanitized == "Contract CTR-123456 for [REDACTED_EMAIL_ADDRESS_1]"
    assert entity_types == ["EMAIL_ADDRESS"]


def test_callable_recognizer_follows_the_scanner_language():
    scanner = Anonymize(Vault(), recognizer_conf=[], recognizers=[CONTRACTS], language="ru")
    sanitized, _, _ = scanner.scan("Договор CTR-123456 подписан")
    assert sanitized == "Договор [REDACTED_CONTRACT_NUMBER_1] подписан"


def test_sensitive_redacts_custom_type():
    scanner = Sensitive(recognizer_conf=[], recognizers=[CONTRACTS], redact=True)
    output, valid, _ = scanner.scan("", "Your contract is CTR-654321")
    assert output == "Your contract is <CONTRACT_NUMBER>"
    assert not valid


def test_detector_results_outside_the_text_are_dropped():
    recognizer = CallableRecognizer(
        lambda text, language: [
            ("CONTRACT_NUMBER", 0, 999),
            {"entity_type": "X", "start": 2, "end": 1},
        ],
        entities=["CONTRACT_NUMBER", "X"],
    )
    assert recognizer.analyze("short", entities=[]) == []


def test_recognizers_must_be_presidio_objects():
    with pytest.raises(GorgetValidationError):
        Anonymize(Vault(), recognizer_conf=[], recognizers=[contracts])


def test_custom_entity_types_skip_builtin_ones():
    conf = make_ner_config(
        "acme/ner", mapping={"CONTRACT": "CONTRACT_NUMBER", "PER": "PERSON", "MISC": "O"}
    )
    assert conf["PRESIDIO_SUPPORTED_ENTITIES"] == ["CONTRACT_NUMBER", "PERSON"]
    custom = get_custom_entity_types(
        recognizer_conf=[conf],
        recognizers=[CONTRACTS],
        regex_groups=[{"name": "TICKET_ID"}, {"name": "UUID"}],
    )
    assert custom == ["CONTRACT_NUMBER", "TICKET_ID"]


# Loading plugins from configuration


def test_load_object_by_import_path():
    assert load_object("tests.test_plugins:contracts") is contracts
    assert load_object("gorget.plugins:Entity.__name__") == "Entity"


def test_build_object_calls_classes_and_factories():
    recognizer = build_object(
        {
            "class": "gorget.plugins:CallableRecognizer",
            "params": {"detector": contracts, "entities": ["CONTRACT_NUMBER"]},
        }
    )
    assert recognizer.supported_entities == ["CONTRACT_NUMBER"]
    assert build_object("tests.test_plugins:CONTRACTS") is CONTRACTS
    assert build_object({"ref": "tests.test_plugins:contracts"}) is contracts

    nested = build_object(
        {
            "class": "gorget.plugins:CallableRecognizer",
            "params": {
                "detector": {"ref": "tests.test_plugins:contracts"},
                "entities": ["CONTRACT_NUMBER"],
                "name": "http://not-a-reference",
            },
        }
    )
    assert nested.name == "http://not-a-reference"
    assert nested.analyze("CTR-000001", entities=[])[0].entity_type == "CONTRACT_NUMBER"


@pytest.mark.parametrize(
    "reference", ["no_such_module_xyz:thing", "gorget.plugins:missing", "nope"]
)
def test_bad_references_are_reported(reference):
    with pytest.raises(GorgetValidationError):
        load_object(reference)
