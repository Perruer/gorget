# Anonymize Scanner

The `Anonymize` Scanner acts as your digital guardian, ensuring your user prompts remain confidential and free from
sensitive data exposure.

## What is PII?

PII, an acronym for Personally Identifiable Information, is the cornerstone of an individual's digital identity. Leaks
or mishandling of PII can unleash a storm of problems, from privacy breaches to identity theft. Global regulations,
including GDPR and HIPAA, underscore the significance of PII by laying out strict measures for its protection.
Furthermore, any unintentional dispatch of PII to LLMs can proliferate this data across various storage points, thus
raising the stakes.

## Attack scenario

Some model providers may train their models on your requests, which can be a privacy concern. Use the scanner to ensure PII is not leaked to the model provider.

## PII entities

- **Credit Cards**: Formats mentioned in [Wikipedia](https://en.wikipedia.org/wiki/Payment_card_number).
    - `4111111111111111`
    - `378282246310005` (American Express)
    - `30569309025904` (Diners Club)
- **Person**: A full person name, which can include first names, middle names or initials, and last names.
    - `John Doe`
- **PHONE_NUMBER**:
    - `5555551234`
- **URL**: A URL (Uniform Resource Locator), unique identifier used to locate a resource on the Internet.
    - `https://protectai.com/`
- **E-mail Addresses**: Standard email formats.
    - `john.doe@protectai.com`
    - `john.doe[AT]protectai[DOT]com`
    - `john.doe[AT]protectai.com`
    - `john.doe@protectai[DOT]com`
- **IPs**: An Internet Protocol (IP) address (either IPv4 or IPv6).
    - `192.168.1.1` (IPv4)
    - `2001:db8:3333:4444:5555:6666:7777:8888` (IPv6)
- **UUID**:
    - `550e8400-e29b-41d4-a716-446655440000`
- **US Social Security Number (SSN)**:
    - `111-22-3333`
- **Crypto wallet number**: Currently only Bitcoin address is supported.
    - `1Lbcfr7sAHTD9CgdQo3HTMTkV8LK4ZnX71`
- **IBAN Code**: The International Bank Account Number (IBAN) is an internationally agreed system of identifying bank
  accounts across national borders to facilitate the communication and processing of cross border transactions with a
  reduced risk of transcription errors.
    - `DE89370400440532013000`

## Features

- **Integration with [Presidio Analyzer](https://github.com/microsoft/presidio/)**: Leverages the Presidio Analyzer
  library, crafted with spaCy, flair and transformers libraries, for precise detection of private data.
- **Enhanced Detection**: Beyond Presidio Analyzer's capabilities, the scanner recognizes specific patterns like Email,
  US SSN, UUID, and more.
- **Entities support**:
    - Peek at
    our [default entities](https://github.com/Perruer/gorget/blob/main/gorget/input_scanners/anonymize.py#L26-L40).
    - View
    the [Presidio's supported entities](https://microsoft.github.io/presidio/supported_entities/#list-of-supported-entities).
    - And, we've
    got [custom regex patterns](https://github.com/Perruer/gorget/blob/main/gorget/resources/sensisitive_patterns.json)
    too!
- **Tailored recognizers**:
    - Balance speed vs. accuracy of the recognizers.
    - **Top Pick: [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER)**
    - Alternative with more parameters: [dslim/bert-large-NER](https://huggingface.co/dslim/bert-large-NER).
    - Chinese recognizer: [gyr66/bert-base-chinese-finetuned-ner](https://huggingface.co/gyr66/bert-base-chinese-finetuned-ner).
    - Russian recognizer: [Gherman/bert-base-NER-Russian](https://huggingface.co/Gherman/bert-base-NER-Russian), used by default for `language="ru"`.
    - Commercial-friendly PII model (Apache-2.0): [gravitee-io/bert-small-pii-detection](https://huggingface.co/gravitee-io/bert-small-pii-detection) as `BERT_SMALL_GRAVITEE_PII_CONF`.
    - Models from AI4Privacy: [Isotonic/distilbert_finetuned_ai4privacy_v2](https://huggingface.co/Isotonic/distilbert_finetuned_ai4privacy_v2) and [Isotonic/deberta-v3-base_finetuned_ai4privacy_v2](https://huggingface.co/Isotonic/deberta-v3-base_finetuned_ai4privacy_v2).
- **Support of multiple languages**: English, Chinese and Russian.
- **Your own data types and models**: pass `recognizers` (Presidio recognizers or any function wrapped
  in `CallableRecognizer`), several NER models in `recognizer_conf`, or your own model through
  `make_ner_config`. Custom types like `CONTRACT_NUMBER` are detected by default. See
  [Your own models](../customization/custom_models.md).

!!! warning "Model licenses"

    The default English model, `Isotonic/deberta-v3-base_finetuned_ai4privacy_v2`, is licensed
    CC-BY-NC-4.0, which does not allow commercial use. LLM Guard used it without saying so; Gorget
    keeps it as the default for compatibility and logs a warning once. For a commercial product pass
    `recognizer_conf=BERT_SMALL_GRAVITEE_PII_CONF` (Apache-2.0) or `BERT_BASE_NER_CONF` (MIT).

### Russian personal data

With `language="ru"` the scanner finds Russian names and addresses with a Russian NER model and
recognizes documents by their format and control sums:

| Entity | What | Check |
|--------|------|-------|
| `RU_INN` | Taxpayer number, 10 or 12 digits | control digits |
| `RU_SNILS` | Insurance number, `123-456-789 01` | control sum |
| `RU_OGRN` | Registration number, 13 or 15 digits | control digit |
| `RU_PASSPORT` | Passport series and number | region and year in the series; needs context such as «паспорт» |
| `PHONE_NUMBER` | Russian and CIS phone numbers | `phonenumbers` |

E-mail, IP addresses, bank cards, IBAN and crypto wallets are recognized as in English text.

```python
scanner = Anonymize(vault, language="ru")
sanitized_prompt, is_valid, risk_score = scanner.scan(
    "Меня зовут Иван Петров, ИНН 500100732259, СНИЛС 112-233-445 95."
)
# Меня зовут [REDACTED_PERSON_1], ИНН [REDACTED_RU_INN_1], СНИЛС [REDACTED_RU_SNILS_1].
```

## Get started

Initialize the `Vault`: The Vault archives data that's been redacted.

```python
from gorget.vault import Vault

vault = Vault()
```

Configure the `Anonymize` Scanner:

```python
from gorget.input_scanners import Anonymize
from gorget.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF

scanner = Anonymize(vault, preamble="Insert before prompt", allowed_names=["John Doe"], hidden_names=["Test LLC"],
                    recognizer_conf=BERT_LARGE_NER_CONF, language="en")
sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
```

- `preamble`: Directs the LLM to bypass specific content.
- `hidden_names`: Transforms specified names to formats like `[REDACTED_CUSTOM_1]`.
- `entity_types`: Opt for particular information types to redact.
- `regex_pattern_groups_path`: Input a path for personalized patterns.
- `use_faker`: Substitutes eligible entities with fabricated data.
- `recognizer_conf`: Configures recognizer for the PII data detection. There are many PII detection models available for various use-cases.
- `threshold`: Sets the acceptance threshold (Default: `0`).
- `language`: Language of the anonymize detect. Default is "en".

To revert to the initial data, utilize the [Deanonymize](../output_scanners/deanonymize.md)
scanner.

## Optimization Strategies

[Read more](../tutorials/optimization.md)

## Benchmarks

Test setup:

- Platform: Amazon Linux 2
- Python Version: 3.11.6
- Input Length: 317
- Test Times: 5

Run the following script:

```sh
python benchmarks/run.py input Anonymize
```

Results:

| Instance                       | Latency Variance | Latency 90 Percentile | Latency 95 Percentile | Latency 99 Percentile | Average Latency (ms) | QPS     |
|--------------------------------|------------------|-----------------------|-----------------------|-----------------------|----------------------|---------|
| AWS m5.xlarge                  | 6.11             | 255.64                | 294.57                | 325.71                | 177.13               | 1789.64 |
| AWS m5.xlarge with ONNX        | 0.73             | 155.64                | 169.13                | 179.93                | 128.64               | 2464.29 |
| AWS g5.xlarge GPU              | 38.50            | 321.59                | 419.60                | 498.01                | 125.18               | 2532.35 |
| AWS g5.xlarge GPU with ONNX    | 1.04             | 70.49                 | 86.47                 | 99.26                 | 38.11                | 8317.53 |
| AWS r6a.xlarge (AMD)           | 0.45             | 266.44                | 276.45                | 284.47                | 244.17               | 1298.29 |
| AWS r6a.xlarge (AMD) with ONNX | 0.35             | 238.15                | 247.22                | 254.47                | 218.91               | 1448.06 |
