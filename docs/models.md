# Models and licenses

Gorget downloads models from the Hugging Face Hub on first use; none are bundled. Each model
keeps its own license. Check it before using a scanner in a commercial product: models marked
**non-commercial** may only be used for research or personal projects, and "not stated" means
the model card gives no license at all.

Pass another model to a scanner with its `model` argument, or `recognizer_conf` for `Anonymize`
and `Sensitive`.

| Model | Used by | License | ONNX export |
|-------|---------|---------|-------------|
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | Relevance | mit | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | Relevance | mit | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | Relevance | mit | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | Anonymize, Sensitive (`recognizer_conf`) | mit | [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) |
| [dslim/bert-large-NER](https://huggingface.co/dslim/bert-large-NER) | Anonymize, Sensitive (`recognizer_conf`) | mit | [dslim/bert-large-NER](https://huggingface.co/dslim/bert-large-NER) |
| [DunnBC22/codebert-base-Malicious_URLs](https://huggingface.co/DunnBC22/codebert-base-Malicious_URLs) | MaliciousURLs | not stated (ONNX export; original repository deleted) | [ProtectAI/codebert-base-Malicious_URLs-onnx](https://huggingface.co/ProtectAI/codebert-base-Malicious_URLs-onnx) |
| [Gherman/bert-base-NER-Russian](https://huggingface.co/Gherman/bert-base-NER-Russian) | Anonymize, Sensitive (default for `language="ru"`) | mit | [onnx-community/bert-base-NER-Russian-ONNX](https://huggingface.co/onnx-community/bert-base-NER-Russian-ONNX) |
| [gravitee-io/bert-small-pii-detection](https://huggingface.co/gravitee-io/bert-small-pii-detection) | Anonymize, Sensitive (`recognizer_conf`) | apache-2.0 | [gravitee-io/bert-small-pii-detection](https://huggingface.co/gravitee-io/bert-small-pii-detection) |
| [guishe/nuner-v1_orgs](https://huggingface.co/guishe/nuner-v1_orgs) | BanCompetitors | cc-by-sa-4.0 | [protectai/guishe-nuner-v1_orgs-onnx](https://huggingface.co/protectai/guishe-nuner-v1_orgs-onnx) |
| [gyr66/bert-base-chinese-finetuned-ner](https://huggingface.co/gyr66/bert-base-chinese-finetuned-ner) | Anonymize, Sensitive (`recognizer_conf`) | not stated | [ProtectAI/gyr66-bert-base-chinese-finetuned-ner-onnx](https://huggingface.co/ProtectAI/gyr66-bert-base-chinese-finetuned-ner-onnx) |
| [Isotonic/deberta-v3-base_finetuned_ai4privacy_v2](https://huggingface.co/Isotonic/deberta-v3-base_finetuned_ai4privacy_v2) | Anonymize, Sensitive (default) | **non-commercial** (cc-by-nc-4.0) | [Isotonic/deberta-v3-base_finetuned_ai4privacy_v2](https://huggingface.co/Isotonic/deberta-v3-base_finetuned_ai4privacy_v2) |
| [Isotonic/distilbert_finetuned_ai4privacy_v2](https://huggingface.co/Isotonic/distilbert_finetuned_ai4privacy_v2) | Anonymize, Sensitive (`recognizer_conf`) | **non-commercial** (cc-by-nc-4.0) | [Isotonic/distilbert_finetuned_ai4privacy_v2](https://huggingface.co/Isotonic/distilbert_finetuned_ai4privacy_v2) |
| [Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2](https://huggingface.co/Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2) | Anonymize, Sensitive (`recognizer_conf`) | **non-commercial** (cc-by-nc-4.0) | [Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2](https://huggingface.co/Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2) |
| [lakshyakh93/deberta_finetuned_pii](https://huggingface.co/lakshyakh93/deberta_finetuned_pii) | Anonymize, Sensitive (`recognizer_conf`) | mit | [protectai/lakshyakh93-deberta_finetuned_pii-onnx](https://huggingface.co/protectai/lakshyakh93-deberta_finetuned_pii-onnx) |
| [madhurjindal/autonlp-Gibberish-Detector-492513457](https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457) | Gibberish | mit | [madhurjindal/autonlp-Gibberish-Detector-492513457](https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457) |
| [MoritzLaurer/bge-m3-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0) | BanTopics | mit | [MoritzLaurer/bge-m3-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0) |
| [MoritzLaurer/deberta-v3-base-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0) | BanTopics, FactualConsistency | mit | [MoritzLaurer/deberta-v3-base-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0) |
| [MoritzLaurer/deberta-v3-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) | BanTopics | mit | [MoritzLaurer/deberta-v3-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) |
| [MoritzLaurer/roberta-base-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/roberta-base-zeroshot-v2.0-c) | BanTopics | mit | [protectai/MoritzLaurer-roberta-base-zeroshot-v2.0-c-onnx](https://huggingface.co/protectai/MoritzLaurer-roberta-base-zeroshot-v2.0-c-onnx) |
| [MoritzLaurer/roberta-large-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/roberta-large-zeroshot-v2.0-c) | BanTopics | mit | [MoritzLaurer/roberta-large-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/roberta-large-zeroshot-v2.0-c) |
| [papluca/xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection) | Language, LanguageSame | mit | [ProtectAI/xlm-roberta-base-language-detection-onnx](https://huggingface.co/ProtectAI/xlm-roberta-base-language-detection-onnx) |
| [philomath-1209/programming-language-identification](https://huggingface.co/philomath-1209/programming-language-identification) | Code | wtfpl | [philomath-1209/programming-language-identification](https://huggingface.co/philomath-1209/programming-language-identification) |
| [protectai/deberta-v3-base-prompt-injection](https://huggingface.co/protectai/deberta-v3-base-prompt-injection) | PromptInjection | apache-2.0 | [ProtectAI/deberta-v3-base-prompt-injection](https://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection) |
| [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | PromptInjection | apache-2.0 | [ProtectAI/deberta-v3-base-prompt-injection-v2](https://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection-v2) |
| [protectai/deberta-v3-small-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-small-prompt-injection-v2) | PromptInjection | apache-2.0 | [protectai/deberta-v3-small-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-small-prompt-injection-v2) |
| [ProtectAI/distilroberta-base-rejection-v1](https://huggingface.co/ProtectAI/distilroberta-base-rejection-v1) | NoRefusal | apache-2.0 | [ProtectAI/distilroberta-base-rejection-v1](https://huggingface.co/ProtectAI/distilroberta-base-rejection-v1) |
| [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) | EmotionDetection | mit | [SamLowe/roberta-base-go_emotions-onnx](https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx) |
| [unitary/unbiased-toxic-roberta](https://huggingface.co/unitary/unbiased-toxic-roberta) | Toxicity | apache-2.0 | [ProtectAI/unbiased-toxic-roberta-onnx](https://huggingface.co/ProtectAI/unbiased-toxic-roberta-onnx) |
| [valurank/distilroberta-bias](https://huggingface.co/valurank/distilroberta-bias) | Bias | other | [ProtectAI/distilroberta-bias-onnx](https://huggingface.co/ProtectAI/distilroberta-bias-onnx) |
| [vishnun/codenlbert-sm](https://huggingface.co/vishnun/codenlbert-sm) | BanCode | not stated | [protectai/vishnun-codenlbert-sm-onnx](https://huggingface.co/protectai/vishnun-codenlbert-sm-onnx) |
| [vishnun/codenlbert-tiny](https://huggingface.co/vishnun/codenlbert-tiny) | BanCode | mit | [protectai/vishnun-codenlbert-tiny-onnx](https://huggingface.co/protectai/vishnun-codenlbert-tiny-onnx) |
