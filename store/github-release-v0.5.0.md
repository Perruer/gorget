Gorget 0.5.0 adds two things users asked for: plugging in your own models, and checks for attacks spread over several messages.

## Install

```bash
pip install -U gorget
```

## Your own models

- **Prompt injection classifier.** `PromptInjection(injection_labels=[...])` works with models whose labels are not `INJECTION` and adds up several injection categories; `detect()` returns the score and the category. `PromptInjection(classifier=...)` takes any callable: a scikit-learn model, an internal service, a vendor API.
- **Sensitive data.** `Anonymize` and `Sensitive` take your own recognizers (`recognizers=`, any function wrapped in `gorget.plugins.CallableRecognizer`), several NER models at once, or only yours (`recognizer_conf=[]`). Your own data types such as `CONTRACT_NUMBER` are detected by default and get their own placeholders.
- **API server.** Classifiers, recognizers, NER models and whole scanner classes can be referenced in `scanners.yml` by import path or by name from the `gorget.plugins` entry point group.

Guide: [Your own models](https://perruer.github.io/gorget/customization/custom_models/).

## Attacks over several messages

`gorget.conversation.scan_conversation` takes a chat history in the OpenAI format. Besides the latest user message it scans recent user messages together (split instructions), tool results the model is about to read (indirect injection) and, opt-in, the injection risk accumulated over the conversation. The API server has `/analyze/conversation` and `/scan/conversation`.

Guide, measurements and limits: [Multi-turn attacks](https://perruer.github.io/gorget/tutorials/conversations/).

## Fixes

- `PromptInjection` read the scores of models whose injection label is not `INJECTION` backwards.
- `Anonymize` and `Sensitive` changed the `entity_types` list passed by the caller.
- API server: `Anonymize` and `Sensitive` ignored `recognizer_conf` and `language` and always loaded the non-commercial English AI4Privacy model.
- Model downloads retry when Hugging Face answers 429.

Full list: [changelog](https://github.com/Perruer/gorget/blob/main/docs/changelog.md).

## Support

Gorget is free. If it protects your product, you can support it on [Boosty](https://boosty.to/mikio_kuroki/donate) or with crypto (addresses in the [README](https://github.com/Perruer/gorget#support-the-project)).
