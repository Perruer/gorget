class GorgetValidationError(ValueError):
    pass


# Name used by LLM Guard, kept so existing `except` clauses keep working.
LLMGuardValidationError = GorgetValidationError
