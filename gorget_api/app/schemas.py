from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class ScanPromptRequest(BaseModel):
    prompt: str = Field(title="Prompt")
    scanners_suppress: List[str] = Field(title="Scanners to suppress", default=[])


class ScanPromptResponse(BaseModel):
    is_valid: bool = Field(title="Whether the prompt is safe")
    scanners: Dict[str, float] = Field(title="Risk scores of individual scanners")


class AnalyzePromptRequest(ScanPromptRequest):
    pass


class AnalyzePromptResponse(ScanPromptResponse):
    sanitized_prompt: str = Field(title="Sanitized prompt")


class ScanOutputRequest(BaseModel):
    prompt: str = Field(title="Prompt")
    output: str = Field(title="Model output")
    scanners_suppress: List[str] = Field(title="Scanners to suppress", default=[])


class ScanOutputResponse(BaseModel):
    is_valid: bool = Field(title="Whether the output is safe")
    scanners: Dict[str, float] = Field(title="Risk scores of individual scanners")


class AnalyzeOutputRequest(ScanOutputRequest):
    pass


class AnalyzeOutputResponse(ScanOutputResponse):
    sanitized_output: str = Field(title="Sanitized output")


class ChatMessage(BaseModel):
    role: str = Field(title="Role: user, assistant, system, tool")
    content: Union[str, List[Any], None] = Field(
        title="Text, or OpenAI content parts", default=None
    )


class ScanConversationRequest(BaseModel):
    messages: List[ChatMessage] = Field(title="Chat history, oldest first")
    scanners_suppress: List[str] = Field(title="Scanners to suppress", default=[])


class ScanConversationResponse(BaseModel):
    is_valid: bool = Field(title="Whether the conversation is safe")
    scanners: Dict[str, float] = Field(
        title="Risk scores of individual checks",
        description="Scanner names for the latest user message, Name[window] for recent messages "
        "scanned together, Name[tool] for tool results, ConversationRisk for the accumulated risk",
    )


class AnalyzeConversationRequest(ScanConversationRequest):
    pass


class AnalyzeConversationResponse(ScanConversationResponse):
    sanitized_prompt: str = Field(title="Sanitized latest user message")
