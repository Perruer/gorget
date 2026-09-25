"""Scanning a whole conversation instead of one prompt.

A single-message check misses attacks spread over several messages: an instruction split
into parts ("remember part 1… now part 2… run it"), a slow escalation where every message
looks harmless on its own, or instructions hidden in tool results and retrieved documents.
`scan_conversation` takes the chat history in the OpenAI format and adds three checks to the
usual scan of the latest user message:

- **window**: the last few user messages are also scanned together, as one text;
- **tool messages**: tool results that came after the latest user message are scanned for
  injections separately (indirect prompt injection);
- **accumulated risk** (opt-in): per-message injection scores add up with a decay, so a
  conversation whose messages are each below the threshold but keep pushing gets flagged.

No state is kept between calls: clients send the whole history with every request, the way
chat completion APIs work, and scores of messages already seen are cached.

The accumulated risk is a heuristic, not a guarantee: gradual jailbreaks can be made of
messages that no classifier finds suspicious. Scanning the model output with the output
scanners remains the most reliable last line of defence for multi-turn attacks.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .evaluate import scan_prompt
from .exception import GorgetValidationError
from .input_scanners.ban_code import BanCode
from .input_scanners.ban_competitors import BanCompetitors
from .input_scanners.ban_substrings import BanSubstrings
from .input_scanners.ban_topics import BanTopics
from .input_scanners.base import Scanner as InputScanner
from .input_scanners.prompt_injection import PromptInjection
from .input_scanners.toxicity import Toxicity
from .util import calculate_risk_score, get_logger

LOGGER = get_logger()

USER_ROLE = "user"
TOOL_ROLES = frozenset({"tool", "function"})

# Scanners that only detect, so running them on joined messages has no side effects
# (Anonymize, for example, would write the joined text into the vault).
WINDOW_SCANNER_TYPES: tuple[type, ...] = (
    PromptInjection,
    BanTopics,
    BanSubstrings,
    BanCompetitors,
    BanCode,
    Toxicity,
)

RISK_RESULT_NAME = "ConversationRisk"


@dataclasses.dataclass(frozen=True)
class Message:
    role: str
    content: str


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # OpenAI content parts: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
            parts.append(str(part.get("text", "")))
        elif isinstance(part, str):
            parts.append(part)
    return "\n".join(parts)


def to_messages(messages: Iterable[Message | dict]) -> list[Message]:
    """Accept `Message` objects or OpenAI-style dicts (``content`` may be a list of parts)."""
    result = []
    for message in messages:
        if isinstance(message, Message):
            result.append(message)
            continue
        if not isinstance(message, dict) or "role" not in message:
            raise GorgetValidationError("Every message needs a role and content")
        result.append(Message(str(message["role"]), _content_text(message.get("content"))))
    return result


class ConversationRisk:
    """
    Accumulated risk of a conversation.

    Every user message gets a score from 0 to 1 (by default the injection score of a
    `PromptInjection` scanner). Scores below `min_score` count as zero, the rest add up with
    weight ``decay ** age``, so recent messages count most: with decay 0.7, three messages
    in a row scoring 0.6 give 0.6 + 0.42 + 0.29 = 1.31. The conversation is flagged when the
    sum reaches `threshold`.

    Parameters:
        score: Callable returning a 0–1 score for one message, for example your own
            escalation classifier. Use `from_scanner` to take the score of a scanner.
        decay: Weight multiplier per message of age, between 0 and 1.
        threshold: Sum at which the conversation is flagged.
        min_score: Scores below this value are treated as zero, so ordinary messages do
            not add up.
        cache_size: How many message scores to remember between calls.
    """

    def __init__(
        self,
        score: Callable[[str], float],
        *,
        decay: float = 0.7,
        threshold: float = 1.2,
        min_score: float = 0.2,
        cache_size: int = 4096,
    ) -> None:
        if not 0 < decay <= 1:
            raise GorgetValidationError("decay must be between 0 and 1")
        if threshold <= 0:
            raise GorgetValidationError("threshold must be positive")
        self._score = score
        self.decay = decay
        self.threshold = threshold
        self.min_score = min_score
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @classmethod
    def from_scanner(cls, scanner: InputScanner, **kwargs) -> ConversationRisk:
        """
        Score messages with a scanner: the injection score of `PromptInjection`, otherwise the
        positive part of the scanner's risk score.
        """
        if isinstance(scanner, PromptInjection):
            return cls(lambda text: scanner.detect(text).score, **kwargs)
        return cls(lambda text: max(0.0, scanner.scan(text)[2]), **kwargs)

    def message_score(self, text: str) -> float:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
        score = float(self._score(text)) if text.strip() else 0.0
        with self._lock:
            self._cache[text] = score
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return score

    def evaluate(self, texts: Sequence[str]) -> float:
        """Accumulated risk of messages given oldest first."""
        total = 0.0
        for age, text in enumerate(reversed(texts)):
            score = self.message_score(text)
            if score >= self.min_score:
                total += score * self.decay**age
        return round(total, 3)


@dataclasses.dataclass
class ConversationScanResult:
    sanitized_prompt: str
    results_valid: dict[str, bool]
    results_score: dict[str, float]

    @property
    def is_valid(self) -> bool:
        return all(self.results_valid.values())


def _scanner_name(scanner: Any) -> str:
    return type(scanner).__name__


def scan_conversation(
    scanners: list[InputScanner],
    messages: Iterable[Message | dict],
    *,
    window: int = 3,
    window_scanners: list[InputScanner] | None = None,
    tool_scanners: list[InputScanner] | None = None,
    risk: ConversationRisk | None = None,
    fail_fast: bool = False,
) -> ConversationScanResult:
    """
    Scan a conversation: the latest user message with all scanners, plus the checks
    described in the module documentation.

    Parameters:
        scanners: Input scanners for the latest user message.
        messages: Chat history, oldest first, as `Message` objects or OpenAI-style dicts.
        window: How many recent user messages to scan together; 1 or less turns it off.
        window_scanners: Scanners for the joined messages. Default: the detection-only
            scanners among `scanners` (PromptInjection, BanTopics, BanSubstrings, BanCompetitors,
            BanCode, Toxicity).
        tool_scanners: Scanners for tool messages after the latest user message. Default:
            the PromptInjection scanners among `scanners`. Pass [] to skip tool messages.
        risk: Accumulated risk over all user messages; off when None.
        fail_fast: Stop at the first failed check.

    Returns:
        The sanitized latest user message and results per check. Window results are named
        like ``PromptInjection[window]``, tool results ``PromptInjection[tool]``, and the
        accumulated risk ``ConversationRisk``.
    """
    history = to_messages(messages)
    user_indexes = [i for i, message in enumerate(history) if message.role == USER_ROLE]
    if not user_indexes:
        raise GorgetValidationError("The conversation has no user message to scan")

    last_user = user_indexes[-1]
    prompt = history[last_user].content
    sanitized_prompt, results_valid, results_score = scan_prompt(scanners, prompt, fail_fast)

    def failed() -> bool:
        return fail_fast and not all(results_valid.values())

    user_texts = [history[i].content for i in user_indexes]

    # 1. Recent user messages together.
    if window_scanners is None:
        window_scanners = [s for s in scanners if isinstance(s, WINDOW_SCANNER_TYPES)]
    window_texts = [text for text in user_texts[-window:] if text.strip()] if window > 1 else []
    if len(window_texts) > 1 and not failed():
        joined = "\n".join(window_texts)
        for scanner in window_scanners:
            _, is_valid, score = scanner.scan(joined)
            name = f"{_scanner_name(scanner)}[window]"
            results_valid[name], results_score[name] = is_valid, score
            if not is_valid:
                LOGGER.warning("Recent messages fail together", scanner=name, risk_score=score)
                if fail_fast:
                    break

    # 2. Tool results the model is about to read.
    if tool_scanners is None:
        tool_scanners = [s for s in scanners if isinstance(s, PromptInjection)]
    tool_texts = [
        message.content
        for message in history[last_user + 1 :]
        if message.role in TOOL_ROLES and message.content.strip()
    ]
    if tool_texts and not failed():
        for scanner in tool_scanners:
            name = f"{_scanner_name(scanner)}[tool]"
            outcomes = [scanner.scan(text) for text in tool_texts]
            results_valid[name] = all(outcome[1] for outcome in outcomes)
            results_score[name] = max(outcome[2] for outcome in outcomes)
            if not results_valid[name]:
                LOGGER.warning("Tool output contains an injection", scanner=name)
                if fail_fast:
                    break

    # 3. Risk accumulated over the conversation.
    if risk is not None and not failed():
        total = risk.evaluate(user_texts)
        results_valid[RISK_RESULT_NAME] = total < risk.threshold
        # 0.5 on the normalised scale is the threshold, as for the other scanners.
        normalised = min(1.0, total / (2 * risk.threshold))
        results_score[RISK_RESULT_NAME] = calculate_risk_score(normalised, 0.5)
        if total >= risk.threshold:
            LOGGER.warning("Conversation risk is too high", risk=total, threshold=risk.threshold)

    return ConversationScanResult(sanitized_prompt, results_valid, results_score)
