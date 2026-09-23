# Gorget

Security scanners for LLM prompts and responses: prompt injection, PII anonymization,
secrets, toxicity, banned topics, malicious URLs and more.

Gorget continues [LLM Guard](https://github.com/protectai/llm-guard), which Protect AI
archived in July 2026. Code written for `llm_guard` keeps working: the package ships an
`llm_guard` compatibility module that points at the same scanners.

> **Status:** work in progress. The first Gorget release is being prepared; until then,
> use the source from this repository.

## Credits

Gorget is based on LLM Guard by Protect AI and its contributors, released under the MIT
License. The original README is kept in [docs/upstream-README.md](docs/upstream-README.md).

Gorget is an independent project. It is not affiliated with, endorsed by or sponsored by
Protect AI or Palo Alto Networks. "LLM Guard" is used only to describe compatibility.
