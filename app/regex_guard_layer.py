"""Input guardrails - layer 1: fast deterministic regex."""

import re

INJECTION_PATTERNS = [
    # Classic instruction-override phrasings
    r"ignore\s+(all\s+)?(previous|prior|above|the|your)\s+(instructions|prompts|rules|system\s*prompt|guidelines)",
    r"disregard\s+(your|all|the)\s+(rules|instructions|guidelines|system\s*prompt)",
    r"forget\s+(everything|all|your)\s+(above|previous|prior)",
    r"new\s+instruction[s]?\s*:",
    r"override\s+(your|the|all)\s+(instructions|rules|prompt)",

    # Fake role markers — trying to open a new "system" turn
    r"\bsystem\s*:\s*",
    r"\[INST\]",
    r"<\|im_start\|>",

    # System-prompt extraction
    r"what\s+(is|are)\s+your\s+(?:(?:system|original|base)\s*)+(prompt|instructions|rules)",
    r"reveal\s+(your|the)\s+(?:(?:system|original|base)\s*)+(prompt|instructions)",
    r"tell\s+me\s+(your|the)\s+(?:(?:system|original|base)\s*)+(prompt|instructions)",
    r"(write|repeat|show|display|print|output)\s+.{0,20}(system\s*prompt|instructions\s*(given|provided))",
    r"(word\s*(to|for)\s*word|verbatim|exactly)\s+.{0,30}(prompt|instructions|rules)",
    r"what\s+(were\s+you|are\s+you)\s+(told|instructed|prompted)\s+(to|before|above|at\s+the\s+start)",
    r"(copy|paste|reproduce)\s+.{0,20}(prompt|instructions|rules)",
]


def detect_prompt_injection(text: str, blocked_phrases: list = None) -> bool:
    """True if the text matches any known injection pattern."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Exact phrases from policy config
    lower = text.lower()
    for phrase in (blocked_phrases or []):
        if phrase.lower() in lower:
            return True

    return False
