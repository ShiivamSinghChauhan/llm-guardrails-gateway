"""Input guardrails - layer 1: fast deterministic regex."""

import re

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|the|your)\s+(instructions|prompts|rules|system\s*prompt|guidelines)",
    r"disregard\s+(your|all|the)\s+(rules|instructions|guidelines|system\s*prompt)",
    r"forget\s+(everything|all|your)\s+(above|previous|prior)",
    r"new\s+instruction[s]?\s*:",
    r"override\s+(your|the|all)\s+(instructions|rules|prompt)",
    r"\bsystem\s*:\s*",
    r"\[INST\]",
    r"<\|im_start\|>",
    r"what\s+(is|are)\s+your\s+(?:(?:system|original|base)\s*)+(prompt|instructions|rules)",
    r"reveal\s+(your|the)\s+(?:(?:system|original|base)\s*)+(prompt|instructions)",
    r"tell\s+me\s+(your|the)\s+(?:(?:system|original|base)\s*)+(prompt|instructions)",
    r"(write|repeat|show|display|print|output)\s+.{0,20}(system\s*prompt|instructions\s*(given|provided))",
    r"(word\s*(to|for)\s*word|verbatim|exactly)\s+.{0,30}(prompt|instructions|rules)",
    r"what\s+(were\s+you|are\s+you)\s+(told|instructed|prompted)\s+(to|before|above|at\s+the\s+start)",
    r"(copy|paste|reproduce)\s+.{0,20}(prompt|instructions|rules)",
]

PII_PATTERNS = {
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
    "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone":       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
    "ip_address":  r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
}

JAILBREAK_PATTERNS = [
    r"act\s+as\s+DAN",
    r"pretend\s+you\s+(are|have)\s+no\s+(restrictions|rules|limits)",
    r"\bjailbroken\b",
    r"do\s+anything\s+now",
    r"bypass\s+(your|the|all)\s+(filters|safety|restrictions|guardrails)",
    r"opposite\s+mode",
    r"evil\s+(mode|version|twin)",
    r"you\s+are\s+now\s+(a|an)\s+\w*\s*(unrestricted|uncensored|jailbroken)",
    r"no\s+(content\s+)?(policy|filters|restrictions|guidelines|limits)\b.{0,30}\b(answer|respond|comply|do)",
    r"(disregard|ignore|bypass)\s+.{0,20}safety",
    r"jailbreak\s+(yourself|your|the|this|now|mode)",
    r"(enable|activate|enter|turn\s+on)\s+developer\s+mode",
]

def detect_jailbreak(text: str) -> bool:
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


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

def _luhn_valid(digits: str) -> bool:
    """Luhn checksum — filters random digit runs out of card detection."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:        
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def detect_pii(text: str) -> list[dict]:
    """Detect PII. Returns [{type, match, evasion?}, ...]."""
    findings = []
    seen = set()

    def add(pii_type, match, evasion=None):
        if (pii_type, match) in seen:
            return
        seen.add((pii_type, match))
        f = {"type": pii_type, "match": match}
        if evasion:
            f["evasion"] = evasion
        findings.append(f)

    # ---- Pass 1: normal text ----
    for pii_type, pattern in PII_PATTERNS.items():
        for match in re.findall(pattern, text, re.IGNORECASE):
            add(pii_type, match)

    # ---- Pass 2: spacing-evasion (collapse whitespace, relax boundaries) ----
    collapsed = re.sub(r"\s+", "", text)
    collapsed_patterns = {
        "credit_card": r"(?:\d[-]?){13,16}",
        "ssn":         r"\d{3}-\d{2}-\d{4}",
        "email":       r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "ip_address":  r"(?:\d{1,3}\.){3}\d{1,3}",
    }
    card_runs = set()
    for pii_type, pattern in collapsed_patterns.items():
        for match in re.findall(pattern, collapsed, re.IGNORECASE):
            digits = re.sub(r"\D", "", match)
            if pii_type == "credit_card":
                # Luhn gate: kills ~90% of random-number false positives
                if not (13 <= len(digits) <= 16) or not _luhn_valid(digits):
                    continue
                card_runs.add(digits)
            # Don't double-report a card's digits as an SSN
            if pii_type == "ssn" and any(digits in run for run in card_runs):
                continue
            add(pii_type, match, evasion="spacing")

    return findings


def redact_pii(text: str, findings: list[dict]) -> str:
    """Replace detected PII with [REDACTED:<TYPE>] placeholders."""
    redacted = text
    for f in findings:
        placeholder = f"[REDACTED:{f['type'].upper()}]"

        if f.get("evasion") == "spacing":
            pattern = r"\s*".join(re.escape(c) for c in f["match"])
            redacted = re.sub(pattern, placeholder, redacted)
        else:
            redacted = redacted.replace(f["match"], placeholder)

    return redacted


def check_input(text: str, policy: dict) -> dict:
    """Run all input guards.

    Returns {safe, violations, notes, redacted_text}.
    `redacted_text` is set when policy says pii_action: redact — in that case
    the PII becomes a non-blocking NOTE instead of a blocking VIOLATION.
    """
    violations, notes = [], []
    redacted_text = None
    cfg = policy.get("input_guards", {})

    # ---- PII ----
    if cfg.get("block_pii"):
        pii = detect_pii(text)
        if pii:
            if cfg.get("pii_action", "block") == "redact":
                redacted_text = redact_pii(text, pii)
                notes.append({"guard": "pii", "action": "redacted", "details": pii})
            else:
                violations.append({"guard": "pii", "details": pii})

    # ---- Injection ----
    if cfg.get("block_prompt_injection"):
        if detect_prompt_injection(text, cfg.get("blocked_phrases", [])):
            violations.append({"guard": "prompt_injection",
                               "details": "Prompt injection detected"})

    # ---- Jailbreak ----
    if cfg.get("block_jailbreak"):
        if detect_jailbreak(text):
            violations.append({"guard": "jailbreak",
                               "details": "Jailbreak attempt detected"})
            
    return {
        "safe": len(violations) == 0,
        "violations": violations,
        "notes": notes,
        "redacted_text": redacted_text,
    }

# text = "My card 4532-0151-1283-0366 and 1234567890123456 and my friend card is: 1 2 3 456 789 01 2      5 4 5    6 , blabla321@example.com, okok3212   @ e xample     . c o m "
# x = detect_pii(text)
# print(redact_pii(text, x))