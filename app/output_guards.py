import json, re
from llm import classify_score


TOXIC_PATTERNS = [
    # "how to build a bomb" and its reorderings
    r"\b(kill|murder|harm|attack|bomb|weapon)\b.{0,40}\b(how|instructions|steps|guide|make|build)\b",
    r"\b(how|instructions|steps|guide|make|build)\b.{0,40}\b(kill|murder|bomb|weapon|explosive)\b",
    r"\b(hack|exploit|breach|crack)\b.{0,40}\b(how|tutorial|steps|into|password)\b",
    r"\b(suicide|self.harm)\b.{0,40}\b(method|way|how)\b",
]

TOXIC_THRESHOLD = 0.8


def check_toxicity(response: str, use_classifier: bool = False,
                   threshold: float = TOXIC_THRESHOLD) -> dict:
    """Regex pre-filter, then an optional safeguard classifier."""
    lower = response.lower()
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, lower):
            return {"safe": False, "reason": "potentially harmful content detected (regex)"}

    if use_classifier:
        score = classify_score(response)
        if score >= threshold:
            return {"safe": False,
                    "reason": f"Harmful content detected (classifier {score:.2f})"}

    return {"safe": True}


def check_schema(response: str, schema: dict) -> dict:
    """Validate the response is JSON with the required fields."""
    if not schema:
        return {"valid": True}
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return {"valid": False, "reason": "Response is not valid JSON"}
    if not isinstance(data, dict):
        return {"valid": False, "reason": "Response JSON is not an object"}
    missing = [f for f in schema.get("required_fields", []) if f not in data]
    if missing:
        return {"valid": False, "reason": f"Missing fields: {missing}"}
    return {"valid": True}
