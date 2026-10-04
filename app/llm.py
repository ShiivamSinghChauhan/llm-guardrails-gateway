import os, time
from functools import lru_cache

@lru_cache
def _get_llm(model: str = 'openai/gpt-oss-20b'):

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model = model,
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        timeout=30, 
        max_retries=0,
    )

CLASSIFY_RETRIES = 4

def classify_score(text: str, model: str = None) -> float:
    """Prompt-Guard-2 probability [0,1] that `text` is an attack.

    FAILS OPEN (returns 0.0) — if the classifier is unreachable we rely on the
    regex layer rather than blocking every user during an outage.
    """
    model = "meta-llama/llama-prompt-guard-2-86m"
    delay = 1.0

    for attempt in range(CLASSIFY_RETRIES):
        try:
            resp = _get_llm(model=model).invoke(text)
            return float(resp.content.strip())

        except (ValueError, TypeError, AttributeError):
            return 0.0          # unparseable -> fail open

        except Exception as e:
            msg = str(e).lower()
            transient = "429" in msg or "rate limit" in msg or "timeout" in msg
            if not transient or attempt == CLASSIFY_RETRIES - 1:
                return 0.0
            time.sleep(delay)
            delay *= 2       # exponential backoff
    return 0.0

