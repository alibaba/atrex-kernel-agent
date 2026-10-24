"""Classify provider error payloads, never normal assistant or tool text."""
import re


def rate_limit_code(message: str) -> str | None:
    codes = re.findall(r'"(?:code|type)"\s*:\s*"([^"\n]+)"', message)
    # Explicit permanent limits take precedence over HTTP status and prose.
    if any(code in {"insufficient_quota", "billing_error", "GoUsageLimitError",
                    "FreeUsageLimitError"} for code in codes):
        return None
    for code in codes:
        if code in {"Throttling", "rate_limit_error"} or code.startswith("Throttling."):
            return code
    if re.search(r"insufficient_quota|out of budget|quota exceeded|billing", message, re.I):
        return None
    if re.search(r"\b429\b|rate.?limit|too many requests", message, re.I):
        return "rate_limit"
    return None
