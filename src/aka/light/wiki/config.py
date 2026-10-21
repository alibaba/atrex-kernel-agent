"""Operator-only Wiki policy, independent of the Optimizer and GPU executor."""
from dataclasses import dataclass
import math
from pathlib import Path
import re
from urllib.parse import urlsplit


@dataclass(frozen=True)
class WikiConfig:
    model: str = ""
    effort: str = "low"
    settings_file: str = ""
    base_url: str = ""
    timeout: float = 120
    bridge_timeout: int = 60
    queue_timeout: float = 15
    concurrency: int = 16

    def __post_init__(self):
        for name in ("timeout", "bridge_timeout", "queue_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Wiki {name} must be finite and positive")
        if type(self.bridge_timeout) is not int or self.bridge_timeout > self.timeout:
            raise ValueError("Wiki bridge_timeout must be an integer no larger than timeout")
        if type(self.concurrency) is not int or not 1 <= self.concurrency <= 16:
            raise ValueError("Wiki concurrency must be 1–16")
        if self.effort not in {"low", "medium", "high", "max"}:
            raise ValueError("Invalid Wiki effort")
        if self.model and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", self.model):
            raise ValueError("Invalid Wiki model")
        if self.settings_file and not Path(self.settings_file).is_absolute():
            raise ValueError("Wiki settings_file must be an absolute operator-owned path")
        if self.base_url:
            endpoint = urlsplit(self.base_url)
            if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
                raise ValueError("Invalid Wiki base_url")
            if not self.settings_file:
                raise ValueError("Wiki base_url requires settings_file for credential endpoint verification")
