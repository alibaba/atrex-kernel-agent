"""Public contracts for deterministic, restricted Agent instruction templates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class InstructionSpec:
    name: str
    phases: tuple[str, ...]
    template: str
    variables: tuple[str, ...] = ()
    order: int = 500

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*", self.name
        ):
            raise ValueError("instruction name must be a lowercase identifier")
        if not isinstance(self.template, str):
            raise TypeError("instruction template must be text")
        phases = tuple(self.phases)
        variables = tuple(self.variables)
        if not phases or any(not isinstance(p, str) or not p for p in phases):
            raise ValueError("instruction phases must be non-empty names")
        if len(set(phases)) != len(phases):
            raise ValueError("instruction phases must be unique")
        if any(
            not isinstance(v, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v)
            for v in variables
        ):
            raise ValueError("instruction variables must be identifiers")
        if len(set(variables)) != len(variables):
            raise ValueError("instruction variables must be unique")
        if isinstance(self.order, bool) or not isinstance(self.order, int):
            raise TypeError("instruction order must be an integer")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "variables", variables)


@dataclass(frozen=True)
class RenderedInstruction:
    name: str
    owner: str
    scope: str
    order: int
    text: str
    identity: str


def instruction_identity(spec: InstructionSpec) -> str:
    raw = json.dumps(
        {
            "name": spec.name,
            "phases": spec.phases,
            "template": spec.template,
            "variables": spec.variables,
            "order": spec.order,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = ["InstructionSpec", "RenderedInstruction", "instruction_identity"]
