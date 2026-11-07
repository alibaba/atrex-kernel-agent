"""Harness-neutral, JSON-serializable tool and instruction contributions.

Definitions contain public metadata only. Execution bindings, credentials and
backend-specific names belong to their consumers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import re

CATALOG_SCHEMA = "aka.contributions.v1"
TOOL_REFERENCE = re.compile(r"\{\{tool:([a-z][a-z0-9_.-]*)\}\}")


@dataclass(frozen=True)
class ToolContribution:
    id: str
    description: str
    input_schema: Mapping
    output_schema: Mapping


@dataclass(frozen=True)
class InstructionContribution:
    id: str
    text: str


@dataclass(frozen=True)
class ModuleContributions:
    id: str
    version: str
    tools: tuple[ToolContribution, ...] = ()
    instructions: tuple[InstructionContribution, ...] = ()


def render_instructions(catalog: Mapping, bindings: Mapping[str, str]) -> str:
    """Resolve semantic tool references against one consumer's exact bindings."""
    ids = {row["id"] for row in catalog["tools"]}
    if set(bindings) != ids:
        raise ValueError("instruction bindings must match the selected tool catalog")

    def replace(match):
        try:
            return bindings[match[1]]
        except KeyError as error:
            raise ValueError(f"unresolved tool reference: {match[1]}") from error

    rendered = "\n\n".join(
        TOOL_REFERENCE.sub(replace, row["text"])
        for row in catalog["instructions"]
    )
    if "{{tool:" in rendered:
        raise ValueError("malformed tool reference in instructions")
    return rendered
