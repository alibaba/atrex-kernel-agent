"""Dependency-free validation for the explicitly supported plugin schema dialect."""

from __future__ import annotations

import json
import math
from pathlib import Path


class PluginError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def decode_json(text: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-JSON numeric constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def read_json(path: Path) -> object:
    try:
        return decode_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PluginError("invalid_config", f"cannot read JSON {path}: {exc}") from exc


def validate_schema(schema: dict, value: object, path: str = "input") -> None:
    kind = schema["type"]
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    if not isinstance(value, types[kind]) or (
        kind in {"integer", "number"} and isinstance(value, bool)
    ):
        raise PluginError("schema_validation", f"{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise PluginError("schema_validation", f"{path}: unsupported value")
    if kind == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise PluginError("schema_validation", f"{path}: missing {key}")
        for key, item in value.items():
            child = properties.get(key, schema.get("additionalProperties", True))
            if child is False:
                raise PluginError("schema_validation", f"{path}: unknown field {key}")
            if isinstance(child, dict):
                validate_schema(child, item, f"{path}.{key}")
    elif kind == "array":
        for index, item in enumerate(value):
            validate_schema(schema["items"], item, f"{path}[{index}]")
    elif kind == "string":
        if len(value.strip()) < schema.get("minLength", 0):
            raise PluginError(
                "schema_validation", f"{path}: string is empty or too short"
            )
    elif kind in {"number", "integer"}:
        if (isinstance(value, float) and not math.isfinite(value)) or not schema.get(
            "minimum", -float("inf")
        ) <= value <= schema.get("maximum", float("inf")):
            raise PluginError(
                "schema_validation", f"{path}: number outside allowed range"
            )


def check_schema(schema: object) -> None:
    allowed = {
        "type",
        "description",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "minLength",
        "minimum",
        "maximum",
    }
    if not isinstance(schema, dict) or set(schema) - allowed:
        raise PluginError("invalid_manifest", "unsupported schema keywords")
    if not isinstance(schema.get("type"), str) or schema["type"] not in {
        "object",
        "array",
        "string",
        "integer",
        "number",
        "boolean",
        "null",
    }:
        raise PluginError("invalid_manifest", "schema requires a supported type")
    typed_keywords = {
        "object": {"properties", "required", "additionalProperties"},
        "array": {"items"},
        "string": {"minLength"},
        "integer": {"minimum", "maximum"},
        "number": {"minimum", "maximum"},
        "boolean": set(),
        "null": set(),
    }
    if set(schema) - {"type", "description", "enum"} - typed_keywords[schema["type"]]:
        raise PluginError(
            "invalid_manifest", "schema constraint does not apply to its type"
        )
    if "enum" in schema and (
        not isinstance(schema["enum"], list) or not schema["enum"]
    ):
        raise PluginError("invalid_manifest", "enum must be a nonempty array")
    if "minLength" in schema and (
        type(schema["minLength"]) is not int or schema["minLength"] < 0
    ):
        raise PluginError("invalid_manifest", "minLength must be a nonnegative integer")
    for bound in ("minimum", "maximum"):
        if bound in schema and (
            type(schema[bound]) not in (int, float)
            or (isinstance(schema[bound], float) and not math.isfinite(schema[bound]))
        ):
            raise PluginError("invalid_manifest", "numeric bounds must be numbers")
    if schema.get("minimum", -float("inf")) > schema.get("maximum", float("inf")):
        raise PluginError("invalid_manifest", "minimum exceeds maximum")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not isinstance(
        schema.get("required", []), list
    ):
        raise PluginError("invalid_manifest", "invalid schema properties/required")
    if any(
        not isinstance(key, str) or key not in properties
        for key in schema.get("required", [])
    ):
        raise PluginError("invalid_manifest", "required key has no property schema")
    for child in properties.values():
        check_schema(child)
    extra = schema.get("additionalProperties", True)
    if isinstance(extra, dict):
        check_schema(extra)
    elif not isinstance(extra, bool):
        raise PluginError("invalid_manifest", "invalid additionalProperties")
    if schema["type"] == "array":
        check_schema(schema.get("items"))
