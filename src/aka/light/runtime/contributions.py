"""Composition-time registration and dispatch without feature dependencies."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import re

from aka.contracts.contributions import CATALOG_SCHEMA, ModuleContributions, TOOL_REFERENCE, render_instructions
from aka.light.runtime.plugins.schema import check_schema, validate_schema


@dataclass(frozen=True)
class ToolBinding:
    invoke: Callable[[object, dict], dict]
    prepare: Callable[[object, dict], object] | None = None


class ContributionRegistry:
    """Register atomically, then seal for one Runtime lifetime.

    Handlers borrow an opaque invocation context. The HTTP host owns its
    authorization, request lock, deadline and recovery discipline.
    """
    def __init__(self):
        self._modules = {}
        self._bindings = {}
        self._sealed = False

    def register(self, module: ModuleContributions, bindings: Mapping[str, ToolBinding]):
        if self._sealed:
            raise ValueError("contribution registry is sealed")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", module.id) or not module.version.strip():
            raise ValueError("module requires an identifier and version")
        if module.id in self._modules:
            raise ValueError(f"duplicate contribution module: {module.id}")
        rows = asdict(module)
        tools = {row["id"]: row for row in rows["tools"]}
        if len(tools) != len(rows["tools"]) or set(tools) != set(bindings):
            raise ValueError("each contributed tool needs exactly one binding")
        names = set()
        for row in [*rows["tools"], *rows["instructions"]]:
            name = row["id"]
            if (not isinstance(name, str) or not re.fullmatch(re.escape(module.id) + r"\.[a-z][a-z0-9_.-]*", name)
                    or name in names):
                raise ValueError("contribution IDs must be unique and owned by their module")
            names.add(name)
        for row in rows["tools"]:
            if not row["description"].strip():
                raise ValueError("tool description is required")
            check_schema(row["input_schema"])
            check_schema(row["output_schema"])
            if row["input_schema"]["type"] != "object" or row["output_schema"]["type"] != "object":
                raise ValueError("portable tools require object input and output schemas")
            if not callable(bindings[row["id"]].invoke):
                raise ValueError("tool binding must be callable")
        for row in rows["instructions"]:
            if not row["text"].strip() or set(TOOL_REFERENCE.findall(row["text"])) - tools.keys():
                raise ValueError("instructions require text and module-owned tool references")
        render_instructions(rows, {name: name for name in tools})
        # Snapshot definitions before accepting them; caller mutation cannot
        # change a live catalog or the schema used during invocation.
        self._modules[module.id] = json.loads(json.dumps(rows, allow_nan=False))
        self._bindings.update(bindings)

    def remove(self, module_id: str):
        if self._sealed:
            raise ValueError("contribution registry is sealed")
        module = self._modules.pop(module_id)
        for row in module["tools"]:
            del self._bindings[row["id"]]

    def seal(self, instruction_overrides: Mapping[str, str] | None = None):
        if self._sealed:
            raise ValueError("contribution registry is already sealed")
        modules = deepcopy(self._modules)
        instructions = {row["id"]: (module, row) for module in modules.values()
                        for row in module["instructions"]}
        for name, text in (instruction_overrides or {}).items():
            if name not in instructions or not isinstance(text, str) or not text.strip():
                raise ValueError(f"invalid instruction override: {name}")
            module, row = instructions[name]
            if set(TOOL_REFERENCE.findall(text)) - {tool["id"] for tool in module["tools"]}:
                raise ValueError(f"unresolved instruction reference: {name}")
            row["text"] = text
        for module in modules.values():
            render_instructions(module, {tool["id"]: tool["id"] for tool in module["tools"]})
        self._modules = modules
        self._sealed = True
        return self

    def catalog(self):
        if not self._sealed:
            raise ValueError("seal the contribution registry before publication")
        modules = [self._modules[key] for key in sorted(self._modules)]
        value = {"schema": CATALOG_SCHEMA,
                 "modules": [{"id": row["id"], "version": row["version"]} for row in modules],
                 "tools": sorted((row for module in modules for row in module["tools"]), key=lambda row: row["id"]),
                 "instructions": sorted((row for module in modules for row in module["instructions"]), key=lambda row: row["id"])}
        value = deepcopy(value)
        value["identity"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return value

    def resolve(self, request):
        if not isinstance(request, dict) or set(request) != {"catalog_id", "tool", "arguments"}:
            raise ValueError("tool request requires catalog_id, tool and arguments")
        catalog = self.catalog()
        if request["catalog_id"] != catalog["identity"]:
            raise ValueError("contribution catalog changed; resume through the owning application")
        tool = next((row for row in catalog["tools"] if row["id"] == request["tool"]), None)
        if tool is None:
            raise ValueError("tool is not selected in this Runtime")
        validate_schema(tool["input_schema"], request["arguments"])
        return tool, self._bindings[tool["id"]]

    def prepare(self, context, request):
        _, binding = self.resolve(request)
        return binding.prepare(context, request["arguments"]) if binding.prepare else None

    def invoke(self, context, request):
        tool, binding = self.resolve(request)
        result = binding.invoke(context, request["arguments"])
        try:
            validate_schema(tool["output_schema"], result, "result")
            json.dumps(result, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise RuntimeError("contributed handler returned an invalid result; inspect Runtime records") from error
        return result
