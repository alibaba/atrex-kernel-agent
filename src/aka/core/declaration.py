"""The plugin declaration contract.

A plugin is a module that exports ``apply``. Everything else it may export -- ``name``,
``Config``, ``Defaults``, ``validate_config``, ``inject``, ``optional_inject``, ``provide``,
``interpolate`` --
is validated here, before the module is ever mounted.

A module declares a plugin one way only. Ambiguous exports and unsupported hook
signatures are rejected before mounting.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from aka.core.plugin_runtime.schema import PluginError, check_schema, validate_schema

from .errors import DeclarationError
from .effects import require_sync
from .keys import NAME, ServiceKey

#: Alternative plugin-object exports. Offering one alongside ``apply`` is ambiguous.
_RIVAL_EXPORTS = ("PLUGIN", "Plugin", "plugin")

_SEQUENCE_EXPORTS = ("inject", "optional_inject", "provide", "interpolate")


@dataclass(frozen=True)
class PluginDeclaration:
    """The validated, immutable description of one plugin module."""

    name: str
    module: str
    apply: Callable[..., Any]
    config_schema: Mapping[str, Any] | None
    defaults: Mapping[str, Any]
    inject: tuple[str, ...]
    optional_inject: tuple[str, ...]
    provide: tuple[str, ...]
    interpolate: tuple[str, ...]
    resume_policy: str = "strict"
    identity_files: tuple[str, ...] = ()
    identity_packages: tuple[str, ...] = ()
    observational_fields: tuple[str, ...] = ()
    doc: str = ""
    config_validator: Callable[[Mapping[str, Any]], Any] | None = None

    @property
    def required_services(self) -> tuple[str, ...]:
        return self.inject

    def with_extra_inject(self, extra: Sequence[str]) -> "PluginDeclaration":
        """Add composition-row level required injections without touching the module.

        A row may force ordering by naming services the module did not declare; the row is
        then responsible for reaching them through ``ctx.get``.
        """
        added = {name for name in extra if name}
        overlap = sorted(added & set(self.provide))
        if overlap:
            # declare() rejects this in a module because such a row can never activate; a row
            # must not be able to reach the same dead state through the back door.
            raise DeclarationError(
                self.module,
                f"composition row injects {', '.join(overlap)}, which this plugin provides; "
                "the row would wait on itself forever",
            )
        merged = tuple(sorted(set(self.inject) | added))
        if merged == self.inject:
            return self
        return PluginDeclaration(
            name=self.name,
            module=self.module,
            apply=self.apply,
            config_schema=self.config_schema,
            defaults=self.defaults,
            inject=merged,
            optional_inject=tuple(
                name for name in self.optional_inject if name not in merged
            ),
            provide=self.provide,
            interpolate=self.interpolate,
            resume_policy=self.resume_policy,
            identity_files=self.identity_files,
            identity_packages=self.identity_packages,
            observational_fields=self.observational_fields,
            doc=self.doc,
            config_validator=self.config_validator,
        )


def is_plugin(module: ModuleType) -> bool:
    return callable(getattr(module, "apply", None))


def declare(
    module: ModuleType, *, known_seams: Mapping[str, ServiceKey] | None = None
) -> PluginDeclaration:
    """Validate one plugin module and return its declaration.

    ``known_seams`` restricts ``inject``/``provide`` to registered seams, so a misspelled
    service name fails at load instead of three hours into a campaign.
    """
    module_name = getattr(module, "__name__", repr(module))
    apply = getattr(module, "apply", None)
    if not callable(apply):
        raise DeclarationError(module_name, "module exports no callable apply")

    # A package that re-exports from its own ``plugin`` submodule necessarily binds that
    # submodule as an attribute; a module is never an alternative declaration, so only
    # non-module values count as rivals.
    rivals = [
        export
        for export in _RIVAL_EXPORTS
        if hasattr(module, export)
        and not isinstance(getattr(module, export), ModuleType)
    ]
    if rivals:
        raise DeclarationError(
            module_name,
            f"exports apply and also {', '.join(rivals)}; a module declares a plugin one "
            "way only, or the loader silently ignores half of the declaration",
        )

    _check_apply_signature(module_name, apply)

    config_validator = getattr(module, "validate_config", None)
    if config_validator is not None:
        _check_config_validator_signature(module_name, config_validator)

    name = getattr(module, "name", None)
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise DeclarationError(
            module_name, f"name must be a lowercase stable identifier, got {name!r}"
        )
    expected = _expected_name(module_name, is_package=hasattr(module, "__path__"))
    if name != expected:
        raise DeclarationError(
            module_name, f'name must be "{expected}" to match its package, got "{name}"'
        )

    sequences = {
        export: _sequence(module_name, export, getattr(module, export, ()))
        for export in _SEQUENCE_EXPORTS
    }
    inject = sequences["inject"]
    optional_inject = sequences["optional_inject"]
    provide = sequences["provide"]
    interpolate = sequences["interpolate"]
    identity_files = _sequence(module_name, "identity_files", getattr(module, "identity_files", ()))
    identity_packages = _sequence(module_name, "identity_packages", getattr(module, "identity_packages", ()))
    resume_policy = getattr(module, "resume_policy", "strict")
    observational_fields = _sequence(module_name, "observational_fields", getattr(module, "observational_fields", ()))
    if resume_policy not in ("strict", "observational"):
        raise DeclarationError(module_name, "resume_policy must be strict or observational")
    if resume_policy == "observational" and not observational_fields:
        raise DeclarationError(module_name, "observational resume_policy requires observational_fields")
    forbidden_observational = set(observational_fields) - {"group"}
    if forbidden_observational:
        raise DeclarationError(module_name, "observational_fields may currently contain only group")

    overlap = set(inject) & set(provide)
    if overlap:
        raise DeclarationError(
            module_name,
            f"cannot inject and provide the same service: {', '.join(sorted(overlap))}",
        )
    overlap = set(inject) & set(optional_inject)
    if overlap:
        raise DeclarationError(
            module_name,
            f"service is both required and optional: {', '.join(sorted(overlap))}",
        )

    if known_seams is not None:
        referenced = set(inject) | set(optional_inject) | set(provide)
        unknown = sorted(referenced - set(known_seams))
        if unknown:
            raise DeclarationError(
                module_name,
                f"references unregistered seams: {', '.join(unknown)}",
            )

    schema = getattr(module, "Config", None)
    if schema is not None:
        if not isinstance(schema, dict):
            raise DeclarationError(module_name, "Config must be a schema dict")
        try:
            check_schema(schema)
        except PluginError as exc:
            raise DeclarationError(module_name, f"Config is invalid: {exc}") from exc
        if schema.get("type") != "object":
            raise DeclarationError(module_name, 'Config must have type "object"')

    defaults = getattr(module, "Defaults", {})
    if not isinstance(defaults, dict):
        raise DeclarationError(module_name, "Defaults must be a dict")
    _check_defaults(module_name, schema, defaults)

    for path in interpolate:
        if not _property_exists(schema, path):
            raise DeclarationError(
                module_name,
                f'interpolate names "{path}", which is not a Config property',
            )

    return PluginDeclaration(
        name=name,
        module=module_name,
        apply=apply,
        config_schema=schema,
        defaults=dict(defaults),
        inject=inject,
        optional_inject=optional_inject,
        provide=provide,
        interpolate=interpolate,
        resume_policy=resume_policy,
        identity_files=identity_files,
        identity_packages=identity_packages,
        observational_fields=observational_fields,
        doc=(inspect.getdoc(module) or "").strip(),
        config_validator=config_validator,
    )


def _expected_name(module_name: str, *, is_package: bool = False) -> str:
    """The name a plugin module must declare: its own last segment, ``_`` to ``-``.

    A plugin package normally re-exports from a ``plugin`` submodule, so ``declare()`` may be
    handed either. Only a *non-package* trailing ``plugin`` segment is treated as that wrapper: a
    package legitimately named ``plugin`` keeps its own name.
    """
    segments = [segment for segment in module_name.split(".") if segment]
    segment = segments[-1] if segments else module_name
    if not is_package and segment in ("plugin", "__init__") and len(segments) > 1:
        segment = segments[-2]
    return segment.replace("_", "-")


def _check_apply_signature(module_name: str, apply: Callable[..., Any]) -> None:
    if not inspect.isfunction(apply):
        raise DeclarationError(
            module_name, "apply must be a plain function taking (ctx, config)"
        )
    try:
        require_sync(apply, "apply")
    except TypeError as exc:
        raise DeclarationError(module_name, str(exc)) from exc
    parameters = list(inspect.signature(apply).parameters.values())
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    variadic = [
        parameter
        for parameter in parameters
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
    required_keyword = [
        parameter
        for parameter in parameters
        if parameter.kind == parameter.KEYWORD_ONLY
        and parameter.default is parameter.empty
    ]
    if variadic:
        raise DeclarationError(
            module_name,
            "apply must not be variadic; declare apply(ctx, config) explicitly",
        )
    if required_keyword:
        raise DeclarationError(
            module_name,
            "apply must not require keyword-only arguments; the loader passes "
            "(ctx, config) positionally",
        )
    if len(positional) != 2:
        raise DeclarationError(
            module_name,
            f"apply takes {len(positional)} positional argument(s); it must take exactly "
            "(ctx, config), otherwise a declared Config can never reach it",
        )


def _check_config_validator_signature(module_name: str, validator: Callable[..., Any]) -> None:
    """Require a side-effect-free one-argument config validator.

    A validator runs during boot preflight, before locks are reconciled and before a plugin is
    applied. Keeping the callable shape narrow makes that timing part of the declaration
    contract instead of an undocumented convention between one provider and one caller.
    """
    if not inspect.isfunction(validator):
        raise DeclarationError(
            module_name, "validate_config must be a plain function taking (config)"
        )
    try:
        require_sync(validator, "validator")
    except TypeError as exc:
        raise DeclarationError(module_name, str(exc)) from exc
    parameters = list(inspect.signature(validator).parameters.values())
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    variadic = [
        parameter
        for parameter in parameters
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
    required_keyword = [
        parameter
        for parameter in parameters
        if parameter.kind == parameter.KEYWORD_ONLY
        and parameter.default is parameter.empty
    ]
    if variadic:
        raise DeclarationError(
            module_name,
            "validate_config must not be variadic; it receives the effective config",
        )
    if required_keyword or len(positional) != 1:
        raise DeclarationError(
            module_name,
            "validate_config must take exactly one positional argument (config)",
        )


def _sequence(module_name: str, export: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise DeclarationError(
            module_name, f"{export} must be a tuple of strings, got {type(value).__name__}"
        )
    names = tuple(value)
    if any(not isinstance(item, str) or not item for item in names):
        raise DeclarationError(module_name, f"{export} contains a non-string entry")
    duplicates = sorted({item for item in names if names.count(item) > 1})
    if duplicates:
        raise DeclarationError(
            module_name, f"{export} repeats {', '.join(duplicates)}"
        )
    return names


def _check_defaults(
    module_name: str, schema: Mapping[str, Any] | None, defaults: Mapping[str, Any]
) -> None:
    if not defaults:
        return
    if schema is None:
        raise DeclarationError(module_name, "Defaults declared without a Config schema")
    properties = schema.get("properties", {})
    unknown = sorted(set(defaults) - set(properties))
    if unknown:
        raise DeclarationError(
            module_name, f"Defaults sets unknown fields: {', '.join(unknown)}"
        )
    for key, value in defaults.items():
        try:
            validate_schema(properties[key], value, f"Defaults.{key}")
        except PluginError as exc:
            raise DeclarationError(module_name, str(exc)) from exc


def _property_exists(schema: Mapping[str, Any] | None, path: str) -> bool:
    node: Any = schema
    for segment in path.split("."):
        if not isinstance(node, dict):
            return False
        properties = node.get("properties")
        if not isinstance(properties, dict) or segment not in properties:
            return False
        node = properties[segment]
    return isinstance(node, dict)


__all__ = ["PluginDeclaration", "declare", "is_plugin"]
