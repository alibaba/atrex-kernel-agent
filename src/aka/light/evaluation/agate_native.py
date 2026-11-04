"""Discover and forward safe native Agate command options.

Agate owns the public spelling, types and enum values.  Light owns credentials,
sources, timeouts, uploads and job recovery, so those options are never copied
from the Agent-facing command line.  Updating the installed Agate package and
restarting the Supervisor refreshes this catalog without a Light release.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


COMMAND_BY_KIND = {
    "run": "eval",
    "profile": "profile",
    "dev": "dev",
    "check": "check",
    "disassemble": "disassemble",
    "env": "env",
}

# Runtime authority or options whose native semantics conflict with the
# Campaign's evaluator/source contract.  Long spellings are used because Light
# never advertises Agate's short aliases.
BLOCKED = frozenset({
    "--help", "--url", "--profile", "--token", "--ak", "--sk",
    "--http-timeout", "--wait-timeout", "--job-timeout", "--timeout", "--poll",
    "--wait", "--no-wait", "--oss-file", "--oss-output", "--gpu", "--num-gpus",
    "--candidate", "--manifest", "--candidate-entry", "--baseline", "--baseline-entry",
    "--reference-dir", "--config", "--solution", "--problem", "--definition-file",
    "--workload-file", "--file", "--working-dir", "--exclude", "--init-kwargs",
    "--requirements", "--recycle", "--no-recycle", "--dev-timeout",
    "--backend", "--harness", "--atrex-bench-version", "--set", "--operator",
    "--num-correctness-cases", "--bench-iters", "--mode", "--lock-clocks",
    # These select a different evaluator/source contract rather than tuning the
    # current operator request. They need first-class Runtime workflows.
    "--collective", "--numel", "--dtype", "--warmup-iters", "--subset",
    "--workload-timeout", "--compile-timeout", "--iterations", "--warmup-runs",
    "--benchmark-reference", "--abba-repeats", "--json",
})

# Native spellings already implemented by Light.  build_parser registers both
# spellings on one argparse action; discovery refreshes choices/help only.
ALIASES = {
    "--level": "--profile-level",
    "--counter": "--profile-counter",
    "--source": "--profile-source",
    "--fmt": "--format",
    "--force": "--env-force",
}


@dataclass(frozen=True)
class NativeOption:
    flag: str
    dest: str
    commands: frozenset[str]
    action: str
    choices: tuple[Any, ...] | None
    value_type: Any
    metavar: Any
    help: str | None


def _action_kind(action: argparse.Action) -> str:
    if isinstance(action, argparse._AppendAction):
        return "append"
    if isinstance(action, argparse._StoreTrueAction):
        return "store_true"
    if isinstance(action, argparse._StoreFalseAction):
        return "store_false"
    if isinstance(action, argparse.BooleanOptionalAction):
        return "boolean_optional"
    return "store"


def _long_flag(action: argparse.Action) -> str | None:
    values = [value for value in action.option_strings if value.startswith("--")]
    return next((value for value in values if not value.startswith("--no-")), None)


@lru_cache(maxsize=1)
def catalog() -> dict[str, NativeOption]:
    """Return safe options from the installed Agate parser, keyed by flag."""
    try:
        from atrex_gateway_client.cli import build_parser as agate_parser
    except (ImportError, ModuleNotFoundError):
        return {}
    parser = agate_parser()
    groups = getattr(parser, "_subparsers", None)
    if groups is None or not groups._group_actions:
        return {}
    choices = groups._group_actions[0].choices
    found: dict[str, NativeOption] = {}
    commands_for_flag: dict[str, set[str]] = {}
    raw: dict[str, argparse.Action] = {}
    for command in set(COMMAND_BY_KIND.values()):
        subparser = choices.get(command)
        if subparser is None:
            continue
        for action in subparser._actions:
            flag = _long_flag(action)
            if flag is None or flag in BLOCKED:
                continue
            commands_for_flag.setdefault(flag, set()).add(command)
            raw.setdefault(flag, action)
    for flag, action in raw.items():
        found[flag] = NativeOption(
            flag=flag,
            dest=action.dest,
            commands=frozenset(commands_for_flag[flag]),
            action=_action_kind(action),
            choices=tuple(action.choices) if action.choices is not None else None,
            value_type=action.type,
            metavar=action.metavar,
            help=None if action.help is argparse.SUPPRESS else action.help,
        )
    return found


def _existing_action(parser: argparse.ArgumentParser, flag: str) -> argparse.Action | None:
    return parser._option_string_actions.get(flag) or parser._option_string_actions.get(ALIASES.get(flag, ""))


def extend_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add installed Agate options to Light's parser under their native names."""
    for option in catalog().values():
        existing = _existing_action(parser, option.flag)
        if existing is not None:
            # The Runtime action owns defaults and validation, while Agate owns
            # the current enum and user-facing description.
            if option.choices is not None:
                existing.choices = option.choices
            if option.help:
                existing.help = option.help
            setattr(existing, "agate_commands", option.commands)
            setattr(existing, "agate_native_flag", option.flag)
            setattr(existing, "agate_value_type", option.value_type)
            continue
        dest = "agate_native__" + option.dest
        kwargs: dict[str, Any] = {
            "dest": dest,
            "default": argparse.SUPPRESS,
            "help": option.help,
        }
        if option.choices is not None:
            kwargs["choices"] = option.choices
        if option.metavar is not None:
            kwargs["metavar"] = option.metavar
        if option.action == "append":
            kwargs["action"] = "append"
            if option.value_type is not None:
                kwargs["type"] = option.value_type
        elif option.action == "store_true":
            kwargs["action"] = "store_true"
        elif option.action == "store_false":
            kwargs["action"] = "store_false"
        elif option.action == "boolean_optional":
            kwargs["action"] = argparse.BooleanOptionalAction
        elif option.value_type is not None:
            kwargs["type"] = option.value_type
        action = parser.add_argument(option.flag, **kwargs)
        setattr(action, "agate_commands", option.commands)
        setattr(action, "agate_native_flag", option.flag)
        setattr(action, "agate_value_type", option.value_type)
    return parser


def values(args: argparse.Namespace, kind: str) -> dict[str, Any]:
    """Return explicitly supplied native values and reject cross-command use."""
    command = COMMAND_BY_KIND.get(kind)
    result: dict[str, Any] = {}
    for option in catalog().values():
        key = "agate_native__" + option.dest
        if key not in vars(args):
            continue
        if command not in option.commands:
            raise ValueError(f"{option.flag} is not supported by agate {command or kind}")
        result[option.dest] = getattr(args, key)
    return result


def cli_arguments(args: argparse.Namespace, kind: str) -> list[str]:
    """Render explicitly supplied native values using Agate's own spellings."""
    supplied = values(args, kind)
    result: list[str] = []
    for option in catalog().values():
        if option.dest not in supplied:
            continue
        value = supplied[option.dest]
        if option.action == "append":
            for item in value:
                result += [option.flag, str(item)]
        elif option.action in {"store_true", "store_false"}:
            enabled = bool(value) if option.action == "store_true" else not bool(value)
            if enabled:
                result.append(option.flag)
        elif option.action == "boolean_optional":
            result.append(option.flag if value else "--no-" + option.flag.removeprefix("--"))
        else:
            result += [option.flag, str(value)]
    return result


def selectors(items: list[str] | None, label: str) -> dict[str, str]:
    """Parse Agate's repeatable KEY=VALUE selector contract."""
    result: dict[str, str] = {}
    for item in items or []:
        key, separator, value = item.partition("=")
        if not separator or not key or not value or key in result:
            raise ValueError(f"{label} requires distinct non-empty KEY=VALUE entries")
        result[key] = value
    return result


def apply_request(request: dict[str, Any], args: argparse.Namespace, kind: str) -> dict[str, Any]:
    """Apply native options to a direct HTTP request using Agate wire names."""
    supplied = values(args, kind)
    cluster = selectors(supplied.pop("cluster_label", None), "--cluster-label")
    node = selectors(supplied.pop("node_label", None), "--node-label")
    if cluster or node:
        placement = {
            **({"cluster_selector": cluster} if cluster else {}),
            **({"node_selector": node} if node else {}),
        }
        request["placement"] = placement
        diagnostic = request.get("__atrex_source_diagnostic")
        if isinstance(diagnostic, dict):
            diagnostic["placement"] = placement
    shape_shards = supplied.pop("shape_shards", None)
    if shape_shards is not None:
        request.setdefault("options", {})["shape_shards"] = shape_shards
    # Agate's typed Profile/Check/Disassemble fields use their parser dest as
    # their wire name. Unknown future additive fields follow that public SDK
    # convention and remain subject to Gateway schema validation.
    if supplied and isinstance(request.get("__atrex_source_diagnostic"), dict):
        names = ", ".join("--" + key.replace("_", "-") for key in sorted(supplied))
        raise ValueError(
            f"{names} cannot be represented by the source-tree diagnostic Dev contract"
        )
    for key, value in supplied.items():
        if key in request and request[key] != value:
            raise ValueError(f"Agate option {key!r} conflicts with the Runtime request")
        request[key] = value
    return request

