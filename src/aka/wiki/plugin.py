"""Public GPU Wiki Core plugin contribution."""
from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

from aka.contracts.agent_instructions import InstructionSpec
from aka.contracts.agent_resources import AgentSkill, WorkspaceResourceContribution
from aka.contracts.attribution import AttributionProviderSpec
from aka.contracts.tools import ToolSpec
from aka.resources.host import EnvironmentBinding, SkillMountSpec

name = "wiki"
Config = {"type": "object", "properties": {"store_root": {"type": "string", "minLength": 1}},
          "required": ["store_root"], "additionalProperties": False}
interpolate = ("store_root",)
identity_packages = ("aka.wiki", "aka.contracts.tools", "aka.contracts.agent_instructions", "aka.contracts.attribution")
TOOL = "gpu-wiki.query"
RESOURCE = "gpu-wiki-public-store"
PROVIDER = "gpu-wiki"


def apply(ctx, config):
    root = Path(config["store_root"]).expanduser().resolve()
    _check_store(root)
    ctx.contribute("workspace_resources", RESOURCE,
                    WorkspaceResourceContribution(RESOURCE, root, visibility="runtime"))
    binding = EnvironmentBinding("gpu-wiki-profile", {"ATREX_WIKI_PROFILE_ROOT": "{workspace}/.gpu_wiki_profile"})
    ctx.contribute("environment_bindings", binding.name, binding)
    task_binding = EnvironmentBinding("gpu-wiki-task-id", {"ATREX_WIKI_TASK_ID": "{campaign_id}"})
    ctx.contribute("environment_bindings", task_binding.name, task_binding)
    schemas = files("aka.wiki").joinpath("schemas")
    input_schema = json.loads(schemas.joinpath("query.input.json").read_text(encoding="utf-8"))
    output_schema = json.loads(schemas.joinpath("query.output.json").read_text(encoding="utf-8"))
    ctx.contribute("tools", TOOL, ToolSpec(
        name=TOOL, description="Query the selected public GPU Wiki for scoped optimization experience and hardware facts.",
        input_schema=input_schema, output_schema=output_schema, entrypoint="aka.wiki.tool:invoke",
        resources=(RESOURCE,), timeout_seconds=360, attribution_provider=PROVIDER))
    instructions = files("aka.wiki").joinpath("instructions")
    phase_map = {"instructions.md": (("common",), ()), "setup.md": (("setup",), ("PLATFORM", "ARCH", "OPERATOR")),
                 "episode.md": (("episode", "fast_episode"), ("PLATFORM", "FRAMEWORK", "OPERATOR", "ARCH")),
                 "framework_baseline.md": (("framework_baseline",), ("PLATFORM", "ARCH", "OPERATOR", "FRAMEWORK"))}
    for filename, (phases, variables) in phase_map.items():
        key = filename.removesuffix(".md")
        text = instructions.joinpath(filename).read_text(encoding="utf-8")
        text = re.sub(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}", r"{\1}", text)
        spec = InstructionSpec(name="gpu-wiki-" + key.replace("_", "-"), phases=phases, template=text, variables=variables)
        ctx.contribute("instructions", spec.name, spec)
    skill_source = root / "3rdparty" / "KernelWiki"
    if skill_source.is_dir() and (skill_source / "SKILL.md").is_file():
        skill = AgentSkill("kernel-wiki", skill_source,
                           "Query the selected public GPU Wiki and retain its source attribution.")
        ctx.contribute("skill_mounts", skill.name, SkillMountSpec(skill, "KernelWiki", optional=True))
    provider = AttributionProviderSpec(
        name=PROVIDER, api_version=1, extract_entrypoint="aka.wiki.attribution:extract",
        validate_entrypoint="aka.wiki.attribution:validate", artifacts_entrypoint="aka.wiki.attribution:artifacts")
    ctx.contribute("attribution_providers", PROVIDER, provider)


def _check_store(root: Path) -> None:
    if not root.is_dir() or not (root / "tools" / "query_nl.py").is_file():
        raise ValueError(f"selected public GPU Wiki store is unavailable: {root}")
