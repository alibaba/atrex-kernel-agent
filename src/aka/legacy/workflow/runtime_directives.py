"""Workflow-owned instructions for the selected Agent backend."""

def _agent_runtime_directive(agent_cli: str) -> str:
    if agent_cli in {'codex', 'pi'}:
        directory = '.agents/skills'
    else:
        directory = '.qoder/skills' if agent_cli=='qodercli' else '.claude/skills'
    return (f'- `{directory}/` contains the selected module Skills. '
            'Use `python -m aka.legacy.resources.cli list` to discover the exact installed '
            'Skills and tools, including any explicitly selected profiling capabilities.')


def _baseline_driver_directive(agent_cli: str) -> str:
    if agent_cli == "codex":
        return (
            "Use the `$gpu-kernel-baseline` skill and complete its baseline workflow in this "
            "session. If Codex collaboration/sub-agent tools are available, delegate that bounded "
            "implementation task and wait for it; otherwise execute the skill directly yourself"
        )
    if agent_cli == "pi":
        return (
            "Use the `/skill:gpu-kernel-baseline` skill and complete its workflow directly in "
            "this Pi session. Pi has no built-in subagent requirement here; do not launch a "
            "nested coding-agent process"
        )
    if agent_cli == "qodercli":
        return (
            "Complete the baseline workflow directly in this Qoder session. Do not launch an "
            "Agent/subagent. Treat the current working directory as the only writable "
            "workspace and use relative paths for every campaign file"
        )
    return (
        "Launch the `gpu-kernel-baseline` subagent (by name). You may spawn it in the "
        "background, but **you MUST wait for it to complete before you exit**"
    )


def _plan_generator_directive(agent_cli: str, version: int) -> str:
    draft = f"plans/v{version}_draft.md"
    plan = f"plans/v{version}_plan.md"
    if agent_cli == "codex":
        return (
            f"Invoke the `$gen-plan` skill with `{draft}` as input and `{plan}` as "
            "output. Use direct/no-discussion mode for this optimization plan. "
            "The skill is installed under `.agents/skills/`; freeze its candidate proposal "
            "and Codex review in this current session before reading the independent Qoder review, "
            "then synthesize."
        )
    if agent_cli == "pi":
        return (
            f"Invoke `/skill:gen-plan` in this Pi session with `{draft}` as input and "
            f"`{plan}` as output. Use direct/no-discussion mode and wait for the plan file before "
            "continuing."
        )
    if agent_cli == "qodercli":
        return (
            f"Read `.agents/skills/gen-plan/SKILL.md` and execute it directly with `{draft}` as input and "
            f"`{plan}` as output in direct/no-discussion mode. Do not launch a planning subagent; "
            "freeze the skill's candidate proposal and Qoder review in this current session before "
            "reading the independent Codex review, then synthesize and wait for the plan file before "
            "continuing."
        )
    return f"```text\n/gen-plan --input {draft} --output {plan} --direct\n```"
