"""Workflow-owned episode instructions for the selected execution endpoint."""
from aka.legacy.resources.prompts import render_prompt
from .resources import PROMPTS_DIR
SANDBOX_FULL_WORKFLOW_PROMPT = PROMPTS_DIR / "sandbox_full_workflow.md"
SANDBOX_SAFETY_BOUNDARY_PROMPT = PROMPTS_DIR / "sandbox_safety_boundary.md"

def _sandbox_endpoint(profile: str = "", url: str = "", ssh: str = "") -> str:
    """Render the endpoint clause shared by sandbox directives."""
    if ssh:
        return f" using OpenSSH target `{ssh}`"
    if url:
        return f" using gateway URL `{url}`"
    if profile:
        return f" using gateway profile `{profile}`"
    return " using agate's configured gateway"

def sandbox_directive(
    hardware: str,
    profile: str = "",
    url: str = "",
    ssh: str = "",
) -> str:
    """Mandatory safety boundary plus full-mode workflow for full episodes."""
    endpoint = _sandbox_endpoint(profile, url, ssh)
    safety = render_prompt(
        SANDBOX_SAFETY_BOUNDARY_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint
    )
    workflow = render_prompt(
        SANDBOX_FULL_WORKFLOW_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint
    )
    return f"{safety.rstrip()}\n\n{workflow.strip()}\n"

def fast_sandbox_directive(
    hardware: str,
    profile: str = "",
    url: str = "",
    ssh: str = "",
) -> str:
    """Mandatory safety boundary for fast episodes.

    The fast episode prompt already describes the fast-specific execution
    contract (single evaluator, no multi-seed, no profile, supervisor-owned
    memory), so only the invariant safety boundary is injected here.
    """
    endpoint = _sandbox_endpoint(profile, url, ssh)
    return render_prompt(SANDBOX_SAFETY_BOUNDARY_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint)
