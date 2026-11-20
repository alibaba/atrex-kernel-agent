"""Implementation requirements over the same immutable candidate as evaluation."""
from __future__ import annotations
import ast
from dataclasses import asdict
import json
from pathlib import Path
import re
import uuid
from .content import atomic_json, identity, materialize, tree


def validate(policy):
    if not isinstance(policy, dict) or set(policy) - {"framework", "self_authored", "forbidden_patterns", "forbidden_imports", "reviewer"}:
        raise ValueError("Invalid implementation requirements")
    if "self_authored" in policy and type(policy["self_authored"]) is not bool:
        raise ValueError("self_authored must be a boolean")
    if "framework" in policy and (not isinstance(policy["framework"], str) or not policy["framework"]):
        raise ValueError("framework must name the required implementation DSL")
    for key in ("forbidden_patterns", "forbidden_imports"):
        if key in policy and (not isinstance(policy[key], list) or any(not isinstance(v, str) or not v for v in policy[key])):
            raise ValueError(key + " must be a list of nonempty strings")
    for pattern in policy.get("forbidden_patterns", []):
        re.compile(pattern)


def static_errors(files, policy):
    errors = []
    for name, data in files.items():
        try:
            source = data.decode()
        except UnicodeDecodeError:
            continue
        for pattern in policy.get("forbidden_patterns", []):
            if re.search(pattern, source):
                errors.append(f"{name}: forbidden source pattern {pattern!r}")
        if name.endswith(".py"):
            try:
                nodes = ast.walk(ast.parse(source, filename=name))
            except SyntaxError as error:
                errors.append(f"{name}:{error.lineno}: {error.msg}")
                continue
            for node in nodes:
                imports = ([v.name for v in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                for module in imports:
                    if any(module == rule or module.startswith(rule + ".") for rule in policy.get("forbidden_imports", [])):
                        errors.append(f"{name}:{node.lineno}: forbidden import {module}")
    return errors


def check(source, policy, *, contract_id, source_id, directory, agent_config, instructions):
    """Reuse the independent reviewer and schema; no candidate code runs here."""
    receipt_path = directory / "implementation.json"
    if receipt_path.exists():
        value = json.loads(receipt_path.read_text())
        if value["complete"] and value["contract_id"] == contract_id and value["source_id"] == source_id:
            return value
    files = tree(source)
    errors = static_errors(files, policy)
    value = {"review_id": uuid.uuid4().hex, "contract_id": contract_id, "source_id": source_id, "complete": True,
             "passed": not errors, "errors": errors, "usage": {}}
    if not errors and (policy.get("framework") or policy.get("self_authored", False)):
        from aka.contracts.agent import AgentAccess
        from aka.agent.provider import Agent, AgentOptions
        from aka.light.acceptance.review import validate_light_production_review
        from .agents import preflight
        config, environment = preflight(policy.get("reviewer") or agent_config or {"runtime": "claude"})
        review_root = directory / "review"
        materialize(review_root / "candidate", files)
        dsl = "pinned" if policy.get("framework") else "guided"
        atomic_json(review_root / "review_request.json", {"schema_version": 2,
            "framework": policy.get("framework"), "dsl_policy": dsl,
            "compute_policy": "self_authored_only" if policy.get("self_authored") else "declared_dependencies",
            "candidate_digest": source_id, "candidate_files": sorted(files),
            "contract_id": contract_id, "module_instructions": instructions,
            "implementation_requirements": {k: v for k, v in policy.items() if k != "reviewer"}})
        prompt_path = Path(__file__).parents[1] / "acceptance/dependency_review.md"
        message = prompt_path.read_text()
        message += ("\nThe request's module_instructions define the external module, which need not use Model "
                    "or kernel.py. Review all candidate files. implementation_requirements are the task's "
                    "policy. When self_authored is false, still report factual categories using the schema; "
                    "the host selects only the checks required by that policy.\n")
        owner = Agent(AgentOptions(timeout_seconds=240))
        session = owner.open_session(config["runtime"], review_root,
            access=AgentAccess(read_only_paths=(review_root / "candidate",)),
            environment={**environment, "ATREX_AGENT_SESSION_KEY": "review-" + source_id,
                         "PI_CODING_AGENT_SESSION_DIR": str(review_root / "sessions/pi")},
            output_directory=review_root / "sessions")
        try:
            observation = session.invoke(message, reasoning_effort=config.get("reasoning_effort", "high"))
            value["usage"] = asdict(observation.terminal_usage)
            value["session_id"] = observation.session_id
            value["exit_status"] = observation.exit_status
            if observation.exit_status != 0 or observation.timed_out:
                raise ValueError("Independent implementation reviewer did not finish successfully")
            if identity(tree(review_root / "candidate")) != source_id:
                raise ValueError("Independent reviewer changed candidate evidence")
            raw = json.loads((review_root / "dependency_review.json").read_text())
            errors, summary, result = validate_light_production_review(raw,
                candidate_files=frozenset(files), dsl_policy=dsl, self_authored=policy.get("self_authored", False))
            value.update(passed=not errors, errors=errors, summary=summary, review=result)
        except (ValueError, OSError) as error:
            value.update(complete=False, passed=False, errors=[str(error)])
        finally:
            session.close()
            owner.close()
    atomic_json(receipt_path, value)
    return value
