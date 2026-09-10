"""Resolve operator-owned suites or author/cache them from the trusted contract."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from long_horizon.remote_numerical import validate_suite
from .infrastructure_retry import check_review_service, retry_infrastructure

PROMPT = Path(__file__).with_name("prompts") / "numerical_suite.md"
EXAMPLES = Path(__file__).resolve().parents[1] / "reference" / "numerical_suites"


def resolve_suite(campaign, workspace, private):
    supplied = private / "numerical_suite.json"
    if supplied.is_file():
        return supplied
    from .session_io import run_session

    sources = {}
    for name in ("input.py", "reference.py", "shapes.json", "agent_problem.json"):
        path = private / name if (private / name).is_file() else workspace / name
        if path.is_file():
            sources[name] = path
    if not {"input.py", "reference.py", "shapes.json"} <= set(sources):
        raise ValueError("numerical suite generation needs trusted input.py, reference.py and shapes.json")
    sources["instructions.md"] = PROMPT
    for name in ("attention", "gemm", "norm"):
        sources[f"examples/{name}.json"] = EXAMPLES / f"{name}.json"
    digest = hashlib.sha256()
    for name, path in sorted(sources.items()):
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    cached = private / ".atrex_numerical" / digest.hexdigest() / "numerical_suite.json"
    if cached.is_file():
        validate_suite(json.loads(cached.read_text()))
        return cached
    with tempfile.TemporaryDirectory(prefix="atrex-numerical-contract-") as temporary:
        root = Path(temporary)
        for name, path in sources.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        def author_once():
            (root / "numerical_suite.json").unlink(missing_ok=True)
            result = run_session(root, PROMPT.read_text(), timeout=600,
                                 agent_cli=campaign.agent_cli, reasoning_effort="high", agent_plugins=False)
            campaign._account(result, "operator numerical suite construction")
            check_review_service(result)
            return result
        retry_infrastructure(workspace, f"numerical-suite:{digest.hexdigest()}", author_once)
        for name, path in sources.items():
            if (root / name).read_bytes() != path.read_bytes():
                raise ValueError("numerical suite author modified its contract evidence")
        suite = validate_suite(json.loads((root / "numerical_suite.json").read_text()))
        if not 3 <= len(suite["cases"]) <= 6 or suite.get("coverage", "compact") != "compact":
            raise ValueError("automatically authored suites require 3..6 compact risk cases")
        if suite.get("evaluator_command") or suite.get("evaluator_files"):
            raise ValueError("custom evaluator adapters must be operator-supplied, not invented by the suite author")
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(suite, indent=2) + "\n")
    return cached
