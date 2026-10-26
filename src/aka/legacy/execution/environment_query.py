"""Selected-backend discovery and matching; never execute a GPU workload here."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import re
import subprocess
from threading import Lock
from urllib.parse import quote, urlsplit, urlunsplit

from aka.contracts.execution import UnsupportedExecution
from aka.contracts.execution_environment import (
    EnvironmentQuery, EnvironmentSnapshot, EnvironmentSupport, requirements_identity,
)

_CACHE = {}
_LOCK = Lock()


def language_alias(value):
    token = re.sub(r"[^a-z0-9]", "", value.casefold())
    return {"cuda": "cuda_cpp", "cudacpp": "cuda_cpp", "cutedsl": "cute_dsl",
            "torch": "pytorch", "python": "pytorch", "tritongluon": "triton"}.get(token, token)


def _hardware(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def safe_endpoint(value):
    if not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    # SSH host aliases and configured profile labels carry no URL credentials.
    return value if re.fullmatch(r"[A-Za-z0-9_.:@-]+", value) else "configured-endpoint"


def _cache_key(backend, endpoint, hardware):
    # Use a digest of the full route, retaining auth/query distinctions privately.
    return backend, hashlib.sha256((endpoint or "default").encode()).hexdigest(), _hardware(hardware)


def _snapshot(backend, endpoint, query, detail, capabilities, errors, operations):
    declarations = {"operations": sorted(operations)}
    for key in ("operations", "supported_languages", "aliases", "limits", "max_gpus",
                "requirements_supported", "dependency_catalog", "dependency_policy"):
        if key in capabilities:
            declarations[key] = capabilities[key]
    if "operations" in declarations:
        declarations["operations"] = [{"eval": "run", "compile": "check"}.get(value, value)
                                     for value in declarations["operations"]]
    observed = {}
    for source in (detail, capabilities):
        declared = source.get("source") in {"declared", "unknown"}
        for key in ("gpu", "aliases", "vendor", "arch", "gpu_model", "sm_count", "total_memory_mb",
                    "driver_version", "toolchain", "frameworks", "libraries", "profilers",
                    "availability", "available", "ready", "image", "worker", "image_digest",
                    "worker_id", "environment_identity", "source", "environments"):
            if key in source:
                if declared and key not in {"source", "environments"}:
                    declarations.setdefault("environment", {})[key] = source[key]
                else:
                    observed[key] = source[key]
        if source.get("probe_error"):
            errors.append("environment-probe-incomplete")
    explicit_identity = capabilities.get("environment_identity", detail.get("environment_identity"))
    identity = explicit_identity if isinstance(explicit_identity, str) else None
    if identity is None and any(observed.get(key) for key in ("worker_id", "worker", "image_digest", "image")):
        identity = hashlib.sha256(json.dumps({"backend": backend, "endpoint": safe_endpoint(endpoint),
            "hardware": observed.get("gpu", query.hardware),
            "worker": observed.get("worker_id", observed.get("worker")),
            "image": observed.get("image_digest", observed.get("image")),
            "toolchain": observed.get("toolchain"), "libraries": observed.get("libraries")},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source = str(capabilities.get("source", detail.get("source", "unknown")))
    return EnvironmentSnapshot(backend, safe_endpoint(endpoint), str(detail.get("gpu", capabilities.get("gpu", query.hardware))),
        identity, declarations, observed, source,
        capabilities.get("probed_at", detail.get("probed_at")), tuple(dict.fromkeys(errors)))


class BackendEnvironment:
    """A gateway or SSH route owned by one bound Executor instance."""
    def __init__(self, *, backend, operations, environment, url="", profile="", ssh=""):
        self.backend = backend
        self.operations = frozenset(operations)
        self.environment = dict(environment)
        self.url, self.profile, self.ssh = url, profile, ssh
        configured_url = ""
        if not url and not profile and not ssh and not self.environment.get("AGATE_URL"):
            from pathlib import Path
            try:
                config = json.loads((Path(self.environment.get("HOME", str(Path.home()))) / ".atrex/config.json").read_text())
                configured_url = config.get("url", "") if isinstance(config, dict) else ""
            except (OSError, ValueError):
                pass
        self.endpoint = url or ("ssh:" + ssh if ssh else "profile:" + profile if profile else
                                self.environment.get("AGATE_URL") or configured_url or "default-gateway")

    def cached(self, query):
        key = _cache_key(self.backend, self.endpoint, query.hardware)
        with _LOCK:
            value = _CACHE.get(key)
            if value is None:
                for (backend, route, _), item in _CACHE.items():
                    aliases = item.observed.get("aliases", ())
                    if backend == key[0] and route == key[1] and _hardware(query.hardware) in {
                        _hardware(alias) for alias in (item.hardware, *aliases)}:
                        value = item
                        break
        return value or EnvironmentSnapshot(self.backend, safe_endpoint(self.endpoint), query.hardware, None,
            {"operations": sorted(self.operations), **({"max_gpus": 1} if self.ssh else {})},
            diagnostics=("environment-not-queried",))

    def query(self, query):
        cached = self.cached(query)
        if not query.fresh and "environment-not-queried" not in cached.diagnostics:
            return replace(cached, source="cache")
        if self.ssh:
            # The SSH sandbox has no stable query/worker identity contract. Do not
            # substitute a host probe for facts about its constrained namespace.
            return replace(cached, diagnostics=("ssh-environment-discovery-not-supported",))
        from .gateway import sandbox
        errors, detail, capabilities = [], {}, {}
        encoded = quote(query.hardware, safe="")
        suffix = "?force=true" if query.fresh else ""
        if self.url:
            for path, destination in ((f"/v1/env/{encoded}" if query.hardware else "/v1/env", detail),
                                      (f"/v1/env/{encoded}/capabilities", capabilities)):
                if destination is capabilities and not query.hardware:
                    continue
                try:
                    options = {"allow_list": True} if not query.hardware else {}
                    value = sandbox._gateway_json(self.url, "GET", path + suffix, None, 30, **options)
                    if isinstance(value, list):
                        destination["environments"] = value
                    else:
                        destination.update(value)
                except Exception as error:
                    errors.append("environment-http-" + str(error.status) if isinstance(error, sandbox.GatewayHTTPError)
                                  else "environment-discovery-" + type(error).__name__)
        else:
            try:
                executable = sandbox._find_agate()
                if executable is None:
                    raise FileNotFoundError("agate")
                base = [executable, "env"]
                if self.profile:
                    base += ["--profile", self.profile]
                if query.hardware:
                    base += [query.hardware]
                base += ["--http-timeout", "30", "--json"]
                if query.fresh:
                    base += ["--force"]
                for flags, destination in (([], detail), (["--frameworks", "--limits"], capabilities)):
                    if destination is capabilities and not query.hardware:
                        continue
                    completed = subprocess.run([*base, *flags], capture_output=True, text=True,
                                               env=self.environment, timeout=60)
                    if completed.returncode:
                        errors.append("environment-cli-rejected")
                        continue
                    value = json.loads(completed.stdout)
                    if isinstance(value, list) and not query.hardware:
                        destination["environments"] = value
                        continue
                    if not isinstance(value, dict):
                        raise ValueError("environment response must be an object")
                    destination.update(value)
            except Exception as error:
                errors.append("environment-discovery-" + type(error).__name__)
        snapshot = _snapshot(self.backend, self.endpoint, query, detail, capabilities, errors, self.operations)
        # An unknown observation stays unknown when cached; explicit fresh queries
        # refresh it without repeating discovery for every candidate submission.
        with _LOCK:
            if len(_CACHE) >= 64:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[_cache_key(self.backend, self.endpoint, query.hardware)] = snapshot
        return snapshot

    def check(self, query, *, discover=True):
        snapshot = self.query(query) if discover else self.cached(query)
        return match_environment(query, snapshot)


def match_environment(query, snapshot):
    """Match backend facts; a missing fact remains unknown, including resource availability."""
    declarations, observed = snapshot.declarations, snapshot.observed
    reasons = list(snapshot.diagnostics)
    dimensions = {}
    aliases = observed.get("aliases", declarations.get("aliases", ()))
    known_hardware = observed.get("gpu")
    dimensions["hardware"] = ("supported" if query.hardware and known_hardware and
        _hardware(query.hardware) in {_hardware(value) for value in (known_hardware, *aliases)} else
        "unsupported" if query.hardware and known_hardware else "unknown")
    operation = {"eval": "run", "compile": "check"}.get(query.operation, query.operation)
    operations = declarations.get("operations")
    dimensions["operation"] = ("supported" if operations is not None and operation in operations else
                                "unsupported" if operations is not None else "unknown")
    max_gpus = declarations.get("max_gpus")
    dimensions["num_gpus"] = ("unsupported" if type(max_gpus) is int and query.num_gpus > max_gpus else
                               "supported" if type(max_gpus) is int or query.num_gpus == 1 else "unknown")
    frameworks = observed.get("frameworks", {})
    if not isinstance(frameworks, Mapping):
        frameworks = {}
        reasons.append("invalid-framework-capabilities")
    language_unavailable = False
    languages = declarations.get("supported_languages")
    for language in query.languages:
        name = language_alias(language)
        declaration = None
        if isinstance(languages, Mapping):
            declaration = next((value for key, value in languages.items() if language_alias(key) == name), None)
        elif isinstance(languages, (list, tuple)):
            declaration = name in {language_alias(key) for key in languages}
        facts = next((value for key, value in frameworks.items() if language_alias(key) == name), None)
        if declaration is False or isinstance(facts, Mapping) and facts.get("supported") is False:
            status = "unsupported"
        elif declaration is True or isinstance(facts, Mapping) and facts.get("available") is True:
            status = "supported"
        else:
            status = "unknown"
        dimensions["language:" + name] = status
        if isinstance(facts, Mapping) and facts.get("available") is False:
            language_unavailable = True
            reasons.append("language-unavailable:" + name)
    installed = {}
    for values in (observed.get("libraries", {}), frameworks, observed.get("toolchain", {})):
        if not isinstance(values, Mapping):
            continue
        for name, value in values.items():
            version = value.get("version") if isinstance(value, Mapping) else value
            if isinstance(version, str) and version not in {"unknown", "unavailable"} and not (
                isinstance(value, Mapping) and value.get("available") is False):
                installed[re.sub(r"[-_.]+", "-", name).casefold()] = version
    catalog = declarations.get("dependency_catalog", ())
    policy = declarations.get("requirements_supported")
    for index, requirement in enumerate(query.requirements):
        available = False
        parsed = None
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.version import InvalidVersion
        try:
            parsed = Requirement(requirement)
            actual = installed.get(re.sub(r"[-_.]+", "-", parsed.name).casefold())
            # Host marker/extras evaluation cannot stand in for backend facts.
            if actual is not None and parsed.url is None and parsed.marker is None and not parsed.extras:
                available = parsed.specifier.contains(actual, prereleases=True)
        except (InvalidRequirement, InvalidVersion):
            pass  # URLs and backend-specific pins remain unknown unless declared in its catalog.
        status = "supported" if available or requirement in catalog else "unsupported" if policy is False else "unknown"
        dimensions[f"requirement:{index}"] = status
    values = set(dimensions.values())
    support = "unsupported" if "unsupported" in values else "unknown" if "unknown" in values else "supported"
    for dimension, status in dimensions.items():
        if status == "unsupported":
            reasons.append("unsupported:" + dimension)
    availability = observed.get("availability", "unknown")
    if availability not in {"available", "unavailable", "unknown"}:
        availability = "unknown"
    if observed.get("available") is False or observed.get("ready") is False or language_unavailable:
        availability = "unavailable"
    elif observed.get("available") is True or observed.get("ready") is True:
        availability = "available"
    return EnvironmentSupport(support, availability, snapshot, dimensions,
                              tuple(dict.fromkeys(reasons)), requirements_identity(query.requirements))


def command_query(command):
    languages, requirements = command.languages, command.requirements
    if command.submission is not None:
        prepared = json.loads(command.submission)
        request = prepared.get("request", {})
        languages = tuple(request.get("spec", {}).get("languages", languages))
        requirements = tuple(request.get("requirements", requirements))
    elif not languages:
        # The actual candidate manifest, including V0, takes precedence over the
        # campaign's intended conversion language. No language is guessed from GPU vendor.
        manifest = command.workspace / "solution.json"
        if manifest.is_file():
            from .gateway.sandbox import _candidate_languages
            solution = json.loads(manifest.read_text(encoding="utf-8"))
            languages = tuple(_candidate_languages(solution))
    path = command.environment.get("ATREX_SANDBOX_REQUIREMENTS_FILE")
    if path and not requirements and command.submission is None:
        from .gateway.sandbox import _runtime_requirements
        requirements = tuple(_runtime_requirements(path, candidate_languages=languages))
    return EnvironmentQuery(command.hardware, command.operation, languages, requirements, command.num_gpus)


def select_requirements(contents, languages=()):
    selected = None
    for line in contents.splitlines():
        if line.casefold().startswith("# aka-candidate-languages:"):
            selected = {language_alias(value) for value in re.split(r"[\s,]+", line.split(":", 1)[1].strip()) if value}
            if not selected:
                raise ValueError("runtime requirements selector declares no languages")
            break
    if selected is not None and selected.isdisjoint({language_alias(value) for value in languages}):
        return ()
    return tuple(line.strip() for line in contents.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def require_supported(support):
    if support.support == "unsupported":
        error = UnsupportedExecution("selected environment rejects request: " + ", ".join(support.reasons))
        error.environment_support = support
        raise error
