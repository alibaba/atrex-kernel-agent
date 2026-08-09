"""Concurrent bucket campaigns and the serialized full-workload aggregation gate."""
from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .campaign import Campaign
from .constants import (
    AGGREGATE_DISPATCH_FILE,
    AGGREGATE_DISPATCH_SCHEMA_VERSION,
    AGGREGATE_KERNELS_DIR,
    AGGREGATE_QUEUE_WAIT_GRACE,
    AGGREGATE_SOURCE_LAYOUT,
    AGGREGATE_VALIDATION_TIMEOUT,
    AGGREGATION_STATE_FILE,
    BUCKETS_DIR,
    DISPATCH_SIGNATURE_RESULT_PREFIX,
    DISPATCH_SIGNATURES_FILE,
    DISPATCH_VISIBILITY_POLICY,
    INITIAL_AGGREGATION_MIN_ITERATIONS,
    PROMPTS_DIR,
    REPO_ROOT,
    WORKLOAD_BUCKET_CONTRACT_FILE,
    WORKLOAD_BUCKETS_FILE,
)
from .dispatch_codegen import build_deterministic_dispatcher
from .dispatch_signatures import (
    build_generalized_dispatch_plan,
    validate_dispatch_bucket_compatibility,
    validate_dispatch_signatures,
)
from .session_io import (
    SessionResult,
    _record_local_test_result,
    _render,
    _sandbox_command,
    _status_is,
    _test_result_from_stdout,
    run_session,
)
from .workload_buckets import (
    WorkloadBucket,
    WorkloadSource,
    _bucket_baseline_memory,
    _materialize_bucket_op,
    _normalized_bucket_manifest,
    _read_workload_source,
    validate_workload_buckets,
)
from .workspace_state import (
    _git_blob_file,
    _git_head_file,
    _normalize_commit_hash,
    best_validated_latency_us,
    git_head,
    git_kernel_blob,
    head_kernel_is_initial_baseline,
    latest_version,
    preserve_interrupted_tracked_changes,
    read_memory,
    resolve_framework_baseline_commit,
    v0_baseline_commit,
)


@dataclass
class WorkloadBucketCoordinator:
    """Inspect, optimize buckets concurrently, and maintain the full-workload kernel."""

    aggregate_campaign: Campaign
    op_dir: Path
    max_buckets: int = 8
    aggregate_min_improvement: float = 0.0
    _aggregate_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _bucket_campaigns: dict[str, Campaign] = field(default_factory=dict, init=False, repr=False)

    @property
    def workspace(self) -> Path:
        return self.aggregate_campaign.workspace

    @property
    def manifest_path(self) -> Path:
        return self.workspace / WORKLOAD_BUCKETS_FILE

    @property
    def state_path(self) -> Path:
        return self.workspace / AGGREGATION_STATE_FILE

    def _account(self, result: SessionResult, label: str) -> None:
        self.aggregate_campaign._account(result, label)

    def _ensure_main_workspace(self) -> None:
        if latest_version(self.workspace) < 0:
            self.aggregate_campaign.setup_baseline()
        else:
            print(
                f"[workload-coordinator] resuming aggregate workspace at "
                f"v{latest_version(self.workspace)}",
                flush=True,
            )
            preserve_interrupted_tracked_changes(
                self.workspace, "resume aggregate workspace"
            )
            self.aggregate_campaign._link_runtime()
        gitignore = self.workspace / ".gitignore"
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        ignore_line = f"/{BUCKETS_DIR}/"
        if ignore_line not in existing.splitlines():
            with gitignore.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n# workload inspector derived inputs and independent bucket git repos\n"
                    f"{ignore_line}\n"
                )
        self._commit_main("workload coordinator: ignore derived bucket workspaces", ".gitignore")
        # Buckets inherit this kernel, so the framework bring-up happens once here rather than
        # once per bucket. Ordered after the ignore rule: the session must never stage the
        # nested bucket repositories of a resumed campaign.
        self.aggregate_campaign.ensure_framework_baseline()

    def _commit_main(self, message: str, *paths: str) -> None:
        subprocess.run(["git", "add", "--", *paths], cwd=str(self.workspace), check=True)
        changed = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=str(self.workspace)
        )
        if changed.returncode == 1:
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(self.workspace),
                check=True,
                stdout=subprocess.DEVNULL,
            )
        elif changed.returncode != 0:
            raise RuntimeError("could not inspect staged aggregate workspace changes")

    def _ensure_dispatch_signatures(
        self, workload_source: WorkloadSource
    ) -> list[dict]:
        """Collect and cache evaluator-faithful structural workload signatures."""
        signature_path = self.workspace / DISPATCH_SIGNATURES_FILE
        if signature_path.is_file():
            try:
                payload = json.loads(signature_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid {signature_path}: {exc}") from exc
            records = validate_dispatch_signatures(payload, workload_source)
            if payload.get("visibility_policy") != DISPATCH_VISIBILITY_POLICY:
                signature_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "visibility_policy": DISPATCH_VISIBILITY_POLICY,
                            "kind": workload_source.kind,
                            "workload_source": workload_source.filename,
                            "workloads": records,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self._commit_main(
                    "workload inspector: certify no-sync dispatch signatures",
                    DISPATCH_SIGNATURES_FILE,
                )
            return records

        timeout = max(
            self.aggregate_campaign.sandbox_timeout,
            AGGREGATE_VALIDATION_TIMEOUT,
        )
        try:
            result = _sandbox_command(
                self.workspace,
                self.aggregate_campaign.sandbox_hardware,
                self.aggregate_campaign.sandbox_profile,
                self.aggregate_campaign.sandbox_url,
                timeout,
                [
                    "python",
                    "tools/collect_dispatch_signatures.py",
                    "--workspace",
                    ".",
                ],
                wall_timeout=timeout + AGGREGATE_QUEUE_WAIT_GRACE,
                dispatch_signatures=True,
                gateway_kind="dev",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"dispatch signature collection failed to run: {exc}") from exc
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
        if result.stderr:
            print(
                result.stderr,
                end="" if result.stderr.endswith("\n") else "\n",
                file=sys.stderr,
                flush=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"dispatch signature collection failed (exit={result.returncode})"
            )
        payload: object = None
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(DISPATCH_SIGNATURE_RESULT_PREFIX):
                try:
                    payload = json.loads(line[len(DISPATCH_SIGNATURE_RESULT_PREFIX) :])
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"dispatch signature collector emitted invalid JSON: {exc}"
                    ) from exc
                break
        records = validate_dispatch_signatures(payload, workload_source)
        persisted = {
            "schema_version": 1,
            "visibility_policy": DISPATCH_VISIBILITY_POLICY,
            "kind": workload_source.kind,
            "workload_source": workload_source.filename,
            "workloads": records,
        }
        signature_path.write_text(
            json.dumps(persisted, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._commit_main(
            "workload inspector: record deterministic dispatch signatures",
            DISPATCH_SIGNATURES_FILE,
        )
        print(
            f"[workload-inspector] recorded {len(records)} deterministic runtime "
            f"signatures in {signature_path}",
            flush=True,
        )
        return records

    def inspect_workloads(self) -> list[WorkloadBucket]:
        workload_source = _read_workload_source(self.op_dir)
        workload_count = len(workload_source.entries)
        signature_records = self._ensure_dispatch_signatures(workload_source)
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            buckets = validate_workload_buckets(manifest, workload_count, self.max_buckets)
            validate_dispatch_bucket_compatibility(buckets, signature_records)
            if manifest.get("dispatch_visibility_policy") != DISPATCH_VISIBILITY_POLICY:
                # A legacy partition is still mechanically safe when every exact
                # production-visible signature has one owner. Remove rationales that may
                # have cited evaluator-only values and certify the partition under the
                # current no-sync policy without renaming its existing bucket workspaces.
                buckets = [
                    WorkloadBucket(
                        name=bucket.name,
                        workload_indices=bucket.workload_indices,
                        rationale=(
                            "Legacy partition revalidated as a union of exact "
                            f"{DISPATCH_VISIBILITY_POLICY} signature classes; no tensor "
                            "contents or workload-source values are dispatch inputs."
                        ),
                    )
                    for bucket in buckets
                ]
                self.manifest_path.write_text(
                    json.dumps(
                        {
                            **_normalized_bucket_manifest(buckets, workload_count),
                            "workload_source": workload_source.filename,
                            "workload_ids": list(workload_source.ids),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self._commit_main(
                    "workload inspector: certify production-visible bucket partition",
                    WORKLOAD_BUCKETS_FILE,
                )
            print(
                f"[workload-inspector] reusing {len(buckets)} validated buckets from "
                f"{self.manifest_path}",
                flush=True,
            )
            return buckets

        # The LLM inspector runs in a data-minimized workspace. It receives only the
        # signatures the generated production dispatcher can recompute without a device
        # synchronization; reference.py, input.py, shapes.json/workload.jsonl, tensor
        # contents, and evaluator metadata are not present.
        with tempfile.TemporaryDirectory(prefix="atrex-dispatch-inspector-") as temp_dir:
            inspector_workspace = Path(temp_dir)
            inspector_signature_path = inspector_workspace / DISPATCH_SIGNATURES_FILE
            inspector_signature_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "visibility_policy": DISPATCH_VISIBILITY_POLICY,
                        "kind": workload_source.kind,
                        "workload_source": workload_source.filename,
                        "workloads": signature_records,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=str(inspector_workspace),
                check=True,
            )
            prompt = _render(
                PROMPTS_DIR / "inspect_workloads.md",
                WORKSPACE=str(inspector_workspace),
                MAX_BUCKETS=self.max_buckets,
                WORKLOAD_COUNT=workload_count,
                WORKLOAD_FILE=workload_source.filename,
                WORKLOAD_KIND=workload_source.kind,
                VISIBILITY_POLICY=DISPATCH_VISIBILITY_POLICY,
                PLATFORM=self.aggregate_campaign.platform,
                FRAMEWORK=self.aggregate_campaign.framework,
            )
            result = run_session(
                inspector_workspace,
                prompt,
                timeout=self.aggregate_campaign.setup_timeout,
                agent_cli=self.aggregate_campaign.agent_cli,
                sandbox_hardware=self.aggregate_campaign.sandbox_hardware,
                sandbox_profile=self.aggregate_campaign.sandbox_profile,
                sandbox_url=self.aggregate_campaign.sandbox_url,
                sandbox_timeout=self.aggregate_campaign.sandbox_timeout,
            )
            inspector_manifest_path = inspector_workspace / WORKLOAD_BUCKETS_FILE
            manifest_text = (
                inspector_manifest_path.read_text(encoding="utf-8")
                if inspector_manifest_path.exists()
                else ""
            )
        self._account(result, "workload inspector")
        if result.exit_status != 0 or result.timed_out:
            raise RuntimeError(
                f"workload inspector failed (exit={result.exit_status}, timed_out={result.timed_out})"
            )
        if not manifest_text:
            raise RuntimeError(f"workload inspector did not produce {WORKLOAD_BUCKETS_FILE}")
        try:
            manifest = json.loads(manifest_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"workload inspector produced invalid JSON: {exc}") from exc
        buckets = validate_workload_buckets(
            manifest,
            workload_count,
            self.max_buckets,
            require_visibility_policy=True,
        )
        validate_dispatch_bucket_compatibility(buckets, signature_records)
        self.manifest_path.write_text(
            json.dumps(
                {
                    **_normalized_bucket_manifest(buckets, workload_count),
                    "workload_source": workload_source.filename,
                    "workload_ids": list(workload_source.ids),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        self._commit_main(
            f"workload inspector: partition into {len(buckets)} buckets",
            WORKLOAD_BUCKETS_FILE,
            ".gitignore",
        )
        print(
            f"[workload-inspector] validated {len(buckets)} disjoint buckets covering "
            f"{workload_count} workloads from {workload_source.filename}",
            flush=True,
        )
        return buckets

    def _load_state(self) -> dict:
        state: dict
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(state, dict):
                    state.setdefault("schema_version", 1)
                    state.setdefault("buckets", {})
                else:
                    state = {"schema_version": 1, "buckets": {}}
            except (OSError, json.JSONDecodeError):
                state = {"schema_version": 1, "buckets": {}}
        else:
            state = {"schema_version": 1, "buckets": {}}

        state.setdefault("pending_improvements", {})
        if "bootstrap" not in state:
            # Compatibility for campaigns that accepted aggregate kernels
            # before the ten-round bootstrap barrier existed.  Those campaigns
            # have already completed their first aggregate and must continue in
            # incremental mode after a restart instead of being gated again.
            legacy_accepted = any(
                bool((read_memory(self.workspace, version) or {}).get("aggregation"))
                for version in range(1, latest_version(self.workspace) + 1)
            )
            state["bootstrap"] = {
                "status": "ACCEPTED" if legacy_accepted else "PENDING",
                "minimum_iterations": INITIAL_AGGREGATION_MIN_ITERATIONS,
                "reason": (
                    "migrated existing accepted aggregate"
                    if legacy_accepted
                    else "waiting for ten rounds and one improvement from every bucket"
                ),
            }
        return state

    def _write_state(self, state: dict) -> None:
        self.state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _bootstrap_accepted(state: dict) -> bool:
        return (state.get("bootstrap") or {}).get("status") == "ACCEPTED"

    def _remember_pending_improvement(
        self,
        bucket: WorkloadBucket,
        campaign: Campaign,
        version: int,
        memory: dict,
    ) -> dict:
        """Persist the newest pre-bootstrap win for one bucket.

        This records provenance only; it never edits or validates the aggregate
        kernel.  Keeping the pending set in the main Git repository makes the
        ten-round barrier restart-safe.
        """
        state = self._load_state()
        pending = state.setdefault("pending_improvements", {})
        entry = {
            "bucket_version": f"v{version}",
            "bucket_latency_us": (memory.get("performance") or {}).get("latency_us"),
            "bucket_head": git_head(campaign.workspace),
            "bucket_kernel_blob": git_kernel_blob(campaign.workspace),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        previous = pending.get(bucket.name) or {}
        if (
            previous.get("bucket_version") == entry["bucket_version"]
            and previous.get("bucket_kernel_blob") == entry["bucket_kernel_blob"]
        ):
            return state
        pending[bucket.name] = entry
        bootstrap = state.setdefault("bootstrap", {})
        bootstrap.update(
            {
                "status": "PENDING",
                "minimum_iterations": INITIAL_AGGREGATION_MIN_ITERATIONS,
                "reason": "waiting for ten rounds and one improvement from every bucket",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write_state(state)
        self._commit_main(
            f"aggregate: defer {bucket.name} v{version} until initial all-bucket barrier",
            AGGREGATION_STATE_FILE,
        )
        print(
            f"[aggregate] deferred {bucket.name} v{version}: initial aggregation waits "
            f"for every bucket to reach v{INITIAL_AGGREGATION_MIN_ITERATIONS} and improve",
            flush=True,
        )
        return state

    def _pending_bootstrap_sources(
        self, state: dict
    ) -> Optional[list[tuple[WorkloadBucket, Campaign, int, dict]]]:
        """Resolve a complete, current all-bucket source set from persisted wins."""
        pending = state.get("pending_improvements") or {}
        if set(pending) < set(self._bucket_campaigns):
            return None
        if any(
            latest_version(campaign.workspace) < INITIAL_AGGREGATION_MIN_ITERATIONS
            for campaign in self._bucket_campaigns.values()
        ):
            return None

        manifest = {
            bucket.name: bucket
            for bucket in validate_workload_buckets(
                json.loads(self.manifest_path.read_text(encoding="utf-8")),
                len(_read_workload_source(self.op_dir).entries),
                self.max_buckets,
            )
        }
        sources: list[tuple[WorkloadBucket, Campaign, int, dict]] = []
        for name in sorted(self._bucket_campaigns):
            entry = pending.get(name) or {}
            version_text = str(entry.get("bucket_version") or "")
            match = re.fullmatch(r"v(\d+)", version_text)
            if match is None or name not in manifest:
                return None
            campaign = self._bucket_campaigns[name]
            if head_kernel_is_initial_baseline(campaign.workspace):
                return None
            version = int(match.group(1))
            memory = read_memory(campaign.workspace, version)
            if not memory or not _status_is(
                (memory.get("quality_gate") or {}).get("result"), "PASS"
            ):
                return None
            sources.append((manifest[name], campaign, version, memory))
        return sources

    def _maybe_initial_aggregation_locked(self) -> bool:
        state = self._load_state()
        if self._bootstrap_accepted(state):
            return False
        sources = self._pending_bootstrap_sources(state)
        if not sources:
            return False
        fingerprint = {
            bucket.name: git_kernel_blob(campaign.workspace)
            for bucket, campaign, _version, _memory in sources
        }
        bootstrap = state.get("bootstrap") or {}
        if (
            bootstrap.get("status") == "REJECTED"
            and bootstrap.get("source_kernel_blobs") == fingerprint
            and bootstrap.get("dispatch_schema_version")
            == AGGREGATE_DISPATCH_SCHEMA_VERSION
            and bootstrap.get("source_layout") == AGGREGATE_SOURCE_LAYOUT
        ):
            return False
        print(
            "[aggregate] initial barrier satisfied: every bucket reached "
            f"v{INITIAL_AGGREGATION_MIN_ITERATIONS} and has a committed improvement; "
            "rebuilding from all bucket HEADs",
            flush=True,
        )
        bucket, campaign, version, memory = sources[-1]
        return self.aggregate_improvement(
            bucket,
            campaign,
            version,
            memory,
            initial_sources=sources,
        )

    def record_bucket_improvement(
        self,
        bucket: WorkloadBucket,
        campaign: Campaign,
        version: int,
        memory: dict,
    ) -> bool:
        """Apply the two-phase aggregation policy to a mechanically accepted win."""
        with self._aggregate_lock:
            state = self._load_state()
            if self._bootstrap_accepted(state):
                return self.aggregate_improvement(bucket, campaign, version, memory)
            self._remember_pending_improvement(bucket, campaign, version, memory)
            return self._maybe_initial_aggregation_locked()

    def bucket_iteration_completed(
        self,
        _campaign: Campaign,
        version: int,
        _memory: Optional[dict],
        _won: bool,
    ) -> None:
        """Open the initial barrier when the slowest bucket completes round ten."""
        if version < INITIAL_AGGREGATION_MIN_ITERATIONS:
            return
        with self._aggregate_lock:
            self._maybe_initial_aggregation_locked()

    def _current_all_bucket_sources(
        self,
        override: tuple[WorkloadBucket, Campaign, int, dict],
    ) -> Optional[list[tuple[WorkloadBucket, Campaign, int, dict]]]:
        """Resolve current committed winners for one-time legacy dispatcher migration."""
        workload_source = _read_workload_source(self.op_dir)
        manifest = {
            bucket.name: bucket
            for bucket in validate_workload_buckets(
                json.loads(self.manifest_path.read_text(encoding="utf-8")),
                len(workload_source.entries),
                self.max_buckets,
            )
        }
        override_bucket, override_campaign, override_version, override_memory = override
        resolved: list[tuple[WorkloadBucket, Campaign, int, dict]] = []
        for name in sorted(manifest):
            if name == override_bucket.name:
                resolved.append(
                    (
                        override_bucket,
                        override_campaign,
                        override_version,
                        override_memory,
                    )
                )
                continue
            campaign = self._bucket_campaigns.get(name)
            if campaign is None or head_kernel_is_initial_baseline(campaign.workspace):
                return None
            best_before: Optional[float] = None
            latest_win: Optional[tuple[int, dict]] = None
            for candidate_version in range(0, latest_version(campaign.workspace) + 1):
                candidate_memory = read_memory(campaign.workspace, candidate_version)
                if not candidate_memory or not _status_is(
                    (candidate_memory.get("quality_gate") or {}).get("result"), "PASS"
                ):
                    continue
                latency = (candidate_memory.get("performance") or {}).get("latency_us")
                if not isinstance(latency, (int, float)):
                    continue
                if (
                    candidate_version > 0
                    and (best_before is None or float(latency) < best_before)
                ):
                    latest_win = (candidate_version, candidate_memory)
                best_before = (
                    float(latency)
                    if best_before is None
                    else min(best_before, float(latency))
                )
            if latest_win is None:
                return None
            candidate_version, candidate_memory = latest_win
            resolved.append(
                (manifest[name], campaign, candidate_version, candidate_memory)
            )
        return resolved

    def _dispatch_inputs(self) -> tuple[WorkloadSource, list[WorkloadBucket], list[dict]]:
        workload_source = _read_workload_source(self.op_dir)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        buckets = validate_workload_buckets(
            manifest, len(workload_source.entries), self.max_buckets
        )
        signature_payload = json.loads(
            (self.workspace / DISPATCH_SIGNATURES_FILE).read_text(encoding="utf-8")
        )
        signatures = validate_dispatch_signatures(signature_payload, workload_source)
        validate_dispatch_bucket_compatibility(buckets, signatures)
        return workload_source, buckets, signatures

    def _materialize_deterministic_dispatch(
        self,
        sources: list[tuple[WorkloadBucket, Campaign, int, dict]],
        *,
        initial: bool,
    ) -> None:
        """Embed committed bucket kernels into one exact static kernel.py."""
        workload_source, buckets, signatures = self._dispatch_inputs()
        expected_names = {bucket.name for bucket in buckets}
        module_dir = self.workspace / AGGREGATE_KERNELS_DIR
        dispatch_path = self.workspace / AGGREGATE_DISPATCH_FILE
        existing_records: dict[str, dict] = {}
        if not initial:
            if not dispatch_path.is_file():
                raise RuntimeError(
                    "incremental deterministic aggregation requires an accepted dispatcher"
                )
            try:
                existing_manifest = json.loads(dispatch_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid accepted dispatcher manifest: {exc}") from exc
            if existing_manifest.get("mode") != "deterministic_dispatch":
                raise RuntimeError("accepted aggregate is not a deterministic dispatcher")
            if existing_manifest.get("schema_version") not in {1, 2, 3}:
                raise RuntimeError("accepted dispatcher has an unsupported schema")
            raw_modules = existing_manifest.get("modules")
            if not isinstance(raw_modules, dict):
                raise RuntimeError("accepted dispatcher manifest has no modules")
            existing_records = {
                str(name): dict(record)
                for name, record in raw_modules.items()
                if isinstance(record, dict)
            }

        module_records = existing_records
        module_sources: dict[str, str] = {}
        for source_bucket, source_campaign, source_version, _source_memory in sources:
            if source_bucket.name not in expected_names:
                raise RuntimeError(f"unknown aggregate bucket: {source_bucket.name}")
            source = _git_head_file(source_campaign.workspace, "kernel.py")
            module_sources[source_bucket.name] = source
            module_records[source_bucket.name] = {
                "embedded": True,
                "bucket_version": f"v{source_version}",
                "bucket_head": git_head(source_campaign.workspace),
                "kernel_blob": git_kernel_blob(source_campaign.workspace),
            }

        if set(module_records) != expected_names:
            missing = sorted(expected_names - set(module_records))
            extra = sorted(set(module_records) - expected_names)
            raise RuntimeError(
                f"deterministic dispatcher source mismatch: missing={missing}, extra={extra}"
            )

        # Incremental schema-v2 aggregates recover unchanged source from the exact
        # recorded Git blob. Schema-v1 aggregates read their tracked module once
        # and are migrated to the single-file format by this regeneration.
        for name, record in module_records.items():
            if name in module_sources:
                continue
            legacy_path = str(record.get("path") or "")
            legacy_source = self.workspace / legacy_path if legacy_path else None
            if legacy_source is not None and legacy_source.is_file():
                module_sources[name] = legacy_source.read_text(encoding="utf-8")
            else:
                campaign = self._bucket_campaigns.get(name)
                if campaign is None:
                    raise RuntimeError(f"no bucket workspace available for aggregate source: {name}")
                module_sources[name] = _git_blob_file(
                    campaign.workspace, record.get("kernel_blob")
                )
            record.pop("path", None)
            record["embedded"] = True

        bucket_by_index = {
            index: bucket.name
            for bucket in buckets
            for index in bucket.workload_indices
        }
        dispatcher = build_deterministic_dispatcher(
            kind=workload_source.kind,
            signature_records=signatures,
            bucket_by_index=bucket_by_index,
            module_records=module_records,
            module_sources=module_sources,
        )
        if module_dir.exists():
            shutil.rmtree(module_dir)
        (self.workspace / "kernel.py").write_text(dispatcher, encoding="utf-8")
        dispatch_manifest = {
            "schema_version": AGGREGATE_DISPATCH_SCHEMA_VERSION,
            "mode": "deterministic_dispatch",
            "dispatch_strategy": "structural_range_tree_v1",
            "source_layout": AGGREGATE_SOURCE_LAYOUT,
            "dispatch_visibility_policy": DISPATCH_VISIBILITY_POLICY,
            "kind": workload_source.kind,
            "workload_source": workload_source.filename,
            "signature_source": DISPATCH_SIGNATURES_FILE,
            "modules": {name: module_records[name] for name in sorted(module_records)},
        }
        dispatch_path.write_text(
            json.dumps(dispatch_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _restore_aggregate_incumbent(self, pre_head: str) -> None:
        """Reset tracked candidates and remove only generated untracked paths."""
        subprocess.run(
            ["git", "reset", "--hard", pre_head],
            cwd=str(self.workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        untracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                AGGREGATE_KERNELS_DIR,
                AGGREGATE_DISPATCH_FILE,
            ],
            cwd=str(self.workspace),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for relative in untracked:
            path = (self.workspace / relative).resolve()
            if self.workspace.resolve() not in path.parents:
                raise RuntimeError(f"refusing to remove aggregate path outside workspace: {path}")
            if path.is_file() or path.is_symlink():
                path.unlink()
        module_dir = self.workspace / AGGREGATE_KERNELS_DIR
        if module_dir.is_dir() and not any(module_dir.iterdir()):
            module_dir.rmdir()

    def _record_aggregation_state(
        self,
        *,
        bucket: WorkloadBucket,
        bucket_head: str,
        bucket_kernel_blob: str,
        bucket_version: int,
        bucket_latency: object,
        status: str,
        reason: str,
        aggregate_latency: object = None,
        state: Optional[dict] = None,
    ) -> dict:
        if state is None:
            state = self._load_state()
        state["buckets"][bucket.name] = {
            "last_seen_head": bucket_head,
            "last_seen_kernel_blob": bucket_kernel_blob,
            "bucket_version": f"v{bucket_version}",
            "bucket_latency_us": bucket_latency,
            "status": status,
            "reason": reason,
            "aggregate_latency_us": aggregate_latency,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return state

    def _refresh_aggregate_solution_metadata(self) -> None:
        """Union framework metadata from bucket solutions into the main solution."""
        solution_path = self.workspace / "solution.json"
        if not solution_path.exists():
            return
        try:
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        spec = solution.setdefault("spec", {})
        languages = list(spec.get("languages") or [])
        dependencies = list(spec.get("dependencies") or [])
        for campaign in self._bucket_campaigns.values():
            bucket_solution_path = campaign.workspace / "solution.json"
            if not bucket_solution_path.exists():
                continue
            try:
                bucket_solution = json.loads(
                    bucket_solution_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            bucket_spec = bucket_solution.get("spec") or {}
            for language in bucket_spec.get("languages") or []:
                if language not in languages:
                    languages.append(language)
            for dependency in bucket_spec.get("dependencies") or []:
                if dependency not in dependencies:
                    dependencies.append(dependency)
        spec["languages"] = languages
        spec["dependencies"] = dependencies
        dispatch_path = self.workspace / AGGREGATE_DISPATCH_FILE
        if dispatch_path.is_file():
            solution["sources"] = [{"path": "kernel.py"}]
        solution["description"] = (
            "Self-contained deterministic full-workload dispatcher over independently "
            "optimized buckets"
        )
        solution_path.write_text(
            json.dumps(solution, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def aggregate_improvement(
        self,
        bucket: WorkloadBucket,
        campaign: Campaign,
        version: int,
        memory: dict,
        *,
        initial_sources: Optional[
            list[tuple[WorkloadBucket, Campaign, int, dict]]
        ] = None,
    ) -> bool:
        """Serialize aggregate edits while bucket optimization threads keep running.

        ``initial_sources`` is supplied exactly for the first accepted aggregate:
        it forces one candidate to be rebuilt from every bucket HEAD after the
        ten-round warmup.  Later calls carry one changed bucket and preserve the
        established aggregate paths.
        """
        with self._aggregate_lock:
            if initial_sources is None and not (
                self.workspace / AGGREGATE_DISPATCH_FILE
            ).is_file():
                migration_sources = self._current_all_bucket_sources(
                    (bucket, campaign, version, memory)
                )
                if migration_sources is not None:
                    initial_sources = migration_sources
                    print(
                        "[aggregate] migrating legacy semantic aggregate to deterministic dispatch",
                        flush=True,
                    )
            is_initial = initial_sources is not None
            sources = initial_sources or [(bucket, campaign, version, memory)]
            bucket_head = git_head(campaign.workspace)
            bucket_kernel_blob = git_kernel_blob(campaign.workspace)
            state = self._load_state()
            previous = (state.get("buckets") or {}).get(bucket.name) or {}
            if (
                not is_initial
                and bucket_kernel_blob
                and previous.get("last_seen_kernel_blob") == bucket_kernel_blob
            ):
                return False
            # Backward-compatible resume for state written before blob ids were
            # recorded. Metadata-only bucket commits must not trigger a rebuild.
            if (
                not is_initial
                and
                not previous.get("last_seen_kernel_blob")
                and bucket_head
                and previous.get("last_seen_head") == bucket_head
            ):
                return False

            bucket_latency = (memory.get("performance") or {}).get("latency_us")
            incumbent_latency_us = best_validated_latency_us(self.workspace)
            aggregate_version = latest_version(self.workspace) + 1
            source_kernel_blobs = {
                source_bucket.name: git_kernel_blob(source_campaign.workspace)
                for source_bucket, source_campaign, _source_version, _source_memory in sources
            }
            if is_initial:
                source_names = ", ".join(source_bucket.name for source_bucket, *_ in sources)
                print(
                    f"[aggregate] initial deterministic dispatch from: {source_names}",
                    flush=True,
                )
            else:
                print(
                    f"[aggregate] bucket={bucket.name} v{version} improved to "
                    f"{bucket_latency} us; updating deterministic dispatch immediately",
                    flush=True,
                )
            pre_head = git_head(self.workspace)
            if not pre_head:
                raise RuntimeError("aggregate workspace has no git incumbent")
            try:
                self._materialize_deterministic_dispatch(
                    sources,
                    initial=is_initial,
                )
                self._refresh_aggregate_solution_metadata()
            except Exception as exc:
                try:
                    self._restore_aggregate_incumbent(pre_head)
                except Exception as cleanup_exc:
                    exc = RuntimeError(f"{exc}; rollback failed: {cleanup_exc}")
                reason = f"deterministic dispatch generation failed: {exc}"
                state = self._record_aggregation_state(
                    bucket=bucket,
                    bucket_head=bucket_head,
                    bucket_kernel_blob=bucket_kernel_blob,
                    bucket_version=version,
                    bucket_latency=bucket_latency,
                    status="REJECTED",
                    reason=reason,
                )
                if is_initial:
                    state["bootstrap"] = {
                        "status": "REJECTED",
                        "minimum_iterations": INITIAL_AGGREGATION_MIN_ITERATIONS,
                        "source_kernel_blobs": source_kernel_blobs,
                        "dispatch_schema_version": AGGREGATE_DISPATCH_SCHEMA_VERSION,
                        "source_layout": AGGREGATE_SOURCE_LAYOUT,
                        "reason": reason,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                self._write_state(state)
                self._commit_main(
                    (
                        "aggregate: reject initial all-bucket bootstrap"
                        if is_initial
                        else f"aggregate: reject {bucket.name} v{version}"
                    ),
                    AGGREGATION_STATE_FILE,
                )
                print(
                    f"[aggregate] rejected "
                    f"{'initial all-bucket bootstrap' if is_initial else f'{bucket.name} v{version}'}: "
                    f"{reason}",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            kernel_path = self.workspace / "kernel.py"
            candidate = kernel_path.read_text(encoding="utf-8") if kernel_path.exists() else ""

            rejection = ""
            test_result: Optional[dict] = None
            if not candidate:
                rejection = "deterministic aggregation produced no kernel.py"
            else:
                changed = subprocess.run(
                    [
                        "git",
                        "status",
                        "--porcelain",
                        "--untracked-files=all",
                        "--",
                        "kernel.py",
                        AGGREGATE_KERNELS_DIR,
                        AGGREGATE_DISPATCH_FILE,
                        "solution.json",
                    ],
                    cwd=str(self.workspace),
                    capture_output=True,
                    text=True,
                )
                if changed.returncode != 0:
                    rejection = "could not compare aggregate kernel candidate"
                elif not changed.stdout.strip():
                    rejection = "deterministic aggregation did not change aggregate sources"

            if not rejection and self.aggregate_campaign.optimization_mode == "production":
                violations = self.aggregate_campaign._production_kernel_violations(
                    self.workspace
                )
                if violations:
                    rejection = "production policy: " + "; ".join(violations)

            if not rejection:
                validation_timeout = max(
                    self.aggregate_campaign.sandbox_timeout,
                    AGGREGATE_VALIDATION_TIMEOUT,
                )
                validation_stages = [
                    (
                        "single-seed",
                        [
                            "python", "test_kernel.py", "--version",
                            f"v{aggregate_version}", "--no-memory",
                        ],
                    ),
                    (
                        "multi-seed",
                        [
                            "python", "test_kernel.py", "--version",
                            f"v{aggregate_version}", "--multi-seed", "5", "--no-memory",
                        ],
                    ),
                ]
                for stage_name, validation_command in validation_stages:
                    try:
                        test = _sandbox_command(
                            self.workspace,
                            self.aggregate_campaign.sandbox_hardware,
                            self.aggregate_campaign.sandbox_profile,
                            self.aggregate_campaign.sandbox_url,
                            validation_timeout,
                            validation_command,
                            wall_timeout=(
                                validation_timeout + AGGREGATE_QUEUE_WAIT_GRACE
                            ),
                            gateway_kind="run",
                        )
                    except (OSError, subprocess.SubprocessError) as exc:
                        rejection = (
                            f"full-workload {stage_name} validation failed to run: {exc}"
                        )
                        break
                    if test.stdout:
                        print(
                            test.stdout,
                            end="" if test.stdout.endswith("\n") else "\n",
                            flush=True,
                        )
                    if test.stderr:
                        print(
                            test.stderr,
                            end="" if test.stderr.endswith("\n") else "\n",
                            file=sys.stderr,
                            flush=True,
                        )
                    if test.returncode != 0:
                        rejection = (
                            f"full-workload {stage_name} validation command failed "
                            f"(exit={test.returncode})"
                        )
                        break
                    try:
                        stage_result = _test_result_from_stdout(test.stdout)
                    except (RuntimeError, json.JSONDecodeError) as exc:
                        rejection = (
                            f"full-workload {stage_name} validation produced no usable "
                            f"result: {exc}"
                        )
                        break
                    test_result = stage_result
                    if not stage_result.get("all_pass"):
                        rejection = (
                            f"full-workload {stage_name} correctness validation failed"
                        )
                        break

            aggregate_latency = (
                test_result.get("latency_us_geomean") if test_result is not None else None
            )
            if not rejection and not isinstance(aggregate_latency, (int, float)):
                rejection = "full-workload validation returned no latency"
            if not rejection and incumbent_latency_us is not None:
                required = incumbent_latency_us * (1.0 - self.aggregate_min_improvement)
                if float(aggregate_latency) >= required:
                    rejection = (
                        f"full-workload latency {float(aggregate_latency):.6g} us did not beat "
                        f"incumbent {incumbent_latency_us:.6g} us"
                    )

            if rejection:
                self._restore_aggregate_incumbent(pre_head)
                state = self._record_aggregation_state(
                    bucket=bucket,
                    bucket_head=bucket_head,
                    bucket_kernel_blob=bucket_kernel_blob,
                    bucket_version=version,
                    bucket_latency=bucket_latency,
                    status="REJECTED",
                    reason=rejection,
                    aggregate_latency=aggregate_latency,
                )
                if is_initial:
                    state["bootstrap"] = {
                        "status": "REJECTED",
                        "minimum_iterations": INITIAL_AGGREGATION_MIN_ITERATIONS,
                        "source_kernel_blobs": source_kernel_blobs,
                        "dispatch_schema_version": AGGREGATE_DISPATCH_SCHEMA_VERSION,
                        "source_layout": AGGREGATE_SOURCE_LAYOUT,
                        "reason": rejection,
                        "aggregate_latency_us": aggregate_latency,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                self._write_state(state)
                self._commit_main(
                    (
                        "aggregate: reject initial all-bucket bootstrap"
                        if is_initial
                        else f"aggregate: reject {bucket.name} v{version}"
                    ),
                    AGGREGATION_STATE_FILE,
                )
                print(
                    f"[aggregate] rejected "
                    f"{'initial all-bucket bootstrap' if is_initial else f'{bucket.name} v{version}'}: "
                    f"{rejection}",
                    flush=True,
                )
                return False

            assert test_result is not None
            memory_path = _record_local_test_result(
                self.workspace, f"v{aggregate_version}", test_result
            )
            aggregate_memory = json.loads(memory_path.read_text(encoding="utf-8"))
            aggregate_memory["aggregation"] = {
                "mode": (
                    "deterministic_dispatch_bootstrap"
                    if is_initial
                    else "deterministic_dispatch_incremental"
                ),
                "source_bucket": bucket.name,
                "source_bucket_version": f"v{version}",
                "source_bucket_head": bucket_head,
                "source_bucket_kernel_blob": bucket_kernel_blob,
                "previous_aggregate_head": pre_head,
            }
            if is_initial:
                aggregate_memory["aggregation"].update(
                    {
                        "sources": [
                            {
                                "bucket": source_bucket.name,
                                "version": f"v{source_version}",
                                "head": git_head(source_campaign.workspace),
                                "kernel_blob": git_kernel_blob(source_campaign.workspace),
                            }
                            for source_bucket, source_campaign, source_version, _source_memory
                            in sources
                        ],
                    }
                )
            aggregate_memory["git_commit_hash"] = None
            memory_path.write_text(
                json.dumps(aggregate_memory, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            state = self._record_aggregation_state(
                bucket=bucket,
                bucket_head=bucket_head,
                bucket_kernel_blob=bucket_kernel_blob,
                bucket_version=version,
                bucket_latency=bucket_latency,
                status="ACCEPTED",
                reason=(
                    "deterministic dispatch passed full-workload correctness and improved geomean"
                ),
                aggregate_latency=aggregate_latency,
            )
            if is_initial:
                state["bootstrap"] = {
                    "status": "ACCEPTED",
                    "minimum_iterations": INITIAL_AGGREGATION_MIN_ITERATIONS,
                    "source_kernel_blobs": source_kernel_blobs,
                    "dispatch_mode": "deterministic_dispatch",
                    "dispatch_schema_version": AGGREGATE_DISPATCH_SCHEMA_VERSION,
                    "source_layout": AGGREGATE_SOURCE_LAYOUT,
                    "reason": (
                        "deterministic dispatch passed full-workload correctness and improved geomean"
                    ),
                    "aggregate_latency_us": aggregate_latency,
                    "aggregate_version": f"v{aggregate_version}",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            self._write_state(state)
            commit_paths = [
                "kernel.py",
                AGGREGATE_DISPATCH_FILE,
                str(memory_path.relative_to(self.workspace)),
                AGGREGATION_STATE_FILE,
            ]
            legacy_modules = subprocess.run(
                ["git", "ls-files", "--", AGGREGATE_KERNELS_DIR],
                cwd=str(self.workspace),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if legacy_modules or (self.workspace / AGGREGATE_KERNELS_DIR).exists():
                commit_paths.append(AGGREGATE_KERNELS_DIR)
            if (self.workspace / "solution.json").exists():
                commit_paths.append("solution.json")
            self._commit_main(
                (
                    f"aggregate: accept initial all-bucket bootstrap "
                    f"({float(aggregate_latency):.3f} us)"
                    if is_initial
                    else f"aggregate: accept {bucket.name} v{version} "
                    f"({float(aggregate_latency):.3f} us)"
                ),
                *commit_paths,
            )
            print(
                f"[aggregate] accepted "
                f"{'initial all-bucket bootstrap' if is_initial else f'{bucket.name} v{version}'}: "
                "full-workload latency "
                f"{float(aggregate_latency):.6g} us, git={git_head(self.workspace)[:8]}",
                flush=True,
            )
            return True

    def _make_bucket_campaign(
        self,
        bucket: WorkloadBucket,
        workload_source: WorkloadSource,
        generalization_evidence: dict,
    ) -> Campaign:
        bucket_root = self.workspace / BUCKETS_DIR
        bucket_op = bucket_root / "ops" / bucket.name
        bucket_runs = bucket_root / "runs"
        # Campaign.setup_baseline() invokes workspace_init.sh with work_dir as
        # its cwd.  Materialize this shared parent before any bucket threads
        # start so every campaign sees a valid cwd.
        bucket_runs.mkdir(parents=True, exist_ok=True)
        _materialize_bucket_op(
            self.op_dir,
            bucket_op,
            bucket,
            workload_source,
            generalization_evidence,
        )
        parent = self.aggregate_campaign

        def on_improvement(campaign: Campaign, version: int, memory: dict) -> None:
            self.record_bucket_improvement(bucket, campaign, version, memory)

        def on_iteration(
            campaign: Campaign, version: int, memory: Optional[dict], won: bool
        ) -> None:
            self.bucket_iteration_completed(campaign, version, memory, won)

        campaign = Campaign(
            name=bucket.name,
            kernel_demo=str(bucket_op / "reference.py"),
            platform=parent.platform,
            framework=parent.framework,
            notes=(
                f"Workload bucket {bucket.name}: {bucket.rationale or 'inspector-grouped workloads'}. "
                f"The immutable {WORKLOAD_BUCKET_CONTRACT_FILE} records an unseen one-step "
                "neighbor that this bucket must support. Optimize the complete routed shape "
                "regime, not an enumeration of benchmark shapes; retain bounds-safe dynamic "
                "launch/grid logic and a generic in-regime path. The coordinator owns "
                "full-kernel aggregation."
            ),
            arch=parent.arch,
            work_dir=str(bucket_runs),
            workspace_suffix=parent.workspace_suffix,
            max_iters=parent.max_iters,
            token_budget=parent.token_budget,
            target_util=parent.target_util,
            iter_timeout=parent.iter_timeout,
            setup_timeout=parent.setup_timeout,
            framework_baseline="never",
            framework_baseline_timeout=parent.framework_baseline_timeout,
            handoff_resumes=parent.handoff_resumes,
            verify_repeats=parent.verify_repeats,
            verify_run_timeout=parent.verify_run_timeout,
            min_improvement_pct=parent.min_improvement_pct,
            max_stall=parent.max_stall,
            convert_after=parent.convert_after,
            sandbox_hardware=parent.sandbox_hardware,
            sandbox_profile=parent.sandbox_profile,
            sandbox_url=parent.sandbox_url,
            sandbox_timeout=parent.sandbox_timeout,
            atrex_bench_root=parent.atrex_bench_root,
            agent_cli=parent.agent_cli,
            optimization_mode=parent.optimization_mode,
            on_improvement=on_improvement,
            on_iteration=on_iteration,
        )
        return campaign

    def _seed_bucket_baseline_from_aggregate(
        self,
        bucket: WorkloadBucket,
        campaign: Campaign,
        workload_source: WorkloadSource,
    ) -> None:
        """Create a bucket V0 from the already-validated aggregate baseline.

        This is deliberately mechanical: no coding-agent session and no GPU
        command are involved.  The bucket receives the aggregate's pinned
        framework-baseline kernel (or its original V0 kernel when the campaign has none)
        plus only its own workload files and per-workload measurements.
        """
        if self.aggregate_campaign.framework_baseline == "never":
            baseline_commit, baseline_version = "", 0
        else:
            baseline_commit, baseline_version = resolve_framework_baseline_commit(self.workspace)
        if latest_version(campaign.workspace) >= 0:
            self._assert_bucket_baseline_provenance(bucket, campaign, baseline_commit)
            return
        aggregate_memory = read_memory(self.workspace, baseline_version)
        if aggregate_memory is None:
            raise RuntimeError(
                "cannot seed bucket baseline before aggregate "
                f"memory/v{baseline_version}.json exists"
            )
        aggregate_passed = _status_is(
            (aggregate_memory.get("quality_gate") or {}).get("result"), "PASS"
        ) or _status_is(
            (aggregate_memory.get("correctness") or {}).get("status"), "PASS"
        )
        if not aggregate_passed:
            raise RuntimeError(
                f"cannot seed bucket baseline from a non-passing aggregate v{baseline_version}"
            )

        aggregate_baseline_commit = baseline_commit or v0_baseline_commit(self.workspace)
        if not aggregate_baseline_commit:
            raise RuntimeError("aggregate workspace has no committed V0 kernel.py")

        def baseline_file(name: str, *, required: bool = False) -> Optional[str]:
            result = subprocess.run(
                ["git", "show", f"{aggregate_baseline_commit}:{name}"],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout
            if required:
                raise RuntimeError(
                    f"aggregate baseline commit {aggregate_baseline_commit[:8]} has no {name}"
                )
            return None

        workspace = campaign.workspace
        for directory in (workspace, workspace / "memory", workspace / "plans", workspace / "profiles"):
            directory.mkdir(parents=True, exist_ok=True)
        if not (workspace / ".git").exists():
            subprocess.run(
                ["git", "init"], cwd=str(workspace), check=True, stdout=subprocess.DEVNULL
            )
            subprocess.run(
                ["git", "config", "user.email", "gpu-kernel-optimizer@local"],
                cwd=str(workspace),
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "GPU Kernel Optimizer"],
                cwd=str(workspace),
                check=True,
            )

        bucket_op = Path(campaign.kernel_demo).resolve().parent
        tracked_files: set[str] = set()
        for source in bucket_op.iterdir():
            if not source.is_file():
                continue
            shutil.copy2(source, workspace / source.name)
            tracked_files.add(source.name)

        # These files define the runnable baseline and must come from the aggregate's validated
        # framework-baseline commit (or its original V0 commit when the campaign has none),
        # never from a later dispatcher HEAD.
        for name in (
            "kernel.py",
            "solution.json",
            "test_kernel.py",
            "profile_driver.py",
            "config.json",
            ".gitignore",
            "CLAUDE.md",
        ):
            content = baseline_file(name, required=name == "kernel.py")
            if content is None and name in {"test_kernel.py", "profile_driver.py"}:
                aggregate_harness = self.workspace / name
                if aggregate_harness.is_file():
                    content = aggregate_harness.read_text(encoding="utf-8")
                elif name == "test_kernel.py":
                    raise RuntimeError("aggregate workspace has no immutable test_kernel.py")
            if content is not None:
                (workspace / name).write_text(content, encoding="utf-8")
                tracked_files.add(name)

        source_label = (
            f"framework baseline v{baseline_version}" if baseline_version > 0 else "V0"
        )
        selected_ids = [workload_source.ids[index] for index in bucket.workload_indices]
        readme = baseline_file("README.md") or "# GPU kernel optimization\n"
        (workspace / "README.md").write_text(
            f"# Workload bucket: {bucket.name}\n\n"
            f"This workspace V0 is derived from aggregate {source_label} "
            f"`{aggregate_baseline_commit[:12]}`. "
            f"It contains workload ids {', '.join(selected_ids)} and was not re-evaluated.\n\n"
            + readme,
            encoding="utf-8",
        )
        tracked_files.add("README.md")
        aggregate_report = baseline_file("baseline_report.md") or ""
        (workspace / "baseline_report.md").write_text(
            f"# Derived bucket V0 baseline — {bucket.name}\n\n"
            f"This baseline reuses the aggregate {source_label} kernel, correctness result, and the "
            "per-workload timings selected below. No separate baseline agent or GPU run was used.\n\n"
            f"- Aggregate baseline: {source_label} at `{aggregate_baseline_commit}`\n"
            f"- Workload indices: {', '.join(map(str, bucket.workload_indices))}\n"
            f"- Workload ids: {', '.join(selected_ids)}\n\n"
            + aggregate_report,
            encoding="utf-8",
        )
        tracked_files.add("baseline_report.md")

        derived_memory = _bucket_baseline_memory(
            aggregate_memory,
            bucket,
            workload_source,
            aggregate_workspace=self.workspace,
            aggregate_baseline_commit=aggregate_baseline_commit,
            aggregate_baseline_version=baseline_version,
        )
        memory_path = workspace / "memory" / "v0.json"
        memory_path.write_text(
            json.dumps(derived_memory, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tracked_files.add("memory/v0.json")

        campaign._link_runtime()
        tracked_files.update(name for name in (".gitignore", "CLAUDE.md") if (workspace / name).is_file())
        # A previously interrupted setup may have staged arbitrary files.  Keep
        # the worktree intact, but rebuild the index so this mechanical commit
        # contains only the deterministic V0 files listed above.
        if git_head(workspace):
            subprocess.run(
                ["git", "reset"],
                cwd=str(workspace),
                check=True,
                stdout=subprocess.DEVNULL,
            )
        else:
            subprocess.run(["git", "read-tree", "--empty"], cwd=str(workspace), check=True)
        subprocess.run(
            ["git", "add", "--", *sorted(tracked_files)],
            cwd=str(workspace),
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"V0: derive bucket baseline from aggregate {source_label}"],
            cwd=str(workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        first_commit = git_head(workspace)
        derived_memory["git_commit_hash"] = first_commit
        memory_path.write_text(
            json.dumps(derived_memory, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "memory/v0.json"], cwd=str(workspace), check=True)
        subprocess.run(
            ["git", "commit", "--amend", "--no-edit"],
            cwd=str(workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print(
            f"[workload-coordinator] derived {bucket.name} V0 from aggregate {source_label} "
            f"for {len(bucket.workload_indices)} workloads (no agent/GPU rerun)",
            flush=True,
        )

    def _ensure_bucket_generalization_contract(
        self, bucket: WorkloadBucket, campaign: Campaign
    ) -> None:
        """Install the current immutable range contract into new and resumed buckets."""
        source = Path(campaign.kernel_demo).resolve().parent / WORKLOAD_BUCKET_CONTRACT_FILE
        destination = campaign.workspace / WORKLOAD_BUCKET_CONTRACT_FILE
        content = source.read_text(encoding="utf-8")
        if destination.is_file() and destination.read_text(encoding="utf-8") == content:
            return
        destination.write_text(content, encoding="utf-8")
        subprocess.run(
            ["git", "add", "--", WORKLOAD_BUCKET_CONTRACT_FILE],
            cwd=str(campaign.workspace),
            check=True,
        )
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"workload coordinator: certify generalized bucket {bucket.name}",
            ],
            cwd=str(campaign.workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )

    def _assert_bucket_baseline_provenance(
        self, bucket: WorkloadBucket, campaign: Campaign, baseline_commit: str
    ) -> None:
        """Refuse to mix baselines: an already-seeded bucket must match the pinned baseline.

        Continuing silently would leave the aggregate advertising a framework baseline while some
        bucket sources still descend from the PyTorch V0 — invisible from the outside and exactly
        the provenance confusion the pin exists to prevent. Re-seeding is not an option either: it
        would destroy that bucket's optimization history.
        """
        if not baseline_commit:
            return
        derivation = (read_memory(campaign.workspace, 0) or {}).get("baseline_derivation") or {}
        seeded_from = _normalize_commit_hash(
            derivation.get("aggregate_baseline_commit") or derivation.get("aggregate_v0_commit")
        )
        if not seeded_from or seeded_from == baseline_commit:
            return
        message = (
            f"bucket {bucket.name} was seeded from aggregate commit {seeded_from[:12]} but the "
            f"aggregate now pins framework baseline {baseline_commit[:12]}; re-seeding would "
            "destroy that bucket's history. Continue with --framework-baseline never to keep the "
            "existing bucket baselines, or start a fresh --workspace."
        )
        if self.aggregate_campaign.optimization_mode == "production":
            raise RuntimeError(message)
        print(f"[workload-coordinator] WARNING: {message}", file=sys.stderr, flush=True)

    def _reconcile_resumed_bucket(
        self, bucket: WorkloadBucket, campaign: Campaign
    ) -> None:
        """Aggregate a committed win that survived a coordinator interruption."""
        current_blob = git_kernel_blob(campaign.workspace)
        if not current_blob:
            return
        state = self._load_state()
        bootstrap_pending = not self._bootstrap_accepted(state)
        previous = (state.get("buckets") or {}).get(bucket.name) or {}
        previous_reason = str(previous.get("reason") or "")
        retry_incomplete_validation = (
            previous.get("status") == "REJECTED"
            and "sandbox test output has no structured RESULT_JSON line" in previous_reason
        )
        if (
            not bootstrap_pending
            and
            previous.get("last_seen_kernel_blob") == current_blob
            and not retry_incomplete_validation
        ):
            return
        if bootstrap_pending and head_kernel_is_initial_baseline(campaign.workspace):
            return

        best_before: Optional[float] = None
        improvement: Optional[tuple[int, dict]] = None
        for version in range(0, latest_version(campaign.workspace) + 1):
            memory = read_memory(campaign.workspace, version)
            if not memory or not _status_is(
                (memory.get("quality_gate") or {}).get("result"), "PASS"
            ):
                continue
            latency = (memory.get("performance") or {}).get("latency_us")
            if not isinstance(latency, (int, float)):
                continue
            if version > 0 and (best_before is None or float(latency) < best_before):
                improvement = (version, memory)
            best_before = (
                float(latency) if best_before is None else min(best_before, float(latency))
            )
        if improvement is not None:
            version, memory = improvement
            print(
                f"[workload-coordinator] recovering unaggregated {bucket.name} v{version}",
                flush=True,
            )
            if bootstrap_pending:
                self.record_bucket_improvement(bucket, campaign, version, memory)
            else:
                self.aggregate_improvement(bucket, campaign, version, memory)

    def run(self) -> str:
        self._ensure_main_workspace()
        buckets = self.inspect_workloads()
        workload_source, validated_buckets, signatures = self._dispatch_inputs()
        if validated_buckets != buckets:
            raise RuntimeError("workload bucket manifest changed during inspection")
        bucket_by_index = {
            index: bucket.name
            for bucket in buckets
            for index in bucket.workload_indices
        }
        _dispatch_plan, generalization_evidence = build_generalized_dispatch_plan(
            signatures, bucket_by_index
        )
        if sum(len(bucket.workload_indices) for bucket in buckets) != len(workload_source.entries):
            raise RuntimeError(
                f"validated workload buckets no longer match {workload_source.filename}"
            )
        self._bucket_campaigns = {
            bucket.name: self._make_bucket_campaign(
                bucket,
                workload_source,
                generalization_evidence[bucket.name],
            )
            for bucket in buckets
        }
        # A resumed bucket may contain staged work left by an interrupted
        # episode. Restore its committed boundary before installing the
        # coordinator-owned contract so that commit cannot absorb agent edits.
        for bucket in buckets:
            campaign = self._bucket_campaigns[bucket.name]
            if latest_version(campaign.workspace) >= 0:
                preserve_interrupted_tracked_changes(
                    campaign.workspace, f"resume bucket {bucket.name}"
                )
        for bucket in buckets:
            self._seed_bucket_baseline_from_aggregate(
                bucket, self._bucket_campaigns[bucket.name], workload_source
            )
            self._ensure_bucket_generalization_contract(
                bucket, self._bucket_campaigns[bucket.name]
            )
        for bucket in buckets:
            self._reconcile_resumed_bucket(bucket, self._bucket_campaigns[bucket.name])
        print(
            f"[workload-coordinator] launching {len(buckets)} bucket optimization lines in parallel; "
            f"aggregate workspace={self.workspace}",
            flush=True,
        )
        reasons: dict[str, str] = {}
        failures: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(buckets), thread_name_prefix="workload-bucket"
        ) as executor:
            future_to_name = {
                executor.submit(self._bucket_campaigns[bucket.name].run): bucket.name
                for bucket in buckets
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    reasons[name] = future.result()
                except Exception as exc:
                    failures[name] = str(exc)
                    print(
                        f"[workload-coordinator] bucket {name} failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

        if failures:
            summary = "; ".join(f"{name}: {error}" for name, error in sorted(failures.items()))
            raise RuntimeError(f"bucket optimization failures: {summary}")
        summary = ", ".join(f"{name}={reason}" for name, reason in sorted(reasons.items()))
        print(
            f"\n[workload-coordinator] all bucket lines finished; aggregate HEAD="
            f"{git_head(self.workspace)[:8]} ({summary})",
            flush=True,
        )
        if self.aggregate_campaign.optimization_mode == "production":
            violations = self.aggregate_campaign._production_kernel_violations(
                self.workspace
            )
            if violations:
                raise RuntimeError(
                    "no production-compliant aggregate kernel: " + "; ".join(violations)
                )
        if (self.workspace / "solution.json").exists():
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "reference" / "sol_finalize.py"),
                        "--workspace",
                        str(self.workspace),
                    ],
                    check=False,
                )
            except OSError:
                pass
        return summary
