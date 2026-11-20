"""Durable parent-facing tasks, exact-source evaluations and incremental candidates."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
import time
import uuid

from aka.contracts.task import Assessment, EvaluationPlan, ExecutionReceipt, Metric, TaskContext
from .binding import load_adapter, objective, validate_budget, validate_spec
from .content import atomic_json, committed_source, digest, document, identity, initialize_source, materialize, relative, tree
from .store import Store


def assessment(value):
    return Assessment(**{**value, "metrics": tuple(Metric(**m) for m in value.get("metrics", []))})


@contextmanager
def exclusive(path, *, wait=False):
    with Path(path).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise TaskBusy("This task operation already has a live owner") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class TaskBusy(ValueError):
    """Another owner is completing this operation; task evidence is unchanged."""


class TaskService:
    def __init__(self, root):
        self.store = Store(root)
        self.root = self.store.root

    def directory(self, task_id):
        self.store.task(task_id)
        return self.root / "tasks" / task_id

    def submit(self, spec_path, *, idempotency_key, start=True):
        """Freeze all caller inputs before returning an idempotent, detached task ID."""
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("An explicit idempotency key is required")
        spec_path = Path(spec_path).resolve()
        raw = json.loads(spec_path.read_text())
        if "adapter" not in raw:
            from .api import bind_contract
            with tempfile.TemporaryDirectory(prefix="aka-contract-") as temporary:
                staging = Path(temporary)
                values = {k: raw.get(k) for k in ("contract", "baseline", "seed", "workload", "validation")}
                for key, value in values.items():
                    if isinstance(value, str):
                        values[key] = spec_path.parent / value
                if isinstance(values["contract"], dict):
                    values["contract"] = {**values["contract"],
                        "source": spec_path.parent / values["contract"]["source"]}
                bound = bind_contract(staging, **values)
                spec = {k: v for k, v in raw.items() if k not in {*values, "execution"}}
                spec.update(bound)
                if "execution" in raw:
                    spec["target"] = raw["execution"]
                spec.setdefault("schema_version", 1)
                spec.setdefault("agent", {"runtime": "claude"})
                spec.setdefault("budget", {})
                atomic_json(staging / "task.json", spec)
                return self.submit(staging / "task.json", idempotency_key=idempotency_key, start=start)
        spec = validate_spec(raw)
        bundles = {role: tree(spec_path.parent / spec[role]) for role in ("contract", "seed")}
        bundles["baseline"] = tree(spec_path.parent / spec["baseline"]) if spec.get("baseline") else bundles["seed"]
        normalized = {key: value for key, value in spec.items() if key not in bundles}
        normalized["content"] = {role: identity(files) for role, files in bundles.items()}
        request_digest = digest(normalized)
        task_id = "task-" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:24]
        location = self.root / "tasks" / task_id
        location.parent.mkdir(parents=True, exist_ok=True)
        with self.store.transaction() as db:
            previous = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if previous:
                if previous["request_digest"] != request_digest:
                    raise ValueError("Idempotency key already belongs to different task content")
            else:
                with tempfile.TemporaryDirectory(prefix=".prepare-", dir=location.parent) as temporary:
                    staging = Path(temporary)
                    for role, files in bundles.items():
                        materialize(staging / role, files)
                    adapter = load_adapter(spec["adapter"], staging / "contract")
                    adapter.validate(staging / "contract", spec.get("config", {}))
                    goal = objective(adapter, spec["objective"])
                    module = sys.modules[type(adapter).__module__]
                    adapter_digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                    normalized["adapter_digest"] = adapter_digest
                    normalized["capabilities"] = sorted(adapter.capabilities)
                    normalized["contract_id"] = digest({
                        **{key: value for key, value in normalized.items()
                           if key not in {"content", "budget", "agent", "parent_task", "name", "notes"}},
                        "contract_content": normalized["content"]["contract"]})
                    normalized["instructions"] = adapter.instructions(staging / "contract", spec.get("config", {}))
                    normalized["objective_description"] = goal.describe()
                    normalized["delivery"] = adapter.delivery(staging / "contract", spec.get("config", {}))
                    initialize_source(staging / "workspace", bundles["seed"])
                    atomic_json(staging / "task.json", normalized)
                    if location.exists():
                        raise ValueError("Unregistered task directory exists; inspect it before retrying")
                    os.replace(staging, location)
                db.execute("INSERT INTO tasks(id,request_digest,manifest,state,created) VALUES(?,?,?,'ready',?)",
                           (task_id, request_digest, document(normalized), time.time()))
                self.store.emit(db, task_id, "task_created", {"contract_id": normalized["contract_id"]})
        if start:
            state = self.store.task(task_id)
            if state["control"] == "run" and state["state"] not in {"finished", "cancelled", "awaiting_candidate", "running"}:
                self.start(task_id)
        return self.status(task_id)

    def bind(self, task_id):
        task = self.store.task(task_id)
        root, spec = self.directory(task_id), task["manifest"]
        for role in ("contract", "baseline", "seed"):
            if identity(tree(root / role)) != spec["content"][role]:
                raise ValueError(f"Frozen {role} content changed; create a new task revision")
        adapter = load_adapter(spec["adapter"], root / "contract")
        module = sys.modules[type(adapter).__module__]
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != spec["adapter_digest"]:
            raise ValueError("Adapter implementation changed; use the task's original runtime or create a new task")
        if sorted(adapter.capabilities) != spec["capabilities"]:
            raise ValueError("Adapter capabilities changed")
        return task, adapter, objective(adapter, spec["objective"])

    def status(self, task_id):
        task = self.store.task(task_id)
        with self.store.connect() as db:
            count = db.execute("SELECT count(*) FROM evaluations WHERE task_id=? AND role!='research'", (task_id,)).fetchone()[0]
            executions = db.execute("SELECT count(*) FROM evaluations WHERE task_id=? AND role='research'", (task_id,)).fetchone()[0]
            submissions = db.execute("SELECT count(*) FROM submissions WHERE task_id=?", (task_id,)).fetchone()[0]
            pending = db.execute("SELECT count(*) FROM evaluations WHERE task_id=? AND state!='finished'", (task_id,)).fetchone()[0]
        if task["state"] in {"pausing", "cancelling"} and not pending:
            try:
                with exclusive(self.directory(task_id) / "worker.lock"):
                    state = "paused" if task["control"] == "pause" else "cancelled"
                    self.store.state(task_id, state)
                    task["state"] = state
            except ValueError:
                pass
        return {key: task[key] for key in ("id", "state", "control", "episodes", "tokens", "stop_reason")} | {
            "contract_id": task["manifest"]["contract_id"], "evaluations": count,
            "executions": executions, "candidates": submissions, "budget": task["manifest"].get("budget", {}),
            "workspace": str(self.directory(task_id) / "workspace")}

    def events(self, task_id, *, after=0):
        return self.store.events(task_id, after)

    def start(self, task_id, *, recover=False):
        task = self.store.task(task_id)
        if task["state"] in {"finished", "cancelled"}:
            raise ValueError("Finished tasks require a new task revision to continue")
        if task["control"] != "run" and not recover:
            raise ValueError("Task is paused/cancelled; explicitly resume a paused task before starting")
        directory = self.directory(task_id)
        # The worker lock is authoritative after crashes; a stale state string is not.
        with (directory / "worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"task_id": task_id, "already_running": True}
            fcntl.flock(lock, fcntl.LOCK_UN)
        with (directory / "worker.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, "-m", "aka.light.tasks.cli", "--root", str(self.root),
                                        "work", task_id], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        threading.Thread(target=process.wait, daemon=True).start()
        return {"task_id": task_id, "pid": process.pid}

    def control(self, task_id, action, *, budget=None):
        task = self.store.task(task_id)
        if action not in {"pause", "resume", "cancel"}:
            raise ValueError("Control must be pause, resume or cancel")
        if budget is not None:
            if action != "resume":
                raise ValueError("Budget updates accompany resume")
            validate_budget(budget)
        if task["control"] == "cancel":
            if action != "cancel":
                raise ValueError("Cancelled tasks retain candidates; continuation requires a new task")
            if task["state"] != "cancelled":
                self.start(task_id, recover=True)
            return self.status(task_id)
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row["control"] == "cancel":
                raise ValueError("Task was cancelled concurrently")
            if budget is not None:
                manifest = json.loads(row["manifest"])
                manifest["budget"] = {**manifest.get("budget", {}), **budget}
                db.execute("UPDATE tasks SET manifest=? WHERE id=?", (document(manifest), task_id))
                self.store.emit(db, task_id, "budget_updated", {"budget": manifest["budget"]})
            control = "run" if action == "resume" else action
            db.execute("UPDATE tasks SET control=? WHERE id=?", (control, task_id))
            if action == "resume" and row["state"] == "finished":
                db.execute("UPDATE tasks SET state='ready',stop_reason='' WHERE id=?", (task_id,))
            pending = db.execute("SELECT count(*) FROM evaluations WHERE task_id=? AND state!='finished'", (task_id,)).fetchone()[0]
            self.store.emit(db, task_id, "control_requested", {"action": action})
        if action == "resume":
            self.start(task_id)
        elif pending or row["state"] in {"running", "pausing", "cancelling"}:
            self.store.state(task_id, "pausing" if action == "pause" else "cancelling")
            if pending:
                self.start(task_id, recover=True)
        else:
            self.store.state(task_id, "paused" if action == "pause" else "cancelled")
        return self.status(task_id)

    def command(self, task_id, *, argv=None, files=None, outputs=None, environment=None, execution_id=None, launch_mode=None, source_mode=None):
        """Execute a research command through the task's selected Executor."""
        if execution_id:
            record = self.store.evaluation(execution_id)
            if record["task_id"] != task_id or record["role"] != "research":
                raise ValueError("Execution belongs to another task or is not a research command")
            if any(value is not None for value in (argv, files, outputs, environment, launch_mode, source_mode)):
                raise ValueError("Recovery uses only the original execution ID")
            record = self.measure(task_id, evaluation_id=execution_id)
        else:
            if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
                raise ValueError("Execution requires a command argv")
            for path in [*(files or {}), *(outputs or [])]:
                relative(path)
            research={"argv":argv,"files":files or {},"outputs":outputs or [],"environment":environment or {}}
            if source_mode is not None:
                if source_mode not in {'workspace','explicit'}:raise ValueError('Invalid research source mode')
                research['source_mode']=source_mode
            target=self.store.task(task_id)['manifest']['target']
            distributed=target.get('backend')=='remote' and target.get('config',{}).get('distributed')
            if launch_mode is not None and (launch_mode not in {'command','ranks'} or not distributed):
                raise ValueError('Launch mode requires a configured REMOTE distributed task')
            if distributed:research['launch_mode']=launch_mode or 'command'
            record = self.measure(task_id, _research=research)
        return {"execution_id": record["id"], "job_id": record["job_id"],
                "source_id": record["candidate_id"], "receipt": record["receipt"]}

    def artifact(self,task_id,artifact_id,offset=0,size=500000):
        """Read only an artifact declared by this task; never arbitrary CAS data."""
        from aka.light.execution.remote_cache import Store as ContentStore,valid_hash
        from aka.light.execution.remote_pool import state_root,PoolDriver
        import base64
        valid_hash(artifact_id)
        if type(offset) is not int or offset<0 or type(size) is not int or not 0<size<=500000:raise ValueError('Invalid artifact read range')
        task=self.store.task(task_id);found=None
        with self.store.connect() as db:
            rows=db.execute('SELECT id,job_id,receipt FROM evaluations WHERE task_id=? AND instr(receipt,?)>0',(task_id,artifact_id)).fetchall()
        for row in rows:
            payload=json.loads(row['receipt'])['payload']
            for name,entry in payload.get('artifacts',{}).items():
                if entry.get('artifact_id')==artifact_id:found=(row,payload,name,entry);break
            if found:break
        if not found:raise ValueError('Artifact does not belong to this task')
        row,payload,name,entry=found;store=ContentStore(state_root()/'artifact-v2')
        if not store.has(artifact_id,entry['bytes']):
            # Rehydrate a locally evicted artifact from its original completed
            # attempt. This path only reads; it never launches a GPU command.
            from aka.light.execution.remote import Executor
            from aka.light.execution.contract import CommandSpec
            directory=self.directory(task_id)/'evaluations'/row['id']
            e=Executor(target=task['manifest']['target'],directory=directory,execution_id=row['id'],previous_job=row['job_id'],
                       on_job=lambda _:None,control=lambda:'run')
            driver=PoolDriver(e,CommandSpec(directory,('unused',)))
            attempt=payload.get('pool',{}).get('attempt',payload.get('attempt'))
            if type(attempt) is not int or attempt<1:raise ValueError('Original pooled artifact attempt is missing')
            e.root=driver.pool_root+'/jobs/'+row['job_id']+'/attempt-'+str(attempt)
            remote=name[4:] if name.startswith('aka-distributed/') else 'workspace/'+name
            from aka.light.execution.remote import MAX_ARTIFACT_BYTES
            if entry.get('remote_only') or entry['bytes']>MAX_ARTIFACT_BYTES:
                return e._artifact_range(remote,entry,offset,size)
            data=e._download(remote,required=True)
            if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=artifact_id:raise ValueError('Original artifact changed')
            store.put(data)
        data=store.get(artifact_id)
        if offset>len(data):raise ValueError('Artifact offset exceeds size')
        return {'artifact_id':artifact_id,'bytes':len(data),'offset':offset,
                'base64':base64.b64encode(data[offset:offset+size]).decode()}

    def measure(self, task_id, *, role="candidate", scope="acceptance", evaluation_id=None, _research=None, _wait=False):
        if role not in {"candidate", "baseline"}:
            raise ValueError("Measurement role must be candidate or baseline")
        root = self.directory(task_id)
        with exclusive(root / "measurement.lock", wait=_wait):
            task, adapter, goal = self.bind(task_id)
            spec = task["manifest"]
            recover_unrecorded = False
            if not evaluation_id and _research is None and scope not in adapter.capabilities:
                raise ValueError(f"Adapter does not support {scope}")
            if not evaluation_id and task["control"] != "run":
                raise ValueError("Task is paused or cancelled")
            if evaluation_id:
                record = self.store.evaluation(evaluation_id)
                if record["task_id"] != task_id:
                    raise ValueError("Evaluation belongs to another task")
                if record["assessment"]:
                    return record
                if record["state"] in {"unknown", "submitting", "running"} and not record["job_id"] and not record["receipt"]:
                    if spec['target']['backend']=='remote' and spec['target'].get('config',{}).get('uid'):
                        recover_unrecorded = True
                    else:
                        raise ValueError("Submission outcome is unknown; inspect the original idempotency key before recovery")
                candidate_id, revision, scope, role = (record[k] for k in ("candidate_id", "revision", "scope", "role"))
                plan = EvaluationPlan(**record["plan"])
            else:
                if _research is not None:
                    from aka.light.execution.package import decode_files
                    role, scope = "research", "research"
                    files = ({} if _research.get('source_mode')=='explicit' else tree(root / "workspace", include_scratch=True))
                    cache=spec['target'].get('config',{}).get('cache',{})
                    if cache.get('mode','cas')=='cas' and cache.get('research_inputs')=='code':
                        # Explicit --input entries are applied after this filter.
                        # Scratch scripts and A/B references remain real inputs.
                        from fnmatch import fnmatch
                        patterns=['*.ncu-rep','*.ncu-rep.gz','*.nsys-rep','*.nsys-rep.gz',
                                  'scratch/research-log.md','scratch/*.log']
                        files={name:data for name,data in files.items() if not any(fnmatch(name,p) for p in patterns)}
                    files.update(decode_files(_research["files"]))
                    revision = "research:" + identity(files)
                elif role == "candidate":
                    revision, files = committed_source(root / "workspace")
                else:
                    files = tree(root / "baseline")
                    revision = "baseline:" + identity(files)
                candidate_id = identity(files)
                source = root / "sources" / candidate_id
                from .content import atomic_snapshot
                target_config=spec['target'].get('config',{})
                use_content=spec['target'].get('backend')=='remote' and 'cache' in target_config and target_config['cache'].get('mode','cas')=='cas'
                if use_content and not source.exists():
                    from aka.light.execution.remote_cache import Store as ContentStore
                    store=ContentStore(self.root/'content-v2');mid=store.put_files(files);store.materialize(mid,source)
                else:atomic_snapshot(source, files, include_ignored=role=='research')
                context = TaskContext(root / "contract", source, spec.get("config", {}), spec["target"], scope)
                from aka.light.execution.package import encode_files
                plan = (EvaluationPlan("command", {**_research, "files": encode_files(files)})
                        if _research is not None else adapter.prepare(context))
                if not isinstance(plan, EvaluationPlan):
                    raise TypeError("Adapter.prepare must return EvaluationPlan")
                if use_content and plan.kind=='command':
                    from aka.light.execution.remote_cache import Store as ContentStore
                    from aka.light.execution.package import decode_files
                    request=dict(plan.request)
                    request['files_ref']=ContentStore(self.root/'content-v2').put_files(decode_files(request.pop('files')))
                    plan=EvaluationPlan('command',request)
                from .execution import validate_plan
                validate_plan(plan, spec["target"])
                evaluation_id = "eval-" + uuid.uuid4().hex
                with self.store.transaction() as db:
                    count = db.execute("SELECT count(*) FROM evaluations WHERE task_id=? AND (role='research')=?",
                                       (task_id, role == "research")).fetchone()[0]
                    budget_key = "executions" if role == "research" else "evaluations"
                    if count >= spec.get("budget", {}).get(budget_key, 100):
                        raise ValueError(f"Task {budget_key} budget exhausted")
                    db.execute("INSERT INTO evaluations(id,task_id,candidate_id,revision,scope,role,state,plan,created) VALUES(?,?,?,?,?,?,'prepared',?,?)",
                               (evaluation_id, task_id, candidate_id, revision, scope, role, document(asdict(plan)), time.time()))
                    self.store.emit(db, task_id, "evaluation_prepared", {"evaluation_id": evaluation_id, "candidate_id": candidate_id})
                record = self.store.evaluation(evaluation_id)
            source = root / "sources" / candidate_id
            if identity(tree(source,include_ignored=role=='research',allow_empty=role=='research')) != candidate_id:
                raise ValueError("Frozen candidate source changed")
            context = TaskContext(root / "contract", source, spec.get("config", {}), spec["target"], scope)
            if plan.kind=='command' and plan.request.get('files_ref'):
                from aka.light.execution.remote_cache import Store as ContentStore
                content_store=ContentStore(self.root/'content-v2')
                try:
                    frozen=content_store.load(plan.request['files_ref'])
                    intact=all(content_store.has(e['sha256'],e['bytes']) for e in frozen['files'].values())
                except (FileNotFoundError,ValueError):intact=False
                if not intact:
                    # Recover only from the frozen source/contract, never the
                    # live Agent workspace. Verify the exact original manifest.
                    if role=='research':recovered=tree(source,include_ignored=True,allow_empty=True)
                    else:
                        from aka.light.execution.package import decode_files
                        recovered=decode_files(adapter.prepare(context).request['files'])
                    if content_store.put_files(recovered)!=plan.request['files_ref']:raise ValueError('Frozen execution dependencies cannot be restored exactly')
            from .execution import execute
            with self.store.transaction() as db:
                db.execute("UPDATE evaluations SET state='submitting' WHERE id=? AND job_id=''", (evaluation_id,))
            try:
                if record["receipt"]:
                    receipt = ExecutionReceipt(**record["receipt"])
                else:
                    receipt = execute(plan, spec["target"], root / "evaluations" / evaluation_id,
                                      evaluation_id, record["job_id"],
                                      lambda job: self.store.save_job(evaluation_id, job),
                                      lambda: self.store.task(task_id)["control"],
                                      **({'recover_unrecorded':True} if recover_unrecorded else {}))
                    with self.store.transaction() as db:
                        db.execute("UPDATE evaluations SET receipt=? WHERE id=?", (document(asdict(receipt)), evaluation_id))
                # Do not let evaluator bugs turn execution failures into passing evidence.
                if role == "research":
                    result = Assessment("unknown", False, reason="Research execution; not acceptance evidence")
                else:
                    result = adapter.interpret(context, receipt)
                    if not isinstance(result, Assessment):
                        raise TypeError("Adapter.interpret must return Assessment")
                    if receipt.status != "completed":
                        from .failures import execution_failure
                        failed = execution_failure(receipt)
                        if result.verdict != "fail" or failed.failure_kind in {"infrastructure", "unknown"}:
                            result = failed
                policy = spec.get("config", {}).get("implementation", {})
                if role == "candidate" and scope == "acceptance" and policy and result.verdict == "pass" and result.complete:
                    from .implementation import check
                    # Formal review uses the same token admission budget as research.
                    current = self.store.task(task_id)
                    limit = current["manifest"].get("budget", {}).get("tokens")
                    review_path = root / "evaluations" / evaluation_id / "implementation.json"
                    cached = json.loads(review_path.read_text()) if review_path.exists() else {}
                    needs_agent = (policy.get("framework") or policy.get("self_authored")) and not cached.get("complete")
                    if limit is not None and needs_agent:
                        if current["tokens"] >= limit:
                            raise ValueError("Implementation review admission budget exhausted; extend the task budget")
                        with self.store.connect() as db:
                            unknown = db.execute("SELECT 1 FROM events WHERE task_id=? AND kind='implementation_review' AND json_extract(data,'$.usage.measurement') IS NOT NULL AND json_extract(data,'$.usage.total_tokens') IS NULL LIMIT 1", (task_id,)).fetchone()
                        if unknown:
                            raise ValueError("Prior implementation review has unknown token usage")
                    review = check(source, policy, contract_id=spec["contract_id"], source_id=candidate_id,
                        directory=root / "evaluations" / evaluation_id, agent_config=spec.get("agent", {}),
                        instructions=spec["instructions"])
                    with self.store.transaction() as db:
                        prior = db.execute("SELECT 1 FROM events WHERE task_id=? AND kind='implementation_review' AND json_extract(data,'$.review_id')=?",
                                           (task_id, review["review_id"])).fetchone()
                        if not prior:
                            db.execute("UPDATE tasks SET tokens=tokens+? WHERE id=?", (review["usage"].get("total_tokens") or 0, task_id))
                            self.store.emit(db, task_id, "implementation_review", {"evaluation_id": evaluation_id, **review})
                    if not review["complete"]:
                        # Keep the GPU receipt. Explicit recovery retries only review.
                        raise RuntimeError("Implementation review interrupted: " + "; ".join(review["errors"]))
                    result = replace(result, details={**result.details, "implementation": review})
                    if not review["passed"]:
                        result = replace(result, verdict="fail",
                            reason="; ".join(review["errors"]), failure_kind="implementation")
                self.bind(task_id)
                if identity(tree(source,include_ignored=role=='research',allow_empty=role=='research')) != candidate_id:
                    raise ValueError("Candidate snapshot changed during evaluation")
                with self.store.transaction() as db:
                    db.execute("UPDATE evaluations SET state='finished',receipt=?,assessment=? WHERE id=?",
                               (document(asdict(receipt)), document(asdict(result)), evaluation_id))
                    self.store.emit(db, task_id, "evaluation_finished", {"evaluation_id": evaluation_id,
                        "verdict": result.verdict, "complete": result.complete,
                        "eligible": result.verdict == "pass" and result.complete and goal.eligible(result)})
            except BaseException as error:
                with self.store.transaction() as db:
                    db.execute("UPDATE evaluations SET state='unknown' WHERE id=?", (evaluation_id,))
                    self.store.emit(db, task_id, "evaluation_interrupted", {"evaluation_id": evaluation_id, "error": type(error).__name__})
                raise
            return self.store.evaluation(evaluation_id)

    def submit_candidate(self, task_id, evaluation_id):
        task, adapter, goal = self.bind(task_id)
        record = self.store.evaluation(evaluation_id)
        if record["task_id"] != task_id or record["role"] != "candidate" or record["scope"] != "acceptance":
            raise ValueError("Submission requires this task's candidate acceptance evaluation")
        if record["state"] != "finished" or not record["assessment"]:
            raise ValueError("Submission requires a completed assessment")
        result = assessment(record["assessment"])
        if result.verdict != "pass" or not result.complete or not goal.eligible(result):
            raise ValueError("Candidate does not satisfy validation, coverage or objective constraints")
        if task["manifest"].get("config", {}).get("implementation"):
            review = result.details.get("implementation", {})
            if not (review.get("complete") and review.get("passed") and review.get("source_id") == record["candidate_id"]
                    and review.get("contract_id") == task["manifest"]["contract_id"]):
                raise ValueError("Candidate requires matching implementation acceptance evidence")
        root = self.directory(task_id)
        if identity(tree(root / "sources" / record["candidate_id"])) != record["candidate_id"]:
            raise ValueError("Candidate evidence source is corrupt")
        submission_id = "candidate-" + digest({"task": task_id, "evaluation": evaluation_id})[:24]
        with self.store.transaction() as db:
            cursor = db.execute("INSERT OR IGNORE INTO submissions VALUES(?,?,?,?,?,?)",
                (submission_id, task_id, record["candidate_id"], evaluation_id, record["revision"], time.time()))
            if cursor.rowcount:
                self.store.emit(db, task_id, "candidate_available", {"candidate_id": submission_id,
                    "source_id": record["candidate_id"], "evaluation_id": evaluation_id})
        return {"candidate_id": submission_id, "source_id": record["candidate_id"], "evaluation_id": evaluation_id}

    def results(self, task_id):
        task, adapter, goal = self.bind(task_id)
        with self.store.connect() as db:
            rows = [dict(row) for row in db.execute("SELECT * FROM submissions WHERE task_id=? ORDER BY created,id", (task_id,))]
            evaluation_rows = [dict(row) for row in db.execute(
                "SELECT id,role,candidate_id,revision,scope,state,job_id,assessment FROM evaluations WHERE task_id=? ORDER BY created,id", (task_id,))]
        for row in evaluation_rows:
            row["assessment"] = json.loads(row["assessment"]) if row["assessment"] else None
        values = []
        for row in rows:
            record = self.store.evaluation(row["evaluation_id"])
            values.append({**row, "assessment": record["assessment"]})
        frontier = [row for row in values if not any(
            other["id"] != row["id"] and goal.compare(assessment(other["assessment"]), assessment(row["assessment"])) == "better"
            for other in values)]
        return {"task_id": task_id, "contract_id": task["manifest"]["contract_id"],
                "objective": goal.describe(), "candidates": values, "frontier": [r["id"] for r in frontier],
                "evaluations": evaluation_rows}

    def feedback(self, task_id, candidate_id, value):
        if candidate_id not in {r["id"] for r in self.results(task_id)["candidates"]}:
            raise ValueError("Feedback must identify an accepted task candidate")
        with self.store.transaction() as db:
            self.store.emit(db, task_id, "integration_feedback", {"candidate_id": candidate_id, "feedback": value})

    def export(self, task_id, candidate_id, destination):
        task, adapter, goal = self.bind(task_id)
        result = next((r for r in self.results(task_id)["candidates"] if r["id"] == candidate_id), None)
        if result is None:
            raise ValueError("Unknown accepted candidate")
        target = Path(destination).resolve()
        if target.exists():
            raise ValueError("Export destination must not exist")
        source = self.directory(task_id) / "sources" / result["candidate_id"]
        files = tree(source)
        if identity(files) != result["candidate_id"]:
            raise ValueError("Candidate source identity mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".export-", dir=target.parent) as temporary:
            staging = Path(temporary)
            materialize(staging / "source", files)
            materialize(staging / "contract", tree(self.directory(task_id) / "contract"))
            atomic_json(staging / "task.json", task["manifest"])
            atomic_json(staging / "candidate.json", result)
            atomic_json(staging / "evaluation.json", self.store.evaluation(result["evaluation_id"]))
            atomic_json(staging / "delivery.json", task["manifest"]["delivery"])
            os.replace(staging, target)
        return {"path": str(target), "candidate_id": candidate_id}
