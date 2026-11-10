"""Read the shared Experiment records without creating a namespace or executing work."""
import hashlib
import json
import re
from pathlib import Path
from aka.contracts.experiment import ExperimentRef
from aka.experiment.file_records import FileExperimentRecords
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.task.source.formats import from_inputs
from aka.workflow.light.measurements import RecordedMeasurements
from aka.workflow.light.support.paths import supervisor_campaign_root
from .legacy_records import LegacyMeasurementReader, source_identity as legacy_identity

KERNEL_ID = re.compile(r"kernel-(?:[0-9a-f]{32}|[0-9a-f]{64})")
RECORD_ID = re.compile(r"gateway-(?:[0-9a-f]{32}|[0-9a-f]{64})")


def source_identity(snapshot):
    digest = hashlib.sha256(snapshot).hexdigest()
    return {"kernel_id": "kernel-" + digest, "kernel_artifact_digest": "sha256:" + digest, "sha256": digest}


class MeasurementStore:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.legacy = None
        self._legacy = LegacyMeasurementReader(supervisor_campaign_root(self.workspace) / "measurements")
        if not self.shared_available(self.workspace):
            self.legacy = LegacyMeasurementReader(supervisor_campaign_root(self.workspace) / "measurements")
            self.root = self.legacy.root
            return
        self._records = FileExperimentRecords(self.workspace.parent / ".aka-experiments", str(self.workspace), create=False)
        self.root = self._records.root
        self._view = RecordedMeasurements(workspace_provider=GitCandidateWorkspace())

    @staticmethod
    def shared_available(workspace):
        workspace = Path(workspace).resolve()
        key = hashlib.sha256(json.dumps(str(workspace), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        return (workspace.parent / ".aka-experiments" / key).is_dir()

    @staticmethod
    def available(workspace):
        return MeasurementStore.shared_available(workspace) or (supervisor_campaign_root(Path(workspace)) / "measurements/records").is_dir()

    def source_identity(self, snapshot, *, kernel_id=None):
        return legacy_identity(snapshot) if self.legacy is not None or kernel_id is not None and len(kernel_id) == 39 else source_identity(snapshot)

    def record_directory(self, record_id):
        if self.legacy is not None or isinstance(record_id, str) and len(record_id) == 40:
            return self._legacy.record_directory(record_id)
        if not RECORD_ID.fullmatch(record_id):
            raise ValueError("invalid record identity")
        return self.root / record_id.removeprefix("gateway-")

    def record_ids(self):
        if self.legacy is not None:
            return self.legacy.record_ids()
        return tuple("gateway-" + record.ref.experiment_id for record in self._records.iter_records()) + self._legacy.record_ids()

    def read(self, record_id):
        if self.legacy is not None or isinstance(record_id, str) and len(record_id) == 40:
            return self._legacy.read(record_id)
        if not RECORD_ID.fullmatch(record_id):
            raise ValueError("invalid record identity")
        try:
            record = self._records.get(ExperimentRef(str(self.workspace), record_id.removeprefix("gateway-")))
            evidence = self._view.project(self._records, record)
        except (LookupError, RuntimeError) as error:
            raise ValueError("measurement evidence unavailable") from error
        operation = "evaluate" if evidence["gateway_kind"] == "run" else evidence["gateway_kind"]
        result = evidence["result"]
        command = record.request.get("command", [])
        request = {"operation": operation, "options": {"command": list(command), "evaluation_mode": result.get("mode"),
            "evaluation_policy": {"mode": result.get("mode")}}}
        request["standard_full_evaluation"] = evidence["standard_full_evaluation"]
        prefix = {"evaluate": "[test_kernel] RESULT_JSON=", "same_allocation_abba": "[sandbox] ABBA_JSON=",
                  "profile": "[sandbox] PROFILE_JSON=", "check": "[sandbox] CHECK_JSON=",
                  "disassemble": "[sandbox] DISASSEMBLE_JSON="}.get(operation, "")
        identity = {key: evidence[key] for key in ("kernel_id", "kernel_artifact_digest", "sha256") if key in evidence}
        response = {"exit_code": evidence["exit_code"], "stdout": prefix + json.dumps(result, ensure_ascii=False) + "\n", "stderr": ""}
        from .legacy_records import digest
        return {"operation": operation, "gateway_record_id": record_id, "request": request,
            "task_digest": digest(request), "result_digest": digest(response),
            "created_at": record.created_at, "cacheable": evidence["execution_status"] == "completed" and result.get("all_pass") is True,
            "kernels": {"candidate": identity}, "response": response,
            "execution_status": evidence["execution_status"], "experiment_ref": record.ref.as_dict()}

    def read_kernel(self, kernel_id):
        if self.legacy is not None or isinstance(kernel_id, str) and len(kernel_id) == 39:
            return self._legacy.read_kernel(kernel_id)
        if not KERNEL_ID.fullmatch(kernel_id):
            raise ValueError("invalid kernel identity")
        for record in self._records.iter_records():
            source = self._view.candidate_source(self._records, record)
            if source is not None and self._view.kernel(source)["kernel_id"] == kernel_id:
                return from_inputs({name: value for name, value in source.files.items() if value is not None})
        raise FileNotFoundError("kernel is unavailable in this Campaign")
