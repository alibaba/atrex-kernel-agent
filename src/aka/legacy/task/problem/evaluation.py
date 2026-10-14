"""Bridge explicit Problem content to the retained flat evaluator protocol."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
import tempfile

from aka.contracts.content import PublicInputContent, WorkspaceFile
from aka.contracts.problem import EvaluationInput, LoadedProblem
from .inputs import BenchInputs


@dataclass
class LegacyEvaluationAdapter(BenchInputs):
    problem: LoadedProblem | None = None

    def __post_init__(self) -> None:
        authored = self._authored_contract
        if authored is not None and authored.is_file():
            record = json.loads(authored.read_text(encoding="utf-8"))
            digest = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            if digest != record["sha256"]:
                raise ValueError("saved authored public contract changed")
            self.generated_digest = digest

    @property
    def evaluation(self) -> EvaluationInput:
        if self.problem is None or self.problem.evaluation is None:
            raise TypeError("legacy evaluation requires explicit Problem evaluation content")
        return self.problem.evaluation

    @property
    def private_reference_dir(self) -> Path | None:
        return self.evaluation.private_reference_dir

    def operator_files(self) -> tuple[WorkspaceFile, ...]:
        return self.problem.agent.files

    @property
    def _authored_contract(self) -> Path | None:
        root = self.problem.snapshot_directory
        return root / "authored-contract.json" if root is not None else None

    def prepare_public(self) -> PublicInputContent:
        authored = self._authored_contract
        if authored is not None and authored.is_file():
            record = json.loads(authored.read_text(encoding="utf-8"))
            text = record["text"]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest != record["sha256"]:
                raise ValueError("saved authored public contract changed")
            self.generated_digest = digest
            destination = self.workspace / "agent_problem.json"
            if destination.is_file() and destination.read_bytes() != text.encode("utf-8"):
                raise ValueError("workspace agent_problem.json differs from its saved authored contract")
            return PublicInputContent(True, (WorkspaceFile("agent_problem.json", text=text),))
        public = super().prepare_public()
        if public.ready and self.generated_digest:
            self.record_generated()
        return public

    def record_generated(self) -> None:
        super().record_generated()
        authored = self._authored_contract
        if authored is None:
            return
        text = (self.workspace / "agent_problem.json").read_text(encoding="utf-8")
        payload = json.dumps(dict(text=text, sha256=self.generated_digest), ensure_ascii=False) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=authored.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(payload)
        try:
            os.replace(temporary, authored)
        finally:
            temporary.unlink(missing_ok=True)
