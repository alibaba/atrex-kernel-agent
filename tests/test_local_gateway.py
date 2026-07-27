from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from tools.local_gateway import JobStore, LocalGateway, LocalScheduler, create_server


class LocalGatewayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="local-gateway-test-")
        self.state_dir = Path(self.temp.name)
        self.gateway = LocalGateway(self.state_dir, workers=1, max_output_bytes=1024 * 1024)
        self.server = create_server(self.gateway, "127.0.0.1", 0)
        self.gateway.start()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()
        self.gateway.close()
        self.temp.cleanup()

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, object]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        request = urllib.request.Request(
            self.base_url + path,
            method=method,
            data=data,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                raw = response.read().decode("utf-8")
                return response.status, json.loads(raw) if raw and raw != "ok" else raw
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    @staticmethod
    def dev_request(command: str, **extra: object) -> dict:
        return {
            "spec": {"target_hardware": ["local"]},
            "command": command,
            "timeout_s": 5,
            "env_vars": {},
            "files": {},
            **extra,
        }

    def submit(self, command: str, **extra: object) -> str:
        status, body = self.request("POST", "/v1/jobs/dev", self.dev_request(command, **extra))
        self.assertEqual(status, 202)
        self.assertIsInstance(body, dict)
        return body["job_id"]

    def wait_for_status(self, job_id: str, wanted: set[str], timeout: float = 5) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, body = self.request("GET", f"/v1/jobs/{job_id}")
            self.assertEqual(status, 200)
            if body["status"] in wanted:
                return body
            time.sleep(0.02)
        self.fail(f"job {job_id} did not reach {wanted}")

    def test_health_env_and_dev_result_match_agate_shape(self) -> None:
        status, body = self.request("GET", "/healthz")
        self.assertEqual((status, body), (200, "ok"))
        status, body = self.request("GET", "/v1/env")
        self.assertEqual(status, 200)
        self.assertEqual(body["env"][0]["gpu"], "local")

        job_id = self.submit(
            "printf '%s:%s' \"$(cat nested/input.txt)\" \"$VALUE\"",
            files={"nested/input.txt": "payload"},
            env_vars={"VALUE": "environment"},
        )
        job = self.wait_for_status(job_id, {"succeeded"})
        self.assertEqual(job["kind"], "dev")
        self.assertEqual(job["result"]["exit_code"], 0)
        self.assertEqual(job["result"]["stdout"], "payload:environment")
        self.assertEqual(job["result"]["stderr"], "")

    def test_fifo_queue_serializes_commands(self) -> None:
        first = self.submit("sleep 0.5; printf first")
        self.wait_for_status(first, {"running"})
        second = self.submit("printf second")

        _, second_queued = self.request("GET", f"/v1/jobs/{second}")
        self.assertEqual(second_queued["status"], "queued")

        status, first_done = self.request("GET", f"/v1/jobs/{first}?wait=true&timeout=2")
        self.assertEqual(status, 200)
        self.assertEqual(first_done["status"], "succeeded")
        second_done = self.wait_for_status(second, {"succeeded"})
        self.assertEqual(second_done["result"]["stdout"], "second")

    def test_cancelled_queued_job_is_never_started(self) -> None:
        first = self.submit("sleep 0.5")
        self.wait_for_status(first, {"running"})
        second = self.submit("printf should-not-run")
        status, cancelled = self.request("POST", f"/v1/jobs/{second}/cancel")
        self.assertEqual(status, 200)
        self.assertEqual(cancelled["status"], "cancelled")

        self.wait_for_status(first, {"succeeded"})
        time.sleep(0.1)
        self.assertFalse((self.state_dir / "jobs" / second).exists())

    def test_cancel_running_job(self) -> None:
        job_id = self.submit("sleep 30")
        self.wait_for_status(job_id, {"running"})
        status, cancelled = self.request("POST", f"/v1/jobs/{job_id}/cancel")
        self.assertEqual(status, 200)
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(self.gateway.store.get(job_id)["status"], "cancelled")

    def test_state_directory_has_single_scheduler_owner(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            LocalGateway(self.state_dir, workers=1)

    def test_configured_gpu_alias_is_accepted(self) -> None:
        alias_dir = self.state_dir / "alias"
        gateway = LocalGateway(alias_dir, workers=1, gpu_aliases=("TEST_GPU",))
        gateway.start()
        try:
            payload = self.dev_request("printf alias")
            payload["spec"]["target_hardware"] = ["TEST_GPU"]
            accepted, _ = gateway.submit_dev(payload, "req-alias")
            completed = gateway.scheduler.wait_for_job(accepted["job_id"], timeout=3)
            self.assertEqual(completed["status"], "succeeded")
            self.assertIn("TEST_GPU", gateway.environment()["aliases"])
        finally:
            gateway.close()

    def test_rejects_unsafe_file_path_and_unsupported_kind(self) -> None:
        payload = self.dev_request("true", files={"../escape": "bad"})
        status, error = self.request("POST", "/v1/jobs/dev", payload)
        self.assertEqual(status, 422)
        self.assertEqual(error["reason"], "validation_error")

        status, error = self.request("POST", "/v1/jobs/eval", {})
        self.assertEqual(status, 501)
        self.assertEqual(error["reason"], "kind_not_supported")

    def test_restart_fails_running_job_and_preserves_queued_job(self) -> None:
        restart_dir = self.state_dir / "restart"
        store = JobStore(restart_dir / "jobs.db")
        first, _ = store.create("dev", self.dev_request("printf first"), "req-first")
        second, _ = store.create("dev", self.dev_request("printf second"), "req-second")
        claimed = store.claim_next()
        self.assertEqual(claimed[0], first["job_id"])
        store.close()

        recovered = JobStore(restart_dir / "jobs.db")
        self.assertEqual(recovered.get(first["job_id"])["status"], "failed")
        self.assertEqual(recovered.get(second["job_id"])["status"], "queued")
        scheduler = LocalScheduler(recovered, restart_dir / "jobs", workers=1, max_output_bytes=1024)
        scheduler.start()
        try:
            completed = scheduler.wait_for_job(second["job_id"], timeout=3)
            self.assertEqual(completed["status"], "succeeded")
            self.assertEqual(completed["result"]["stdout"], "second")
        finally:
            scheduler.stop()
            recovered.close()


if __name__ == "__main__":
    unittest.main()
