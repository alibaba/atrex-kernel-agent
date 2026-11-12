import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from aka.light.evaluation import agate_native, response_policy, ssh_devices
from aka.light.evaluation import ssh_native, ssh_worker
from aka.light.experiment import records
from aka.legacy.agent import dsh


def _jsonl(*events):
    return "\n".join(json.dumps(event) for event in events) + "\n"


def test_dsh_runtime_connection_is_private_and_ephemeral(tmp_path):
    environment = {
        "DSH_HOME": str(tmp_path / "dsh"),
        "ATREX_AKA_RUNTIME_URL": "http://runtime.invalid",
        "ATREX_AKA_RUNTIME_TOKEN": "secret-token",
    }

    prepared = dsh.prepare_runtime_connection(environment)
    connection = Path(prepared[dsh.CONNECTION_ENV])
    assert connection.is_file()
    assert connection.stat().st_mode & 0o777 == 0o600
    assert json.loads(connection.read_text()) == {
        "url": "http://runtime.invalid",
        "token": "secret-token",
    }
    assert dsh.CONNECTION_ENV not in environment

    dsh.release_runtime_connection(prepared)
    assert not connection.exists()


def test_dsh_stream_reports_session_failure_and_exact_usage():
    stdout = _jsonl(
        {"type": "session", "sessionId": "session_1"},
        {"type": "status", "phase": "step_start", "turn": 0, "step": 0},
        {
            "type": "status",
            "phase": "step_end",
            "turn": 0,
            "step": 0,
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "cacheReadTokens": 2,
                "cacheWriteTokens": 1,
                "totalTokens": 17,
            },
        },
        {
            "type": "final",
            "sessionId": "session_1",
            "reason": {"kind": "completed"},
        },
    )

    assert dsh.session_id_from_stream(stdout) == "session_1"
    assert dsh.terminal_failure(stdout) == (None, None)
    events, usage = dsh.normalize_stream(stdout)
    assert [event.kind for event in events] == ["usage_delta", "terminal_usage"]
    assert usage.total_tokens == 17
    assert usage.measurement == "exact"

    limited = _jsonl(
        {"type": "session", "sessionId": "session_1"},
        {
            "type": "final",
            "sessionId": "session_1",
            "reason": {"kind": "max-tokens"},
        },
    )
    assert dsh.terminal_failure(limited) == ("output_limit", "max_output_tokens")


def test_dsh_settings_reject_paths_and_unknown_authority():
    assert dsh.settings_values('{"provider":"anthropic","profile":"production"}') == {
        "provider": "anthropic",
        "profile": "production",
    }
    with pytest.raises(ValueError, match="profile name"):
        dsh.settings_values('{"profile":"../private/profile.json"}')
    with pytest.raises(ValueError, match="accept only"):
        dsh.settings_values('{"token":"secret"}')


def _abba_state(phase, endpoint, payload, *, succeeded=False):
    state = {
        "phase": phase,
        "identity": {
            "kind": "eval",
            "url_fingerprint": endpoint,
            "payload": payload,
        },
    }
    if phase == "terminal":
        job = {
            "status": "succeeded" if succeeded else "failed",
            "result": {"complete": True},
        }
        state["process"] = {
            "returncode": 0 if succeeded else 1,
            "stdout": json.dumps(job),
            "stderr": "",
        }
    return state


def test_abba_replacement_evidence_requires_exact_payload_and_new_gateway(monkeypatch):
    monkeypatch.setattr(
        records,
        "cacheable_job",
        lambda job, identity: job["status"] == "succeeded",
    )
    payload = {"abba": {"repeats": 2}, "candidate": "source", "shape_ids": ["s0"]}
    accepted = _abba_state("accepted", "old-gateway", payload)
    replacement = _abba_state("terminal", "new-gateway", payload, succeeded=True)

    assert records._abba_acceptance_states([accepted, replacement]) == [replacement]

    changed = _abba_state(
        "terminal",
        "new-gateway",
        {**payload, "shape_ids": ["s1"]},
        succeeded=True,
    )
    assert records._abba_acceptance_states([accepted, changed]) == [accepted, changed]

    same_gateway = _abba_state("terminal", "old-gateway", payload, succeeded=True)
    assert records._abba_acceptance_states([accepted, same_gateway]) == [accepted, same_gateway]


def test_agate_native_placement_is_forwarded_to_diagnostic(monkeypatch):
    option = agate_native.NativeOption(
        flag="--cluster-label",
        dest="cluster_label",
        commands=frozenset({"eval"}),
        action="append",
        choices=None,
        value_type=str,
        metavar="KEY=VALUE",
        help=None,
        schema_type="string",
    )
    monkeypatch.setattr(agate_native, "catalog", lambda: {option.flag: option})
    args = Namespace(agate_native__cluster_label=["pool=L20D", "zone=cn-a"])
    request = {"__atrex_source_diagnostic": {}}

    result = agate_native.apply_request(request, args, "run")

    expected = {"cluster_selector": {"pool": "L20D", "zone": "cn-a"}}
    assert result["placement"] == expected
    assert result["__atrex_source_diagnostic"]["placement"] == expected


def test_unknown_gateway_outcome_is_not_misclassified_as_known():
    assert response_policy.known_gateway_error(
        {"error": {"error_class": "timeout", "message": "deadline"}}
    )
    assert not response_policy.known_gateway_error(
        {"error": {"error_class": "new_server_failure", "message": "opaque"}}
    )


def test_ssh_worker_materializes_source_tree_and_rejects_traversal(tmp_path):
    request = {
        "__atrex_eval_archives": {
            "candidate.tar.gz": {
                "kernel.py": "from impl.kernel import Model\n",
                "impl/kernel.py": "class Model: pass\n",
            }
        }
    }
    entry = ssh_worker.materialize(
        {"archive": "candidate.tar.gz", "entry_point": "kernel.py"},
        request,
        tmp_path,
    )
    assert entry == tmp_path / "kernel.py"
    assert (tmp_path / "impl/kernel.py").read_text() == "class Model: pass\n"

    with pytest.raises(ValueError, match="relative"):
        ssh_worker.write(tmp_path, "../escape.py", "bad")


def test_ssh_native_manifest_fails_closed_before_compilation(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "ssh_worker", ssh_worker)
    manifest = {
        "targets": {"hardware": ["L20D"]},
        "source": {"archive": "native.tar.gz", "files": {}},
    }
    request = {
        "spec": {"target_hardware": ["L20N"]},
        "reference": {"shapes": {}},
        "__atrex_eval_archives": {"native.tar.gz": {}},
    }
    with pytest.raises(ValueError, match="targets do not cover"):
        ssh_native.materialize(manifest, request, tmp_path)


def test_ssh_device_map_uses_explicit_character_devices():
    assignment = ssh_devices.resolve(
        0,
        device_map={
            "0": {
                "vendor": "nvidia",
                "uuid": "GPU-test",
                "devices": ["/dev/null"],
                "environment": {"CUDA_VISIBLE_DEVICES": "GPU-test"},
            }
        },
    )
    assert assignment["index"] == 0
    assert assignment["devices"] == ["/dev/null"]
    with pytest.raises(ValueError, match="device or runtime"):
        ssh_devices.resolve(0, isolation="process", device_map={})
