#!/usr/bin/env python3
"""Deterministic contracts for the lightweight natural-language front door."""
from __future__ import annotations

import json
import os
import stat
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agent_launch  # noqa: E402
import query_nl  # noqa: E402

STORE = query_nl.OWN_STORE_ROOT
STORE_OK = (STORE / "kernel_wiki" / "records" / "index.json").is_file()


def stub_intent(**overrides):
    doc = {
        "architecture": "sm_100", "vendor": "nvidia", "dsl": "triton",
        "operator_terms": ["rmsnorm"], "measured_symptoms": [],
        "free_text_terms": ["fusion"], "intents": ["technique"],
        "hardware_requests": [],
    }
    doc.update(overrides)
    return doc


def run_front_door(*argv):
    out, err = StringIO(), StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = query_nl.main(list(argv))
    except SystemExit as exc:
        code = exc.code
    return code, out.getvalue(), err.getvalue()


class fake_bridge:
    def __init__(self, intent, *, write=True, timed_out=False, sequence=None):
        self.intent, self.write, self.timed_out = intent, write, timed_out
        self.sequence = list(sequence) if sequence is not None else None
        self.prompts = []
        self.clis = []

    def __enter__(self):
        self.real = agent_launch.run
        self.real_json = agent_launch.run_claude_json

        def fake(cli, prompt, cwd, timeout=None, env=None):
            self.clis.append(cli)
            self.prompts.append(prompt)
            if self.write:
                body = self.intent if isinstance(self.intent, str) else json.dumps(self.intent)
                (Path(cwd) / query_nl.INTENT_FILE).write_text(body, encoding="utf-8")
            return "", "", 0, self.timed_out

        agent_launch.run = fake

        def fake_json(prompt, cwd, timeout=None, env=None):
            del cwd, timeout, env
            self.clis.append("claude")
            self.prompts.append(prompt)
            if not self.write:
                return "", "", 0, self.timed_out
            call_index = len(self.clis) - 1
            payload = (self.sequence[min(call_index, len(self.sequence) - 1)]
                       if self.sequence else self.intent)
            result = payload if isinstance(payload, str) else json.dumps(payload)
            envelope = {
                "type": "result",
                "result": result,
                "duration_api_ms": 12,
                "ttft_ms": 4,
                "num_turns": 1,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
            return json.dumps(envelope), "", 0, self.timed_out

        agent_launch.run_claude_json = fake_json
        return self

    def __exit__(self, *exc):
        agent_launch.run = self.real
        agent_launch.run_claude_json = self.real_json
        return False


class IntentValidationTests(unittest.TestCase):
    def test_rejects_non_object(self):
        with self.assertRaises(SystemExit):
            query_nl.validate_intent([])

    def test_rejects_executable_query_fields(self):
        for field in ("queries", "flags", "tool", "reading_guide"):
            with self.subTest(field=field), self.assertRaises(SystemExit):
                query_nl.validate_intent(dict(stub_intent(), **{field: []}))

    def test_rejects_bad_hardware_request(self):
        with self.assertRaises(SystemExit):
            query_nl.validate_intent(stub_intent(
                hardware_requests=[{"kind": "search", "value": "b200"}]))

    def test_normalizes_and_caps_semantic_lists(self):
        intent = query_nl.validate_intent(stub_intent(
            free_text_terms=[str(i) for i in range(20)], intents=["technique", "bogus"]))
        self.assertEqual(len(intent["free_text_terms"]), query_nl.MAX_TERMS)
        self.assertEqual(intent["intents"], ["technique"])

    def test_duplicate_product_fact_requests_collapse_to_full_product(self):
        intent = query_nl.validate_intent(stub_intent(hardware_requests=[
            {"kind": "product", "value": "B200", "field": "registers", "vs": None},
            {"kind": "product", "value": "b200", "field": "shared_memory", "vs": None},
            {"kind": "product", "value": "B200", "field": "registers", "vs": None},
        ]))
        self.assertEqual(intent["hardware_requests"], [{
            "kind": "product", "value": "B200", "field": None, "vs": None,
        }])

    def test_plain_json_contract_rejects_missing_extra_and_wrong_types(self):
        invalid = [
            {k: v for k, v in stub_intent().items() if k != "architecture"},
            dict(stub_intent(), queries=[]),
            stub_intent(operator_terms="rmsnorm"),
            stub_intent(intents=["unknown"]),
            stub_intent(hardware_requests=[{
                "kind": "product", "value": "B200", "field": None,
            }]),
        ]
        for doc in invalid:
            with self.subTest(doc=doc), self.assertRaises(ValueError):
                query_nl.strictly_validate_intent(doc)

    def test_plain_json_contract_accepts_complete_typed_intent(self):
        doc = stub_intent()
        self.assertIs(query_nl.strictly_validate_intent(doc), doc)


@unittest.skipUnless(STORE_OK, "store not built")
class PlannerTests(unittest.TestCase):
    def test_runtime_architecture_is_resolved_by_script(self):
        normalized, _ = query_nl.normalize_intent(
            query_nl.validate_intent(stub_intent()), STORE)
        self.assertEqual(normalized["arch"], "blackwell")

    def test_product_spelling_and_hardware_lookup_share_one_architecture_map(self):
        cases = {
            "NVIDIA B200 GPU": "blackwell",
            "NVIDIA B300 GPU": "blackwell-ultra",
            "NVIDIA RTX PRO 5000 GPU": "blackwell-geforce",
            "AMD Instinct MI308X GPU": "cdna3",
            "AMD MI355X GPU": "cdna4",
        }
        allowed = {row["arch"] for row in query_nl.hardware_identity.HARDWARE_IDENTITIES.values()}
        for external, arch in cases.items():
            with self.subTest(external=external):
                self.assertEqual(query_nl._resolve_arch(external, allowed), arch)

    def test_unknown_scope_terms_degrade_to_text(self):
        normalized, notes = query_nl.normalize_intent(
            query_nl.validate_intent(stub_intent(dsl="ImaginaryDSL")), STORE)
        self.assertIsNone(normalized["dsl"])
        self.assertIn("ImaginaryDSL", normalized["terms"])
        self.assertTrue(any("free text" in n for n in notes))

    def test_natural_language_symptom_spelling_is_normalized(self):
        normalized, notes = query_nl.normalize_intent(
            query_nl.validate_intent(stub_intent(
                measured_symptoms=["register pressure", "tail effect"])), STORE)
        self.assertEqual(normalized["symptom"], "register-pressure")
        self.assertIn("tail effect", normalized["terms"])
        self.assertFalse(any("did not match" in note for note in notes))

    def test_all_recorded_product_spellings_use_internal_addresses(self):
        examples = {
            "NVIDIA B200 GPU": "b200",
            "NVIDIA B300 GPU": "b300",
            "AMD Instinct MI300X GPU": "mi300x",
            "AMD MI308X accelerator": "mi308x",
            "AMD MI355X GPU": "mi355x",
            "SM_120": "sm120",
        }
        for external, internal in examples.items():
            with self.subTest(external=external):
                normalized, _ = query_nl.normalize_intent(
                    query_nl.validate_intent(stub_intent(hardware_requests=[{
                        "kind": "product", "value": external,
                        "field": None, "vs": None,
                    }])), STORE)
                self.assertEqual(normalized["hardware_requests"][0]["value"],
                                 internal)

    def test_runtime_architecture_misclassified_as_feature_is_ignored(self):
        normalized, notes = query_nl.normalize_intent(
            query_nl.validate_intent(stub_intent(hardware_requests=[
                {"kind": "product", "value": "B200", "field": None, "vs": None},
                {"kind": "feature", "value": "sm_100", "field": None, "vs": None},
            ])), STORE)
        self.assertEqual(normalized["hardware_requests"], [{
            "kind": "product", "value": "b200", "field": None, "vs": None,
        }])
        self.assertTrue(any("classified as a feature" in note for note in notes))

    def test_product_misclassified_as_feature_is_recovered(self):
        normalized, notes = query_nl.normalize_intent(
            query_nl.validate_intent(stub_intent(hardware_requests=[{
                "kind": "feature", "value": "NVIDIA B-200 GPU",
                "field": None, "vs": None,
            }])), STORE)
        self.assertEqual(normalized["hardware_requests"], [{
            "kind": "product", "value": "b200", "field": None, "vs": None,
        }])
        self.assertTrue(any("treated as product b200" in note for note in notes))

    def test_legacy_internal_kernel_projection_strips_bias_fields(self):
        projected = query_nl._project_kernel_record("rank-1", {
            "applies_to": {"arch": "blackwell"},
            "match": {"arch": "exact"},
            "signals": {"bottleneck": "memory"},
            "records": {
                "id": "private.stable.id",
                "type": "strategy",
                "payload": {"summary": "isolated"},
                "evidence": {"resolution": "must not be served"},
            },
        })
        self.assertEqual(projected, ("private.stable.id", {
            "source": "kernel_wiki",
            "type": "strategy",
            "applies_to": {"arch": "blackwell"},
            "match": {"arch": "exact"},
            "payload": {"summary": "isolated"},
        }))

    def test_widening_never_drops_arch_and_never_falls_back(self):
        normalized, _ = query_nl.normalize_intent(
            query_nl.validate_intent(stub_intent()), STORE)
        for options in ({}, {"drop_operator": True}, {"drop_operator": True,
                         "drop_symptom": True, "any_terms": True},
                        {"drop_operator": True, "drop_symptom": True,
                         "any_terms": True, "cross_arch": True}):
            flags = query_nl._kernel_flags(normalized, **options)
            self.assertIn("--arch", flags)
            self.assertIn("blackwell", flags)
            self.assertIn("--no-fallback", flags)
            self.assertNotIn("--fallback-ratio", flags)


class StoreResolutionTests(unittest.TestCase):
    def test_explicit_root_wins(self):
        self.assertEqual(query_nl.resolve_store_root(str(STORE)), STORE)

    def test_bad_root_is_refused(self):
        with TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            query_nl.resolve_store_root(tmp)

    def test_env_root_is_honoured(self):
        old = os.environ.get(query_nl.STORE_ENV)
        os.environ[query_nl.STORE_ENV] = str(STORE)
        try:
            self.assertEqual(query_nl.resolve_store_root(None), STORE)
        finally:
            if old is None:
                os.environ.pop(query_nl.STORE_ENV, None)
            else:
                os.environ[query_nl.STORE_ENV] = old

    def test_default_never_implicitly_switches_to_sibling_store(self):
        old = os.environ.pop(query_nl.STORE_ENV, None)
        try:
            self.assertEqual(query_nl.resolve_store_root(None), STORE)
        finally:
            if old is not None:
                os.environ[query_nl.STORE_ENV] = old

    def test_placeholder_internal_directory_remains_an_empty_module_slot(self):
        old_internal = query_nl.SIBLING_INTERNAL
        with TemporaryDirectory() as tmp:
            placeholder = Path(tmp) / "internal_gpu_wiki"
            placeholder.mkdir()
            (placeholder / "SOURCE.txt").write_text("source only", encoding="utf-8")
            query_nl.SIBLING_INTERNAL = placeholder
            try:
                self.assertEqual(query_nl.resolve_store_roots(None), [
                    (query_nl.PUBLIC_STORE, STORE),
                    (query_nl.INTERNAL_STORE, placeholder.resolve()),
                ])
            finally:
                query_nl.SIBLING_INTERNAL = old_internal

    def test_excluded_ids_remain_store_isolated(self):
        exclude = "public.id,internal_gpu_wiki::private.id"
        self.assertEqual(
            query_nl._exclude_for_store(exclude, query_nl.PUBLIC_STORE),
            "public.id",
        )
        self.assertEqual(
            query_nl._exclude_for_store(exclude, query_nl.INTERNAL_STORE),
            "private.id",
        )


@unittest.skipUnless(STORE_OK, "store not built")
class FrontDoorTests(unittest.TestCase):
    REQUEST = "fused rmsnorm in triton on sm_100"

    def test_dry_run_is_store_blind_and_spawns_nothing(self):
        with fake_bridge(None, write=False) as bridge:
            code, out, _ = run_front_door(self.REQUEST, "--dry-run",
                                          "--store-root", str(STORE),
                                          "--agent-cli", "qodercli")
        self.assertEqual(code, 0)
        self.assertEqual(bridge.prompts, [])
        self.assertIn("query_bridge_agent", out)
        self.assertIn("query_intent.json", out)
        self.assertNotIn("query_wiki.py      (experience", out)

    def test_prompt_names_new_skill_and_forbids_queries(self):
        with fake_bridge(None, write=False):
            _, out, _ = run_front_door(self.REQUEST, "--dry-run",
                                       "--store-root", str(STORE),
                                       "--agent-cli", "qodercli")
        self.assertIn("query_bridge_agent/SKILL.md", out)
        self.assertIn("MUST NOT run query_wiki.py", out)

    def test_missing_or_invalid_intent_fails_loudly(self):
        with fake_bridge(None, write=False):
            code, _, err = run_front_door(
                self.REQUEST, "--store-root", str(STORE),
                "--agent-cli", "qodercli")
        self.assertEqual(code, 4)
        self.assertIn("no-intent", err)
        with fake_bridge("not json"):
            code, _, err = run_front_door(
                self.REQUEST, "--store-root", str(STORE),
                "--agent-cli", "qodercli")
        self.assertEqual(code, 2)
        self.assertIn("bad-intent", err)

    def _answer(self, intent, *extra):
        with fake_bridge(intent):
            code, out, err = run_front_door(self.REQUEST, "--store-root", str(STORE),
                                            *extra)
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def test_output_has_only_records_and_notes(self):
        answer = self._answer(stub_intent())
        self.assertEqual(set(answer), {"records", "notes"})

    def test_records_are_id_keyed_lightweight_and_payload_is_exact(self):
        answer = self._answer(stub_intent(free_text_terms=[]), "--max-records", "3")
        self.assertTrue(answer["records"])
        index = json.loads((STORE / "kernel_wiki/records/index.json").read_text())
        paths = {e["id"]: e["path"] for e in index["records"]}
        for rid, entry in answer["records"].items():
            self.assertEqual(set(entry), {
                "store", "source", "type", "applies_to", "match", "payload",
            })
            self.assertIn(entry["store"], {
                query_nl.PUBLIC_STORE, query_nl.INTERNAL_STORE,
            })
            self.assertNotIn("evidence", json.dumps(entry))
            if (entry["source"] == "kernel_wiki"
                    and entry["store"] == query_nl.PUBLIC_STORE):
                stored = json.loads((STORE / "kernel_wiki" / paths[rid]).read_text())
                self.assertEqual(entry["payload"], stored["payload"])

    def test_hardware_hit_shares_mapping_but_keeps_its_payload_isolated(self):
        intent = stub_intent(hardware_requests=[{
            "kind": "product", "value": "b200",
            "field": "peak_compute.bf16.dense", "vs": None,
        }])
        answer = self._answer(intent, "--max-records", "10")
        hit = answer["records"]["nvidia.blackwell.spec-sheet.b200"]
        self.assertEqual(hit["source"], "hardware_wiki")
        self.assertEqual(hit["type"], "spec-sheet")
        self.assertEqual(hit["payload"]["value"], 2250)

    def test_prefixed_product_name_still_returns_hardware_wiki(self):
        intent = stub_intent(hardware_requests=[{
            "kind": "product", "value": "NVIDIA B200 GPU",
            "field": None, "vs": None,
        }])
        answer = self._answer(intent, "--max-records", "10")
        self.assertEqual(
            answer["records"]["nvidia.blackwell.spec-sheet.b200"]["source"],
            "hardware_wiki",
        )

    def test_installed_internal_store_is_queried_and_id_namespaced(self):
        old_internal = query_nl.SIBLING_INTERNAL
        with TemporaryDirectory() as tmp:
            internal = Path(tmp) / "internal_gpu_wiki"
            internal.mkdir()
            for name in ("tools", "kernel_wiki", "hardware_wiki"):
                os.symlink(STORE / name, internal / name, target_is_directory=True)
            query_nl.SIBLING_INTERNAL = internal
            try:
                intent = stub_intent(hardware_requests=[{
                    "kind": "product", "value": "b200", "field": None, "vs": None,
                }])
                with fake_bridge(intent):
                    code, out, err = run_front_door(
                        self.REQUEST, "--max-records", "6")
                self.assertEqual(code, 0, err)
                records = json.loads(out)["records"]
            finally:
                query_nl.SIBLING_INTERNAL = old_internal
        self.assertTrue(any(
            row["store"] == query_nl.PUBLIC_STORE for row in records.values()))
        self.assertTrue(any(
            row["store"] == query_nl.INTERNAL_STORE for row in records.values()))
        self.assertTrue(any(
            rid.startswith(query_nl.INTERNAL_STORE + "::") for rid in records))
        self.assertLessEqual(len(records), 6)

    def test_unknown_product_field_falls_back_to_complete_product_record(self):
        intent = stub_intent(hardware_requests=[{
            "kind": "product", "value": "b200",
            "field": "model_invented_field", "vs": None,
        }])
        answer = self._answer(intent, "--max-records", "10")
        hit = answer["records"]["nvidia.blackwell.spec-sheet.b200"]
        self.assertEqual(hit["source"], "hardware_wiki")
        self.assertIn("facts", hit["payload"])
        self.assertTrue(any("served the complete b200 product record" in note
                            for note in answer["notes"]))

    def test_duplicate_kernel_records_are_impossible_by_construction(self):
        answer = self._answer(stub_intent(free_text_terms=["zzznosuchterm"]))
        self.assertEqual(len(answer["records"]), len(set(answer["records"])))


class LauncherTests(unittest.TestCase):
    def stub(self, tmp: Path, body: str, name="codex"):
        path = tmp / name
        path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return dict(os.environ, PATH="%s:%s" % (tmp, os.environ.get("PATH", "")))

    def test_prompt_is_last_cli_argument(self):
        for cli in agent_launch.SUPPORTED:
            self.assertEqual(agent_launch.build_command(cli, "PROMPT")[-1], "PROMPT")

    def test_claude_settings_are_forwarded_before_prompt(self):
        argv = agent_launch.build_command(
            "claude", "PROMPT", settings='{"model":"qwen3-8"}')
        self.assertEqual(argv[-3:], ["--settings", '{"model":"qwen3-8"}', "PROMPT"])

    def test_plain_json_claude_is_bare_tool_free_low_effort(self):
        argv = agent_launch.build_claude_json_command(
            "PROMPT",
            session_id="11111111-2222-4333-8444-555555555555",
        )
        self.assertIn("--bare", argv)
        self.assertEqual(argv[argv.index("--effort") + 1], "low")
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertNotIn("--json-schema", argv)
        self.assertEqual(argv[-1], "PROMPT")

    def test_claude_front_door_uses_one_plain_json_call_on_success(self):
        with fake_bridge(stub_intent()) as bridge:
            code, out, err = run_front_door(
                "fused rmsnorm in triton on sm_100",
                "--store-root", str(STORE), "--max-records", "1",
                "--agent-cli", "claude",
            )
        self.assertEqual(code, 0, err)
        self.assertEqual(bridge.clis, ["claude"])
        self.assertEqual(set(json.loads(out)), {"records", "notes"})
        self.assertNotIn("query_bridge_agent/SKILL.md", bridge.prompts[0])

    def test_plain_json_result_exposes_bridge_telemetry(self):
        payload, telemetry = query_nl.parse_claude_json_output(json.dumps({
            "result": json.dumps(stub_intent()),
            "duration_api_ms": 50,
            "ttft_ms": 10,
            "num_turns": 1,
            "usage": {"input_tokens": 11, "output_tokens": 12},
        }))
        self.assertEqual(payload, stub_intent())
        self.assertEqual(telemetry["bridge_num_turns"], 1)
        self.assertEqual(telemetry["bridge_duration_api_ms"], 50)
        self.assertEqual(telemetry["bridge_output_tokens"], 12)

    def test_invalid_plain_json_is_retried_once(self):
        invalid = {k: v for k, v in stub_intent().items() if k != "architecture"}
        with fake_bridge(None, sequence=[invalid, stub_intent()]) as bridge:
            code, _, err = run_front_door(
                "fused rmsnorm in triton on sm_100",
                "--store-root", str(STORE), "--max-records", "1",
                "--agent-cli", "claude",
            )
        self.assertEqual(code, 0, err)
        self.assertEqual(bridge.clis, ["claude", "claude"])
        self.assertIn("failed strict local validation", bridge.prompts[1])

    def test_invalid_plain_json_twice_fails_after_one_retry(self):
        invalid = {k: v for k, v in stub_intent().items() if k != "architecture"}
        with fake_bridge(None, sequence=[invalid, invalid]) as bridge:
            code, _, err = run_front_door(
                "fused rmsnorm in triton on sm_100",
                "--store-root", str(STORE), "--max-records", "1",
                "--agent-cli", "claude",
            )
        self.assertEqual(code, 4)
        self.assertEqual(bridge.clis, ["claude", "claude"])
        self.assertIn("failed after 2 attempts", err)

    def test_bridge_cli_can_be_selected_by_environment(self):
        old = os.environ.get(query_nl.BRIDGE_CLI_ENV)
        os.environ[query_nl.BRIDGE_CLI_ENV] = "claude"
        try:
            with fake_bridge(stub_intent()) as bridge:
                code, _, err = run_front_door(
                    "fused rmsnorm in triton on sm_100", "--store-root", str(STORE),
                    "--max-records", "1", "--agent-cli", "qodercli")
            self.assertEqual(code, 0, err)
            self.assertEqual(bridge.clis, ["claude"])
        finally:
            if old is None:
                os.environ.pop(query_nl.BRIDGE_CLI_ENV, None)
            else:
                os.environ[query_nl.BRIDGE_CLI_ENV] = old

    def test_metrics_are_jsonl_and_do_not_change_served_shape(self):
        with TemporaryDirectory() as tmp:
            old_log = os.environ.get(query_nl.METRICS_LOG_ENV)
            old_task = os.environ.get(query_nl.TASK_ID_ENV)
            log = Path(tmp) / "wiki.jsonl"
            os.environ[query_nl.METRICS_LOG_ENV] = str(log)
            os.environ[query_nl.TASK_ID_ENV] = "test-op"
            try:
                with fake_bridge(stub_intent()):
                    code, out, err = run_front_door(
                        "fused rmsnorm in triton on sm_100", "--store-root", str(STORE),
                        "--max-records", "1")
                self.assertEqual(code, 0, err)
                self.assertEqual(set(json.loads(out)), {"records", "notes"})
                metric = json.loads(log.read_text(encoding="utf-8"))
                self.assertEqual(metric["task_id"], "test-op")
                self.assertEqual(metric["status"], "ok")
                self.assertEqual(metric["record_count"], 1)
                self.assertEqual(metric["records_by_store"], {
                    "gpu_wiki": 1,
                    "internal_gpu_wiki": 0,
                })
                self.assertGreaterEqual(metric["latency_ms"], 0)
            finally:
                if old_log is None:
                    os.environ.pop(query_nl.METRICS_LOG_ENV, None)
                else:
                    os.environ[query_nl.METRICS_LOG_ENV] = old_log
                if old_task is None:
                    os.environ.pop(query_nl.TASK_ID_ENV, None)
                else:
                    os.environ[query_nl.TASK_ID_ENV] = old_task

    def test_launcher_captures_output(self):
        with TemporaryDirectory() as tmp:
            env = self.stub(Path(tmp), "echo out; echo err >&2; exit 3\n")
            out, err, code, timed_out = agent_launch.run("codex", "p", Path(tmp), 20, env)
        self.assertEqual((out.strip(), err.strip(), code, timed_out),
                         ("out", "err", 3, False))


if __name__ == "__main__":
    unittest.main()
