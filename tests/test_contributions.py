"""Portable contribution behavior and feature independence."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest

from aka.contracts.contributions import (
    InstructionContribution, ModuleContributions, ToolContribution, render_instructions,
)
from aka.light.agent.contributions import codex_instructions, project_command
from aka.light.measurement.contributions import declarations
from aka.light.runtime.contribution_composition import compose
from aka.light.runtime.contributions import ContributionRegistry, ToolBinding


ROOT = Path(__file__).resolve().parents[1]


def echo_module(name):
    return ModuleContributions(name, "1", (
        ToolContribution(name + ".echo", "Echo a value", {
            "type": "object", "properties": {"text": {"type": "string"}},
            "required": ["text"], "additionalProperties": False}, {"type": "object"}),
    ), (InstructionContribution(name + ".usage", "Use {{tool:" + name + ".echo}}."),))


def request(registry, tool, arguments):
    return {"catalog_id": registry.catalog()["identity"], "tool": tool, "arguments": arguments}


class ContributionsTest(unittest.TestCase):
    def test_modules_register_and_remove_independently(self):
        registry = ContributionRegistry()
        registry.register(echo_module("first"), {"first.echo": ToolBinding(lambda _, value: value)})
        registry.register(echo_module("second"), {"second.echo": ToolBinding(lambda _, value: value)})
        registry.remove("first")
        registry.seal()
        catalog = registry.catalog()
        self.assertEqual([row["id"] for row in catalog["tools"]], ["second.echo"])
        self.assertEqual([row["id"] for row in catalog["instructions"]], ["second.usage"])
        self.assertEqual(registry.invoke(None, request(registry, "second.echo", {"text": "ok"})), {"text": "ok"})
        with self.assertRaisesRegex(ValueError, "sealed"):
            registry.remove("second")

    def test_invalid_registration_is_atomic(self):
        for module, bindings in (
            (echo_module("example"), {}),
            (replace(echo_module("example"), tools=echo_module("example").tools * 2),
             {"example.echo": ToolBinding(lambda _, value: value)}),
            (replace(echo_module("example"), instructions=(InstructionContribution("example.usage", "{{tool:another.echo}}"),)),
             {"example.echo": ToolBinding(lambda _, value: value)}),
        ):
            with self.subTest(module=module):
                registry = ContributionRegistry()
                with self.assertRaises(ValueError):
                    registry.register(module, bindings)
                self.assertEqual(registry.seal().catalog()["modules"], [])

    def test_registration_snapshots_metadata_and_catalog_reads_are_detached(self):
        module = echo_module("example")
        registry = ContributionRegistry()
        registry.register(module, {"example.echo": ToolBinding(lambda _, value: value)})
        module.tools[0].input_schema["required"].clear()
        registry.seal()
        first = registry.catalog()
        first["tools"].clear()
        self.assertEqual(len(registry.catalog()["tools"]), 1)
        with self.assertRaises(ValueError):
            registry.invoke(None, request(registry, "example.echo", {}))

    def test_selection_identity_and_validation_precede_execution(self):
        calls = []
        registry = ContributionRegistry()
        registry.register(echo_module("example"), {"example.echo": ToolBinding(lambda _, value: calls.append(value) or value)})
        registry.seal()
        for body in (
            {"catalog_id": "old", "tool": "example.echo", "arguments": {"text": "ok"}},
            request(registry, "missing.echo", {}), request(registry, "example.echo", {}),
            request(registry, "example.echo", {"text": "ok", "extra": True}),
        ):
            with self.subTest(body=body), self.assertRaises(ValueError):
                registry.invoke(None, body)
        self.assertEqual(calls, [])

    def test_invalid_handler_result_is_a_runtime_failure(self):
        registry = ContributionRegistry()
        registry.register(echo_module("example"), {"example.echo": ToolBinding(lambda *_: "invalid")})
        registry.seal()
        with self.assertRaisesRegex(RuntimeError, "invalid result"):
            registry.invoke(None, request(registry, "example.echo", {"text": "ok"}))

    def test_instruction_override_is_local_versioned_and_validated(self):
        normal = compose(lambda *_args, **_kwargs: {})
        custom = compose(lambda *_args, **_kwargs: {}, instruction_overrides={
            "measurement.usage": "Measure with {{tool:measurement.evaluate}}."})
        self.assertNotEqual(normal.catalog()["identity"], custom.catalog()["identity"])
        self.assertEqual(normal.catalog()["tools"], custom.catalog()["tools"])
        for overrides in ({"unknown.usage": "text"}, {"measurement.usage": "{{tool:other.run}}"},
                          {"measurement.usage": "{{tool:invalid reference}}"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                compose(lambda *_args, **_kwargs: {}, instruction_overrides=overrides)

    def test_measurement_uses_one_execution_port_for_all_consumers(self):
        calls = []
        def execute(context, route, payload, **options):
            calls.append((context, route, payload, options))
            return {"agent_result": {"status": "completed", "record_id": "fixture"}}
        registry = compose(execute)
        body = request(registry, "measurement.evaluate", {
            "mode": "full", "shape_ids": ["shape-1"], "multi_seed": 2, "timed_runs": 7})
        self.assertEqual(registry.invoke("session", body), {"status": "completed", "record_id": "fixture"})
        self.assertEqual(calls[0][2], {"argv": ["--kind", "run", "--mode", "full", "--multi-seed", "2",
                                                     "--timed-runs", "7", "--shape-id", "shape-1"]})
        self.assertEqual(compose(execute, modules=()).catalog()["tools"], [])
        for arguments in ({"hardware": "override"}, {"multi_seed": True}, {"mode": "other"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                registry.invoke("session", request(registry, "measurement.evaluate", arguments))
        self.assertEqual(len(calls), 1)

    def test_consumer_bindings_resolve_the_same_instruction(self):
        catalog = compose(lambda *_args, **_kwargs: {}).catalog()
        native = render_instructions(catalog, {row["id"]: "aka_" + row["id"].replace(".", "_") for row in catalog["tools"]})
        cli = codex_instructions(catalog)
        self.assertIn("aka_measurement_evaluate", native)
        self.assertIn("tools/aka.py call measurement.evaluate", cli)
        self.assertNotIn("{{tool:", native + cli)
        for text in (native, cli):
            self.assertIn("reconcile retained records", text)
        command = ["codex", "exec", "resume", "--json", "session", "Optimize"]
        projected = project_command(command, {"ATREX_AKA_CONTRIBUTIONS": json.dumps(catalog)})
        self.assertEqual(projected[:-1], command[:-1])
        self.assertTrue(projected[-1].startswith("Optimize\n\n"))
        self.assertEqual(project_command(["dsh", "--", "Optimize"], {"ATREX_AKA_CONTRIBUTIONS": json.dumps(catalog)}), ["dsh", "--", "Optimize"])

    def test_feature_and_generic_adapters_have_no_cross_feature_imports(self):
        source = ROOT / "src/aka/light/measurement/contributions.py"
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module.startswith("aka."):
                self.assertEqual(node.module, "aka.contracts.contributions")
        for relative in ("runtime/contributions.py", "agent/contributions.py"):
            tree = ast.parse((ROOT / "src/aka/light" / relative).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(any("." + feature + "." in node.module for feature in ("measurement", "wiki", "workflow")))

    def test_unselected_module_is_not_imported(self):
        code = '''import sys
from aka.light.runtime.contribution_composition import compose
assert compose(lambda *_args, **_kwargs: {}, modules=()).catalog()['tools'] == []
assert not any(name.startswith('aka.light.measurement') for name in sys.modules)
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_transport_incompatible_schemas_are_rejected_during_registration(self):
        module = echo_module('example')
        for key in ('input_schema', 'output_schema'):
            tool = replace(module.tools[0], **{key: {'type': 'array', 'items': {'type': 'string'}}})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'object input and output'):
                ContributionRegistry().register(replace(module, tools=(tool,)),
                    {'example.echo': ToolBinding(lambda _, value: value)})


if __name__ == "__main__":
    unittest.main()
