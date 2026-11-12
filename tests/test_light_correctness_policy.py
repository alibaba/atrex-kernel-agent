"""Regression tests for Runtime-owned AKA-Light correctness coverage."""

from contextlib import ExitStack
import base64
import io
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from aka.contracts.evaluation import derived_evaluation_timeout
from aka.contracts.workflow import CampaignRuntimeSettings, WorkflowSettings
from aka.contracts.workspace import CandidateHandle
from aka.contracts.workflow_launch import decode_launch
from aka.legacy.task.problem.layout import agent_visible_operator_files
from aka.legacy.task.problem.legacy import LegacyProblemProvider
from aka.legacy.task.problem.native_harness import _parser as native_harness_parser
from aka.light.acceptance import numerical as numerical_review
from aka.light.acceptance.numerical import NumericalCancellation
from aka.light.acceptance.provider import (
    CandidateValidation,
    RecordedAcceptance,
    candidate_acceptance_argv,
)
from aka.light.acceptance.recorded_verifier import RecordedABBAValidator
from aka.light.acceptance.verifier import merge_abba_batch_payloads
from aka.light.acceptance.source_policy import (
    install_workspace_policy,
    optimization_mode_directive,
    production_structure_violations,
    workspace_policy_block,
)
from aka.light.evaluation.gateway import (
    COMMAND_WORKSPACE_MANIFEST,
    _command_input_paths,
    _make_input_bundle,
    _requested_gateway_kind,
    _run_typed_gateway,
    _with_workspace_performance_score,
    build_parser,
    build_typed_request,
    evaluation_policy,
    measurement_inputs,
    requested_sync_paths,
    validate_evaluation_options,
)
from aka.light.evaluation.hardware import hardware_directive
from aka.light.evaluation.evaluation_failures import (
    embedded_candidate_wall_clock_timeouts,
)
from aka.light.evaluation.evaluation_feedback import (
    evaluation_feedback,
    project_feedback,
)
from aka.light.evaluation.policy import (
    BENCH_ITERS,
    CORRECTNESS_CASES,
    SHAPE_BATCH_SIZE,
    WARMUP_ITERS,
    measurement_defaults,
    runner_overrides,
    resolved_gpu_clock_mhz,
)
from aka.light.evaluation.native_abba import (
    _clock_configuration_error,
    _clock_configuration_process,
)
from aka.light.evaluation.operations import (
    comparison_allocation_timeout,
    comparison_run_timeout,
)
from aka.light.evaluation.public_result import (
    bound_diagnostic_response,
    canonical_result,
    compact_receipt,
)
from aka.light.evaluation.projection import (
    evaluation as project_evaluation,
    gateway_error_from_process,
    gateway_error_result,
    project_response,
)
from aka.light.evaluation.result_conversion import (
    merge_optimizer_results,
    optimizer_result_from_eval,
)
from aka.light.evaluation.shapes import (
    batch_shape_ids,
    effective_shape_batch_size,
    request_shape_ids,
    shape_batch_request,
    shape_request_timeout,
)
from aka.light.execution.gateway_jobs import (
    DEFAULT_MAX_DEADLINE_RETRIES,
    DEFAULT_MAX_INFRA_RETRIES,
    TERMINAL_VALIDATION_ENV,
    annotate_deadline_timeout,
    configured_max_deadline_retries,
    configured_max_infra_retries,
    execute_job,
    retry_kind,
)
from aka.light.experiment.store import JOB_ROOT_ENV, EvidenceUnavailable
from aka.light.execution.retry import ReviewerOutputUnavailable
from aka.light.runtime.server import (
    RuntimeConfig,
    SupervisorRuntime,
    filter_options,
    managed_gateway_help,
    managed_gateway_argv,
    parse_gateway,
)
from aka.light.runtime.catalog import PluginRegistry
from aka.light.runtime.plugins import PluginError
from aka.light.runtime.request_errors import AgentRequestError
from aka.light.runtime.request_feedback import gateway_schema
from aka.light.workflow.bootstrap import (
    Campaign,
    _FRAMEWORK_BASELINE_CORRECTNESS_CONTEXT_PATHS,
    _FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_MARKERS,
    framework_baseline_workspace_contract,
    framework_baseline_validation_command,
)
from aka.light.workflow.context import conversion_required, session_execution_policy
from aka.light.workflow.plugin import Defaults as WORKFLOW_PLUGIN_DEFAULTS
from aka.light.workflow.provider import WorkflowOptions
from aka.light.workflow.output_limit import dependency_policy_recovery_prompt
from aka.light.workflow.episode import LongHorizonCampaign
from aka.light.workflow.search import SearchOptions
from aka.light.workflow.session_io import (
    _evaluation_performance_timeout,
    _render as render_file_prompt,
)
from aka.light.workspace.sources import default_tree, install, instructions, pack
from aka.light.workspace.episode import _claude_instructions
from aka.legacy.workflow.cli_conversion import adapt_argv


class RuntimeOwnedCorrectnessPolicyTests(unittest.TestCase):
    def test_native_abba_preserves_agate_managed_clock_configuration_error(self):
        raw = {
            "job_id": "ev_private",
            "status": "failed",
            "error": {
                "error_class": "infra",
                "reason": "execution_failed",
                "details": {
                    "cause_class": "AtrexConfigError",
                    "message": (
                        "manage mode requires a positive integer graphics_mhz value"
                    ),
                },
            },
        }
        process = _clock_configuration_process(
            raw, subprocess.CompletedProcess([], 1, json.dumps(raw), "private stderr")
        )

        self.assertIsNotNone(process)
        self.assertEqual(process.returncode, 75)
        projected = gateway_error_from_process(process, "same_allocation_abba")
        self.assertNotIn("job_id", projected)
        self.assertEqual(
            projected["error"],
            {
                "error_class": "infra",
                "reason": "agate_clock_configuration_invalid",
                "message": "manage mode requires a positive integer graphics_mhz value",
            },
        )
        self.assertIn("not a Kernel verdict", projected["next_action"])
        self.assertIn("Do not change the Kernel", projected["next_action"])

    def test_native_abba_clock_error_does_not_scan_candidate_output(self):
        self.assertIsNone(_clock_configuration_error({
            "result": {
                "stdout": "manage mode requires a positive integer graphics_mhz value",
                "source": "# manage mode requires a positive integer graphics_mhz value",
            }
        }))

    def test_unknown_agate_error_envelope_is_not_reclassified(self):
        envelope = {
            "operation": "check",
            "status": "running",
            "job_id": "cp_future",
            "error": None,
            "backend_state": "queued",
            "future_diagnostic": {"scheduler": "warming"},
        }

        self.assertEqual(gateway_error_result(envelope), envelope)

    def test_canonical_result_preserves_unknown_agate_error_envelope(self):
        envelope = {
            "operation": "evaluate",
            "status": "running",
            "job_id": "ev_future",
            "error": None,
            "backend_state": "queued",
            "future_diagnostic": {"scheduler": "warming"},
        }
        result = canonical_result(
            {
                "exit_code": 75,
                "stdout": "[sandbox] GATEWAY_ERROR_JSON=" + json.dumps(envelope) + "\n",
                "stderr": "",
            },
            {"operation": "evaluate", "options": {}},
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["result"]["job_id"], "ev_future")
        self.assertEqual(result["result"]["backend_state"], "queued")
        self.assertEqual(
            result["result"]["future_diagnostic"], {"scheduler": "warming"}
        )
        self.assertIsNone(result["result"]["error"])
        self.assertNotIn("next_action", result["result"])

    def test_unknown_agate_marker_is_preserved_with_universal_redaction(self):
        process = subprocess.CompletedProcess(
            [],
            75,
            '[agate] FUTURE_JSON={"shape":[128,256],"access_key":"secret",'
            '"message":"torch.Size([128, 256])"}\n',
            "candidate shape=128x256 secret=secret",
        )

        result = project_response(
            process,
            generalized=True,
            private_values=("secret",),
            operation="check",
        )

        self.assertIn("[agate] FUTURE_JSON=", result["stdout"])
        self.assertNotIn("128", result["stdout"])
        self.assertNotIn("256", result["stdout"])
        self.assertNotIn("secret", result["stdout"])
        self.assertNotIn("128", result["stderr"])
        self.assertNotIn("secret", result["stderr"])
        self.assertIn("<hidden shape>", result["stdout"])
        self.assertIn("<redacted>", result["stdout"])

        canonical = canonical_result(
            result,
            {"operation": "check", "options": {}},
        )
        self.assertEqual(canonical["status"], "failed")
        self.assertIn("[agate] FUTURE_JSON=", canonical["result"]["stdout"])
        self.assertEqual(canonical["result"]["exit_code"], 75)

        exposed = project_response(
            process,
            generalized=False,
            private_values=("secret",),
            operation="check",
        )
        self.assertIn("128", exposed["stdout"])
        self.assertIn("256", exposed["stdout"])
        self.assertNotIn("secret", exposed["stdout"])
        self.assertNotIn("<hidden shape>", exposed["stdout"])
        self.assertNotIn("outcome", canonical["result"])
        self.assertNotIn("next_action", canonical["result"])
        self.assertNotIn("error", canonical["result"])

    def test_historical_abba_result_renames_both_git_kernel_identities(self):
        result = canonical_result(
            {
                "agent_result": {
                    "operation": "evaluate",
                    "status": "completed",
                    "result": {"correct": True},
                    "gateway_record_id": "gateway-" + "1" * 32,
                    "kernel_id": "a" * 40,
                    "baseline_kernel_id": "b" * 40,
                }
            },
            {"operation": "evaluate"},
        )

        self.assertEqual(result["candidate_kernel_commit_id"], "a" * 40)
        self.assertEqual(result["baseline_kernel_commit_id"], "b" * 40)
        self.assertNotIn("kernel_commit_id", result)
        self.assertNotIn("kernel_id", result)
        self.assertNotIn("baseline_kernel_id", result)

    def test_single_candidate_result_keeps_unqualified_kernel_commit_identity(self):
        result = canonical_result(
            {
                "agent_result": {
                    "operation": "evaluate",
                    "status": "completed",
                    "result": {"correct": True},
                    "gateway_record_id": "gateway-" + "1" * 32,
                    "kernel_id": "a" * 40,
                }
            },
            {"operation": "evaluate"},
        )

        self.assertEqual(result["kernel_commit_id"], "a" * 40)
        self.assertNotIn("candidate_kernel_commit_id", result)

    def test_compact_receipt_keeps_summary_and_moves_bulk_to_record(self):
        record_id = "gateway-" + "1" * 32
        stdout = "begin\n" + "x" * 6000 + "\nend\n"
        agent_result = {
            "operation": "dev",
            "status": "completed",
            "gateway_record_id": record_id,
            "kernel_commit_id": "a" * 40,
            "result": {
                "correct": True,
                "latency_us_geomean": 4.5,
                "latency_us_by_shape": {str(index): index + 0.5 for index in range(100)},
                "stdout": stdout,
                "stderr": "warning\n",
                "exit_code": 0,
            },
        }

        receipt = compact_receipt(agent_result)

        self.assertEqual(receipt["gateway_record_id"], record_id)
        self.assertEqual(receipt["record_path"], f"records/{record_id}.json")
        self.assertTrue(receipt["result"]["correct"])
        self.assertEqual(receipt["result"]["latency_us_geomean"], 4.5)
        self.assertNotIn("latency_us_by_shape", receipt["result"])
        self.assertNotIn("stdout", receipt["result"])
        self.assertEqual(receipt["result"]["stdout_bytes"], len(stdout.encode()))
        self.assertTrue(receipt["result"]["stdout_truncated"])
        self.assertIn("begin", receipt["result"]["stdout_preview"])
        self.assertIn("end", receipt["result"]["stdout_preview"])
        self.assertEqual(receipt["stdout_location"], {
            "path": f"records/{record_id}.json",
            "field": "result.stdout",
        })
        self.assertEqual(receipt["stderr_location"], {
            "path": f"records/{record_id}.json",
            "field": "result.stderr",
        })
        self.assertIn("latency_us_by_shape", receipt["result_fields"])
        self.assertIn("Read or query that JSON file directly", receipt["details"])
        self.assertNotIn("tools/record.py", receipt["details"])
        self.assertEqual(agent_result["result"]["stdout"], stdout)
        self.assertEqual(len(agent_result["result"]["latency_us_by_shape"]), 100)

    def test_compact_receipt_surfaces_bounded_actionable_compiler_tail(self):
        record_id = "gateway-" + "3" * 32
        compiler_tail = "kernel.cu:91: error: no matching function for call"
        long_message = "template instantiation note\n" * 400 + compiler_tail
        diagnostics = [
            {
                "stage": "compile",
                "shape_id": str(index),
                "message": long_message if index == 0 else f"diagnostic {index}",
            }
            for index in range(6)
        ]
        agent_result = {
            "operation": "evaluate",
            "status": "failed",
            "gateway_record_id": record_id,
            "result": {
                "correct": False,
                "actionable_diagnostics": diagnostics,
            },
        }

        receipt = compact_receipt(agent_result)

        self.assertEqual(len(receipt["actionable_diagnostics"]), 4)
        first = receipt["actionable_diagnostics"][0]
        self.assertEqual(first["stage"], "compile")
        self.assertEqual(first["shape_id"], "0")
        self.assertIn(compiler_tail, first["message"])
        self.assertTrue(first["message_truncated"])
        self.assertEqual(first["message_bytes"], len(long_message.encode()))
        self.assertEqual(receipt["actionable_diagnostics_count"], 6)
        self.assertTrue(receipt["actionable_diagnostics_truncated"])
        self.assertEqual(receipt["actionable_diagnostics_location"], {
            "path": f"records/{record_id}.json",
            "field": "result.actionable_diagnostics",
        })
        self.assertEqual(
            agent_result["result"]["actionable_diagnostics"][0]["message"],
            long_message,
        )

    def test_compact_receipt_omits_actionable_diagnostics_when_absent(self):
        receipt = compact_receipt({
            "operation": "evaluate",
            "status": "completed",
            "result": {"correct": True},
        })

        self.assertNotIn("actionable_diagnostics", receipt)
        self.assertNotIn("actionable_diagnostics_location", receipt)

    def test_env_gateway_response_preserves_redacted_executor_stdout(self):
        from aka.light.runtime.server import agent_gateway_response

        stdout = (
            '[sandbox] ENV_JSON={"gpu":"L20N","compute_capability":"8.9",'
            '"frameworks":{"cuda":{"available":true}}}\n'
        )
        response, execution_exit_code = agent_gateway_response({
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "agent_result": {
                "operation": "env",
                "status": "completed",
                "result": {
                    "gpu": "L20N",
                    "compute_capability": "8.9",
                    "frameworks": {"cuda": {"available": True}},
                },
            },
        })

        self.assertEqual(response, {
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
        })
        self.assertIsNone(execution_exit_code)

    def test_measurement_gateway_response_remains_compact(self):
        from aka.light.runtime.server import agent_gateway_response

        record_id = "gateway-" + "2" * 32
        response, execution_exit_code = agent_gateway_response({
            "exit_code": 0,
            "stdout": "private compatibility stream\n",
            "stderr": "",
            "agent_result": {
                "operation": "evaluate",
                "status": "completed",
                "gateway_record_id": record_id,
                "kernel_commit_id": "a" * 40,
                "result": {"correct": True, "latency_us_geomean": 4.0},
            },
        })

        self.assertEqual(execution_exit_code, 0)
        self.assertEqual(response["record_path"], f"records/{record_id}.json")
        self.assertNotIn("stdout", response)
        self.assertEqual(response["result"]["latency_us_geomean"], 4.0)

    def test_new_abba_result_names_candidate_and_baseline_symmetrically(self):
        result = canonical_result(
            {
                "exit_code": 0,
                "stdout": "[sandbox] ABBA_JSON=" + json.dumps({
                    "correct": True,
                    "baseline": {"correct": True, "latency_us_geomean": 12.0},
                    "candidate": {"correct": True, "latency_us_geomean": 10.0},
                }) + "\n",
                "stderr": "",
            },
            {"operation": "same_allocation_abba", "options": {}},
            record_id="gateway-" + "1" * 32,
            kernels={
                "candidate": {"kernel_id": "a" * 40},
                "baseline": {"kernel_id": "b" * 40},
            },
        )

        self.assertEqual(result["candidate_kernel_commit_id"], "a" * 40)
        self.assertEqual(result["baseline_kernel_commit_id"], "b" * 40)
        self.assertNotIn("kernel_commit_id", result)

    def test_episode_claude_instructions_include_current_mode_policy(self):
        production = workspace_policy_block(
            "production",
            "Cuda",
            source_checks_enabled=True,
            specify_dsl=True,
        )
        projected = _claude_instructions(production).decode()
        self.assertIn("# Kernel engineering constraints", projected)
        self.assertIn("Optimization mode: production (hard gate)", projected)
        self.assertIn("submitted candidate's implementation framework is exactly **Cuda**", projected)
        self.assertIn("During development, you may use any language or DSL", projected)
        self.assertIn("final-candidate constraint, not a development-method restriction", projected)
        self.assertEqual(projected.count("ATREX_OPTIMIZATION_MODE_POLICY_BEGIN"), 1)

        without_numerical = workspace_policy_block(
            "production",
            "Cuda",
            source_checks_enabled=True,
            specify_dsl=True,
            numerical_validation_enabled=False,
        )
        self.assertIn("candidate_ready production candidate", without_numerical)
        self.assertIn("When the optional Framework Baseline stage is enabled", without_numerical)
        self.assertNotIn("Numerical review produces targeted", without_numerical)

        leaderboard = _claude_instructions(
            workspace_policy_block(
                "leaderboard",
                "Cuda",
                source_checks_enabled=True,
                specify_dsl=True,
            )
        ).decode()
        self.assertIn("Optimization mode: leaderboard", leaderboard)
        self.assertNotIn("production (hard gate)", leaderboard)

        open_dsl = _claude_instructions(
            workspace_policy_block(
                "production",
                "Cuda",
                source_checks_enabled=True,
                specify_dsl=False,
            )
        ).decode()
        self.assertIn("Optimization mode: production (open DSL)", open_dsl)
        self.assertNotIn("submitted candidate's implementation framework is exactly **Cuda**", open_dsl)

        with self.assertRaisesRegex(TypeError, "workspace_policy must be text"):
            RuntimeConfig("L20D", workspace_policy=1)

    def test_search_role_specialization_defaults_off_and_can_be_enabled(self):
        self.assertFalse(SearchOptions().role_specialization)
        self.assertTrue(SearchOptions(role_specialization=True).role_specialization)
        with self.assertRaisesRegex(TypeError, "role_specialization must be a boolean"):
            SearchOptions(role_specialization=1)

        for enabled in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                base = SimpleNamespace(
                    workspace=workspace,
                    platform="L20D",
                    arch="sm_103",
                    notes="none",
                    research_records=False,
                    research_modules=(),
                    search_coordinate={
                        "epoch": 1,
                        "trajectory": 1,
                        "episode": 1,
                        "role_specialization": enabled,
                    },
                    module_options=SimpleNamespace(goal_episodes=False),
                    specify_dsl=True,
                    framework="Cuda",
                    dsl_context="- DSL: `Cuda`",
                    plugin_directive=lambda stage: "",
                    tool_runtime=SimpleNamespace(
                        pause_agent_during_requests=True
                    ),
                )
                prompt = LongHorizonCampaign(base)._prompt(
                    episode=1,
                    version=1,
                    worktree=CandidateHandle(
                        1, "a" * 40, "work", workspace
                    ),
                    conversion_pending=False,
                )
            self.assertIn(
                "Search location: Epoch 1, Trajectory 1, Episode 1", prompt
            )
            self.assertEqual(
                "T1 role: bounded exploitation" in prompt,
                enabled,
            )
            self.assertIn("PROGRESS.md", prompt)
            self.assertNotIn("## Research Board", prompt)
            self.assertIn("unfinished optimization WIP", prompt)
            self.assertNotIn("active Direction and major-redesign WIP", prompt)
            self.assertNotIn("high-upside Direction", prompt)
            self.assertNotIn("selected Direction or optimization path", prompt)
            self.assertNotIn("first failed Direction", prompt)

    def test_episode_prompt_has_one_workflow_and_modular_contracts(self):
        prompt_dir = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "aka"
            / "light"
            / "workflow"
            / "prompts"
        )
        episode = (prompt_dir / "episode.md").read_text(encoding="utf-8")
        self.assertEqual(episode.count("## Workflow"), 1)
        self.assertNotIn("## Terms", episode)
        self.assertLess(len(episode.split()), 400)
        self.assertIn("{{RESEARCH_MODULE_RULES}}", episode)
        self.assertIn("{{HANDOFF_CONTRACT}}", episode)
        self.assertIn("{{RUNTIME_WAIT_POLICY}}", episode)
        self.assertIn("skills/runtime-records/SKILL.md", episode)
        self.assertIn("Agent-visible comparison base: `{{BASE_COMMIT}}`", episode)
        self.assertIn("measurable candidate identity: clean Git HEAD", episode)
        self.assertNotIn("Promoted campaign version", episode)
        self.assertIn("Hardware: `{{ARCH}}`", episode)
        self.assertIn("Gateway Records are measurement authority", episode)
        self.assertIn("highest plausible full-workload latency impact", episode)
        self.assertIn("A Direction is a durable optimization program", episode)
        self.assertIn(
            "A single implementation, parameter sweep, capability probe, or mechanism "
            "test is an Experiment, not a new Direction",
            episode,
        )
        self.assertIn("discarding higher-upside work", episode)
        self.assertIn("commit the complete candidate", episode)
        self.assertIn("smallest measurement that resolves the current question", episode)
        self.assertIn("Repeat `probe → candidate → measurement → revert/keep` cycles", episode)
        self.assertNotIn("at most one local Direction `in_progress`", episode)
        self.assertIn("Submit `episode-report` only when", episode)
        self.assertIn("Use `candidate_ready` only for a changed candidate", episode)
        self.assertIn("submit `pivot`; never add a nominal edit", episode)
        self.assertIn("`cross_trajectory_share` is available only", episode)
        self.assertNotIn("high-upside Direction", episode)
        self.assertNotIn("selected Direction or optimization path", episode)
        self.assertNotIn("{{RESEARCH_BOARD_SECTION}}", episode)
        for removed in (
            "{{MODE_POLICY}}",
            "{{EVALUATOR}}",
            "{{SANDBOX}}",
            "{{AGENT_RUNTIME}}",
            "{{WIKI_DIRECTIVE}}",
            "{{REPORT_SELECTION}}",
        ):
            self.assertNotIn(removed, episode)
        for name in (
            "episode_unrecorded.md",
            "episode_experiments.md",
            "episode_directions.md",
            "episode_research.md",
        ):
            fragment = (prompt_dir / name).read_text(encoding="utf-8")
            self.assertNotIn("## Workflow", fragment)
            self.assertNotIn("## Handoff", fragment)
        self.assertIn(
            "supporting_experiment_ids",
            (prompt_dir / "episode_research.md").read_text(encoding="utf-8"),
        )
        directions = (prompt_dir / "episode_directions.md").read_text(
            encoding="utf-8"
        )
        research = (prompt_dir / "episode_research.md").read_text(
            encoding="utf-8"
        )
        for fragment in (directions, research):
            self.assertIn("only an `in_progress` Direction may", fragment)
        self.assertIn("closure evidence uses `gateway_record_ids`", directions)
        self.assertIn(
            "A supported Experiment updates the evidence for its Direction; "
            "it does not imply that the Direction is complete",
            research,
        )
        self.assertIn(
            "Keep the Direction active while any high-value implementation variant, "
            "parameter family, workload regime, or combination remains unexplored",
            research,
        )
        self.assertIn(
            "The default next action is to update the current Direction's remaining plan "
            "and continue",
            research,
        )
        self.assertIn(
            "Create a derived Direction only when the optimization objective or "
            "applicability changes materially",
            research,
        )
        self.assertIn(
            "After each Experiment, compare the remaining branches within the current "
            "Direction before deciding whether the overall optimization program is exhausted",
            research,
        )

        claim_review = (prompt_dir / "research_claim_review.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("important enough to organize durable work", claim_review)
        self.assertIn("plan covers a meaningful strategy space", claim_review)
        self.assertIn(
            "Reject a proposed Direction when it describes a single candidate, parameter "
            "choice, isolated probe, or one-step mechanism",
            claim_review,
        )
        self.assertIn(
            "Do not treat one supported Experiment or one supported mechanism as completion",
            claim_review,
        )

        baseline = (prompt_dir / "framework_baseline.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("existing SourceProvider layout", baseline)
        self.assertIn("`WORKSPACE.md`", baseline)
        self.assertIn("Reference version:", baseline)
        self.assertIn("Hardware: `{{ARCH}}`", baseline)
        self.assertIn("Wiki query is a knowledge request, not a Kernel measurement", baseline)
        self.assertIn("Work like a normal Episode", baseline)
        self.assertIn("Use Runtime Evaluate to establish correctness", baseline)
        self.assertIn("commit the complete candidate and leave the worktree clean", baseline)
        self.assertIn("does not require a speedup over V0", baseline)
        self.assertIn("passes the standard full Evaluate", baseline)
        self.assertNotIn("Do not invoke GPU measurement", baseline)
        self.assertNotIn("recovery header", baseline)
        self.assertNotIn("rerun the bounded V1 smoke", baseline)

        wiki_plugin = Path(__file__).resolve().parents[1] / "plugins/gpu-wiki"
        wiki_baseline = (wiki_plugin / "framework_baseline.md").read_text(
            encoding="utf-8"
        )
        wiki_common = (wiki_plugin / "instructions.md").read_text(
            encoding="utf-8"
        )
        for wiki_prompt in (wiki_baseline, wiki_common):
            self.assertIn("publicly disclosed Shape/domain facts", wiki_prompt)
            self.assertIn("infer or request private Shape cases", wiki_prompt)
        boundary = (prompt_dir / "sandbox_safety_boundary.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("selected SourceProvider layout", boundary)
        self.assertIn("SourceProvider explicitly declares", boundary)
        full_workflow = (prompt_dir / "sandbox_full_workflow.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("configured coverage, seeds and repetitions", full_workflow)
        self.assertNotIn("five-case correctness", full_workflow)

        claude = (
            Path(__file__).resolve().parents[1]
            / "src/aka/light/workspace/resources/reference/CLAUDE.md"
        ).read_text(encoding="utf-8")
        self.assertIn("selected SourceProvider layout", claude)
        self.assertIn("`shapes.json` is the authoritative public evaluator workload", claude)
        self.assertIn("generated tensor values", claude)
        self.assertIn("Runtime regenerates after accepted state changes", claude)
        self.assertNotIn("continuously rewrites root `PROGRESS.md`", claude)

        runtime_records = (
            Path(__file__).resolve().parents[1]
            / "src/aka/light/workspace/resources/skills/runtime-records/SKILL.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Multiple local Directions may be `in_progress`", runtime_records)
        self.assertNotIn("at most three distinct Directions", runtime_records)
        self.assertIn("not necessarily an immediate source edit", runtime_records)
        self.assertIn("should normally span multiple candidates", runtime_records)
        self.assertIn(
            "A single implementation, parameter sweep, capability probe, or mechanism "
            "test belongs under an existing Direction",
            runtime_records,
        )
        self.assertIn(
            "Do not use Directions for Runtime/evaluator maintenance", runtime_records
        )
        self.assertIn("genealogy only", (
            Path(__file__).resolve().parents[1]
            / "src/aka/light/workspace/resources/skills/runtime-records/references/journal.md"
        ).read_text(encoding="utf-8"))
        self.assertIn("does not inherit parent status", (
            Path(__file__).resolve().parents[1]
            / "src/aka/light/workspace/resources/skills/runtime-records/references/journal.md"
        ).read_text(encoding="utf-8"))

        for prompt_path in sorted(prompt_dir.glob("runtime_*.md")):
            source = prompt_path.read_text(encoding="utf-8")
            self.assertNotIn("supplemental numerical validation when enabled, and promotion", source)
            self.assertNotIn("Report acceptance is a handoff, not promotion", source)
            for language, body in re.findall(
                r"```(json|jsonl)\n(.*?)\n```", source, flags=re.DOTALL
            ):
                if language == "json":
                    json.loads(body)
                else:
                    for line in body.splitlines():
                        if line.strip():
                            json.loads(line)
        self.assertIn("Report acceptance is a handoff, not promotion", episode)

        agent_docs = [
            *prompt_dir.glob("*.md"),
            *(Path(__file__).resolve().parents[1]
              / "src/aka/light/workspace/resources/skills/runtime-records").rglob("*.md"),
            *(Path(__file__).resolve().parents[1]
              / "src/aka/light/workspace/resources/skills/runtime-report").rglob("*.md"),
            *(Path(__file__).resolve().parents[1]
              / "src/aka/light/workspace/resources/skills/gpu-measurement").rglob("*.md"),
        ]
        for path in agent_docs:
            source = path.read_text(encoding="utf-8")
            self.assertNotRegex(source, r"gateway-\.\.\.")
            self.assertNotRegex(source, r"experiment_\.\.\.")
            self.assertNotRegex(source, r"direction_\.\.\.")

        single_file_contract = instructions(None)
        self.assertIn("## Single-file Kernel", single_file_contract)
        self.assertIn("`kernel.py` is the complete candidate", single_file_contract)
        self.assertIn("do not add auxiliary candidate sources", single_file_contract)

        resource_dir = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "aka"
            / "light"
            / "workspace"
            / "resources"
            / "skills"
        )
        records_reference = (
            resource_dir / "runtime-records" / "references" / "records.md"
        ).read_text(encoding="utf-8")
        self.assertIn("`shapes.json` only in exposed mode", records_reference)
        self.assertNotIn("examples. Shape IDs are opaque.", records_reference)
        records_skill = (resource_dir / "runtime-records" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        measurement_skill = (resource_dir / "gpu-measurement" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        measurement_requests = (
            resource_dir / "gpu-measurement" / "references" / "requests.md"
        ).read_text(encoding="utf-8")
        sandbox_boundary = (
            prompt_dir / "sandbox_safety_boundary.md"
        ).read_text(encoding="utf-8")
        for source in (measurement_skill, measurement_requests, sandbox_boundary):
            normalized = re.sub(r"\s+", " ", source)
            self.assertIn("self-contained single-file probe", normalized)
            self.assertIn("other workspace", normalized)
        self.assertIn(
            "Dev uploads only its declared inputs",
            re.sub(r"\s+", " ", measurement_requests),
        )
        self.assertNotIn("contracted candidate is supplied", measurement_requests)
        self.assertIn("Knowledge queries, when installed", records_skill)
        report_skill = (resource_dir / "runtime-report" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("same Git and Runtime measurement loop", report_skill)
        self.assertIn("submit the ordinary `episode-report`", report_skill)
        self.assertIn("does not require ABBA", report_skill)
        self.assertIn("`pivot` and `blocked` reports omit them", report_skill)
        self.assertNotIn("smoke-and-exit procedure", report_skill)
        wiki_plugin = Path(__file__).resolve().parents[1] / "plugins/gpu-wiki"
        wiki_episode = (wiki_plugin / "episode.md").read_text(encoding="utf-8")
        wiki_common = (wiki_plugin / "instructions.md").read_text(encoding="utf-8")
        self.assertIn("tools/sandbox.py --kind wiki-query", wiki_episode)
        self.assertIn("skills/KernelWiki/SKILL.md", wiki_common)
        self.assertIn("missing hardware or implementation fact", wiki_episode)
        self.assertIn("runtime architecture {{ARCH}}", wiki_episode)
        self.assertIn("skills/KernelWiki/SKILL.md", wiki_episode)
        self.assertNotIn("aka.legacy.resources.cli", wiki_episode)
        self.assertNotIn("aka.legacy.resources.cli", wiki_common)

        numerical_review = (prompt_dir / "numerical_review.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("candidate/<actual-source-file>:location", numerical_review)
        self.assertNotIn("candidate/kernel.py:location", numerical_review)

        dependency_review = (
            Path(__file__).resolve().parents[1]
            / "src/aka/legacy/workflow/prompts/dependency_review.md"
        ).read_text(encoding="utf-8")
        self.assertIn("candidate/<replace-with-an-actual-file>:line-or-field", dependency_review)
        self.assertIn("native manifest candidate", dependency_review)

    def test_every_research_module_combination_renders_one_complete_prompt(self):
        expected_heading = {
            (): "## Research records",
            ("directions",): "## Direction records",
            ("experiments",): "## Experiment records",
            ("directions", "experiments"): "## Direction and Experiment records",
        }
        for modules, heading in expected_heading.items():
            with self.subTest(modules=modules), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                base = SimpleNamespace(
                    workspace=workspace,
                    platform="L20D",
                    arch="sm_103a",
                    notes="none",
                    research_records=bool(modules),
                    research_modules=modules,
                    search_coordinate=None,
                    module_options=SimpleNamespace(goal_episodes=False),
                    specify_dsl=True,
                    framework="Cuda",
                    dsl_context="- Required final candidate DSL: `Cuda`.",
                    plugin_directive=lambda stage: "",
                    tool_runtime=SimpleNamespace(pause_agent_during_requests=True),
                )
                prompt = LongHorizonCampaign(base)._prompt(
                    episode=1,
                    version=1,
                    worktree=CandidateHandle(1, "a" * 40, "work", workspace),
                    conversion_pending=False,
                )
                self.assertNotIn("{{", prompt)
                self.assertEqual(prompt.count("## Workflow"), 1)
                self.assertEqual(prompt.count(heading), 1)
                self.assertEqual(prompt.count("## Handoff"), 1)
                self.assertIn(heading, prompt)
                self.assertIn("skills/runtime-records/SKILL.md", prompt)
                if "experiments" in modules:
                    self.assertIn("selected_experiment_id", prompt)
                else:
                    self.assertIn("gateway_record_ids", prompt)
                if modules == ("directions",):
                    self.assertIn("`supporting_experiment_ids`", prompt)
                    self.assertIn("Experiment recording is disabled", prompt)

    def test_pooled_prompt_does_not_enable_roles_when_coordinate_omits_the_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            base = SimpleNamespace(
                workspace=workspace,
                platform="L20D",
                arch="sm_103a",
                notes="none",
                research_records=False,
                research_modules=(),
                search_coordinate={"epoch": 1, "trajectory": 2, "episode": 1},
                module_options=SimpleNamespace(goal_episodes=False),
                specify_dsl=True,
                framework="Cuda",
                dsl_context="- Required final candidate DSL: `Cuda`.",
                plugin_directive=lambda stage: "",
                tool_runtime=SimpleNamespace(pause_agent_during_requests=True),
            )
            prompt = LongHorizonCampaign(base)._prompt(
                episode=1,
                version=1,
                worktree=CandidateHandle(1, "a" * 40, "work", workspace),
                conversion_pending=False,
            )
        self.assertIn("Search location: Epoch 1, Trajectory 2, Episode 1", prompt)
        self.assertNotIn("T2 role:", prompt)

    def test_optional_conversion_and_ppu_prompt_fragments_render_completely(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            base = SimpleNamespace(
                workspace=workspace,
                platform="ZW-M890P",
                arch="ppu15",
                notes="none",
                research_records=True,
                research_modules=("directions", "experiments"),
                search_coordinate=None,
                module_options=SimpleNamespace(goal_episodes=False),
                specify_dsl=True,
                framework="Gluon",
                dsl_context="- Required final candidate DSL: `Gluon`.",
                plugin_directive=lambda stage: "",
                tool_runtime=SimpleNamespace(pause_agent_during_requests=True),
            )
            prompt = LongHorizonCampaign(base)._prompt(
                episode=2,
                version=3,
                worktree=CandidateHandle(2, "a" * 40, "work", workspace),
                conversion_pending=True,
            )
        self.assertNotIn("{{", prompt)
        self.assertIn("## Required conversion", prompt)
        self.assertIn("## PPU diagnostics", prompt)
        self.assertIn("within 5%", prompt)

    def test_framework_review_prompt_is_complete_and_dsl_neutral(self):
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "aka"
            / "light"
            / "workflow"
            / "prompts"
            / "framework_baseline_correctness_review.md"
        )
        rendered = render_file_prompt(
            prompt_path,
            REVIEWER="Codex",
            IMPLEMENTATION_TARGET="a Kernel in an evaluator-supported GPU DSL",
            PLATFORM="L20D",
            ARCH="sm_103a",
        )
        self.assertNotIn("{{", rendered)
        self.assertIn("TARGET_IMPLEMENTATION_RISKS:", rendered)
        self.assertIn("PRE_VALIDATION_CHECKS:", rendered)
        self.assertIn("context/shapes.json", rendered)
        self.assertIn("eventual implementation path", rendered)
        self.assertNotIn("CUDA_IMPLEMENTATION_RISKS:", rendered)
        with self.assertRaisesRegex(ValueError, "IMPLEMENTATION_TARGET"):
            render_file_prompt(
                prompt_path,
                REVIEWER="Codex",
                PLATFORM="L20D",
                ARCH="sm_103a",
            )
        self.assertIn(
            "metadata.json", _FRAMEWORK_BASELINE_CORRECTNESS_CONTEXT_PATHS
        )
        self.assertNotIn(
            "public_shapes.json", _FRAMEWORK_BASELINE_CORRECTNESS_CONTEXT_PATHS
        )
        self.assertIn(
            "TARGET_IMPLEMENTATION_RISKS:",
            _FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_MARKERS,
        )

    def test_plugin_prompt_fragments_reject_unresolved_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode.md").write_text(
                "Hardware {{ARCH}}; missing {{UNDECLARED}}", encoding="utf-8"
            )
            registry = object.__new__(PluginRegistry)
            registry.plugins = [SimpleNamespace(
                root=root,
                manifest={"instructions": {"episode": "episode.md"}},
            )]
            with self.assertRaisesRegex(PluginError, "UNDECLARED"):
                registry.instructions("episode", ARCH="sm_103a")

            (root / "episode.md").write_text(
                "Hardware {{ARCH}}", encoding="utf-8"
            )
            self.assertIn(
                "Hardware sm_103a",
                registry.instructions("episode", ARCH="sm_103a"),
            )

    def test_optional_dsl_settings_and_cli_normalization(self):
        settings = WorkflowSettings(platform="L20D", framework="Cuda")
        self.assertTrue(settings.specify_dsl)
        self.assertTrue(settings.expose_evaluator_shapes)
        self.assertEqual(settings.framework_baseline, "auto")
        with self.assertRaises(TypeError):
            WorkflowSettings(platform="L20D", framework="Cuda", specify_dsl=1)
        self.assertFalse(WorkflowSettings(
            platform="L20D", framework="Cuda", expose_evaluator_shapes=False,
        ).expose_evaluator_shapes)
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            WorkflowSettings(
                platform="L20D", framework="Cuda", expose_evaluator_shapes=1,
            )

        launch = decode_launch(
            adapt_argv(
                [
                    "--op-dir",
                    str(Path.cwd()),
                    "--platform",
                    "L20D",
                    "--framework",
                    "Cuda",
                    "--no-specify-dsl",
                ]
            )
        )
        self.assertFalse(launch.request.settings.specify_dsl)
        self.assertEqual(launch.request.settings.framework, "")
        self.assertEqual(launch.request.settings.framework_baseline, "auto")
        self.assertTrue(launch.request.settings.expose_evaluator_shapes)
        self.assertIn("specify_dsl", launch.provided_settings)

        launch = decode_launch(
            adapt_argv(
                [
                    "--op-dir",
                    str(Path.cwd()),
                    "--platform",
                    "L20D",
                    "--framework",
                    "Cuda",
                    "--no-expose-evaluator-shapes",
                ]
            )
        )
        self.assertFalse(launch.request.settings.expose_evaluator_shapes)
        self.assertIn("expose_evaluator_shapes", launch.provided_settings)

    def test_cli_mounts_one_or_many_reference_directories_by_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nvidia = root / "nvidia"
            cuda_guides = root / "cuda-guides"
            nvidia.mkdir()
            cuda_guides.mkdir()

            launch = decode_launch(
                adapt_argv(
                    [
                        "--op-dir",
                        str(Path.cwd()),
                        "--platform",
                        "L20D",
                        "--framework",
                        "Cuda",
                        "--agent-reference-dir",
                        str(nvidia),
                        str(cuda_guides),
                    ]
                )
            )

            self.assertEqual(
                launch.request.settings.agent_reference_paths,
                (
                    ("nvidia", str(nvidia.resolve())),
                    ("cuda-guides", str(cuda_guides.resolve())),
                ),
            )
            self.assertIn("agent_reference_paths", launch.provided_settings)

    def test_reference_directories_require_bwrap_at_campaign_boundary(self):
        with self.assertRaisesRegex(ValueError, "requires.*sandbox=bwrap"):
            Campaign(
                name="operator",
                kernel_demo="reference.py",
                platform="L20D",
                framework="Cuda",
                agent_reference_paths=(("nvidia", "/knowledge/nvidia"),),
            )

    def test_campaign_uses_framework_baseline_auto_policy_by_default(self):
        campaign = Campaign(
            name="operator",
            kernel_demo="reference.py",
            platform="cuda",
            framework="cuda",
        )
        self.assertEqual(campaign.framework_baseline, "auto")

    def test_open_dsl_policy_is_neutral_and_disables_conversion(self):
        directive = optimization_mode_directive(
            "production", "Cuda", specify_dsl=False
        )
        self.assertIn("No DSL is prescribed", directive)
        self.assertNotIn("`Cuda`", directive)
        campaign = SimpleNamespace(
            source_checks_enabled=True,
            specify_dsl=False,
            framework="Triton",
            convert_after=0,
        )
        self.assertFalse(conversion_required(campaign, 99, Path(".")))

    def test_leaderboard_open_dsl_policy_does_not_request_framework_benchmarking(self):
        directive = optimization_mode_directive(
            "leaderboard", "Cuda", specify_dsl=False
        )
        self.assertIn(
            "You may freely choose any evaluator-supported GPU implementation DSL.",
            directive,
        )
        self.assertNotIn("measured evidence", directive)
        self.assertNotIn("Choose the implementation framework", directive)

    def test_open_dsl_structure_and_workspace_policy_do_not_pin_framework(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            policy, files = default_tree(
                {"kernel.py": b"def run(*args):\n    return args\n"}
            )
            install(workspace, pack(files, policy))
            self.assertEqual(
                production_structure_violations(
                    workspace, "", specify_dsl=False
                ),
                [],
            )
            self.assertIn(
                "unsupported production framework",
                production_structure_violations(
                    workspace, "", specify_dsl=True
                )[0],
            )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(
                "def run(*args):\n    return args\n", encoding="utf-8"
            )
            self.assertEqual(
                production_structure_violations(
                    workspace, "", specify_dsl=False
                ),
                [],
            )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            install_workspace_policy(
                workspace,
                "production",
                "Cuda",
                specify_dsl=False,
                update_tracked_files=False,
            )
            state = json.loads(
                (workspace / ".orchestrator_mode.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["framework"], "")
            self.assertFalse(state["specify_dsl"])
            self.assertEqual(state["evaluator_shape_visibility"], "exposed")
            install_workspace_policy(
                workspace,
                "production",
                "Cuda",
                specify_dsl=False,
                expose_evaluator_shapes=False,
                update_tracked_files=False,
            )
            state = json.loads(
                (workspace / ".orchestrator_mode.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["evaluator_shape_visibility"], "exposed")

    def test_open_dsl_framework_baseline_prompt_is_neutral(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            campaign = Campaign(
                name="op",
                kernel_demo=str(Path(directory) / "reference.py"),
                platform="L20D",
                framework="Cuda",
                specify_dsl=False,
                work_dir=directory,
            )
            campaign.tool_runtime = SimpleNamespace(
                wiki=False,
                pause_agent_during_requests=True,
            )
            campaign.plugin_directive = lambda _phase: ""
            campaign._framework_baseline_correctness_guidance_text = lambda: ""
            campaign._mode_directive = lambda: optimization_mode_directive(
                "production", "", specify_dsl=False
            )
            prompt = campaign._framework_baseline_prompt(1)
            self.assertIn("Choose a supported DSL", prompt)
            self.assertNotIn("- DSL: `Cuda`", prompt)
            self.assertNotIn("yourself in `Cuda`", prompt)
            self.assertNotIn("{{", prompt)

    def test_named_reference_knowledge_is_explained_without_becoming_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            campaign = Campaign(
                name="op",
                kernel_demo=str(Path(directory) / "reference.py"),
                platform="L20D",
                framework="Cuda",
                work_dir=directory,
                agent_sandbox="bwrap",
                agent_reference_paths=(("nvidia", "/knowledge/nvidia"),),
            )
            campaign.tool_runtime = SimpleNamespace(
                wiki=False,
                pause_agent_during_requests=True,
            )
            campaign.plugin_directive = lambda _phase: ""
            campaign._framework_baseline_correctness_guidance_text = lambda: ""
            prompt = campaign._framework_baseline_prompt(1)
        self.assertIn("`references/nvidia/`", prompt)
        self.assertIn("read-only engineering references", prompt)
        self.assertIn("not the task contract, measurement evidence, or candidate source", prompt)

    def test_disabled_framework_correctness_review_has_no_prompt_section(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            campaign = Campaign(
                name="op",
                kernel_demo="reference.py",
                platform="L20D",
                framework="Cuda",
                work_dir=directory,
            )
            self.assertEqual(
                campaign._framework_baseline_correctness_guidance_text(), ""
            )

    def test_hardware_prompt_contains_only_dynamic_runtime_arch(self):
        directive = hardware_directive("masked-product", "vendor_arch_987")

        self.assertEqual(directive, "Hardware: **vendor_arch_987**\n")
        self.assertNotIn("masked-product", directive)
        self.assertNotIn("sm_103", directive)
        self.assertNotIn("tcgen05", directive)

    def _framework_baseline_campaign(self, directory):
        return Campaign(
            name="operator",
            kernel_demo=str(Path(directory) / "reference.py"),
            platform="cuda",
            framework="cuda",
            work_dir=directory,
            optimization_mode="leaderboard",
            framework_baseline="auto",
        )

    def _framework_baseline_decision_patches(
        self,
        directory,
        *,
        reference_blob,
        implementation_blob,
        structure_side_effect=None,
    ):
        def committed_blob(_workspace, _revision, path):
            if path == "reference.py":
                return reference_blob
            if path == "kernel.py":
                return implementation_blob
            return ""

        structure_patch = (
            patch(
                "aka.light.workflow.bootstrap.production_structure_violations",
                side_effect=structure_side_effect,
            )
            if structure_side_effect is not None
            else patch(
                "aka.light.workflow.bootstrap.production_structure_violations",
                return_value=[],
            )
        )
        return (
            patch("aka.light.workflow.bootstrap.latest_version", return_value=0),
            patch("aka.light.workflow.bootstrap.git_head", return_value="head"),
            patch(
                "aka.light.workflow.bootstrap.resolve_framework_baseline_commit",
                return_value=(None, None),
            ),
            patch(
                "aka.light.workflow.bootstrap.head_kernel_is_initial_baseline",
                return_value=True,
            ),
            patch(
                "aka.light.workflow.bootstrap.framework_baseline_progress_path",
                return_value=Path(directory) / "missing-progress.json",
            ),
            patch.object(
                Campaign,
                "source_checks_enabled",
                new_callable=PropertyMock,
                return_value=True,
            ),
            structure_patch,
            patch.object(Campaign, "_v0_baseline_commit", return_value="v0"),
            patch("aka.light.workspace.sources.revision_contains_reference",
                  return_value=reference_blob == implementation_blob),
            patch("aka.light.workspace.sources.layout_at",
                  return_value=SimpleNamespace(replaceable_baseline=True)),
            patch("aka.light.workspace.sources.read",
                  return_value=SimpleNamespace(snapshot=b"source-tree")),
            patch("aka.light.workspace.sources.from_revision",
                  return_value=SimpleNamespace(snapshot=b"source-tree")),
            patch(
                "aka.light.workflow.bootstrap.git_path_blob",
                side_effect=committed_blob,
            ),
            patch("aka.light.workflow.bootstrap.is_sol_op", return_value=False),
        )

    def test_known_v0_reference_wrapper_skips_production_review(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self._framework_baseline_campaign(directory)
            reviewer = patch.object(
                Campaign,
                "_production_kernel_violations",
                side_effect=AssertionError("known V0 wrapper must not be reviewed"),
            )
            patches = self._framework_baseline_decision_patches(
                directory,
                reference_blob="same-blob",
                implementation_blob="same-blob",
            )
            with ExitStack() as stack:
                stack.enter_context(reviewer)
                for current in patches:
                    stack.enter_context(current)
                action, reason = campaign._framework_baseline_decision()

        self.assertEqual(action, "run")
        self.assertIn("PyTorch reference wrapper", reason)

    def test_open_dsl_framework_baseline_runs_without_pinning_a_framework(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo=str(Path(directory) / "reference.py"),
                platform="cuda",
                framework="",
                specify_dsl=False,
                work_dir=directory,
                optimization_mode="leaderboard",
                framework_baseline="always",
            )

            def check_structure(_workspace, framework, *, specify_dsl):
                self.assertEqual(framework, "")
                self.assertFalse(specify_dsl)
                return []

            patches = self._framework_baseline_decision_patches(
                directory,
                reference_blob="same-blob",
                implementation_blob="same-blob",
                structure_side_effect=check_structure,
            )
            with ExitStack() as stack:
                for current in patches:
                    stack.enter_context(current)
                action, reason = campaign._framework_baseline_decision()

        self.assertEqual(action, "run")
        self.assertIn("Agent-selected supported GPU DSL", reason)

    def test_open_dsl_framework_baseline_acceptance_keeps_structure_check_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo=str(Path(directory) / "reference.py"),
                platform="cuda",
                framework="",
                specify_dsl=False,
                work_dir=directory,
                optimization_mode="leaderboard",
                framework_baseline="always",
            )
            campaign.workspace.mkdir(parents=True, exist_ok=True)
            (campaign.workspace / "WORKSPACE.md").write_bytes(b"contract")

            def check_structure(_workspace, framework, *, specify_dsl):
                self.assertEqual(framework, "")
                self.assertFalse(specify_dsl)
                return []

            with (
                patch.object(
                    Campaign,
                    "source_checks_enabled",
                    new_callable=PropertyMock,
                    return_value=True,
                ),
                patch(
                    "aka.light.workspace.sources.workspace_identity",
                    return_value=b"candidate",
                ),
                patch(
                    "aka.light.workflow.bootstrap.framework_baseline_workspace_contract",
                    return_value=b"contract",
                ),
                patch(
                    "aka.light.workflow.bootstrap.production_structure_violations",
                    side_effect=check_structure,
                ),
                patch("aka.light.workflow.bootstrap.git_path_blob", return_value=b""),
            ):
                problem = campaign._framework_baseline_problem(
                    "v0", "baseline", include_policy_review=False
                )

        self.assertEqual(problem, "")

    def test_open_dsl_production_check_forwards_neutral_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo=str(Path(directory) / "reference.py"),
                platform="cuda",
                framework="",
                specify_dsl=False,
                work_dir=directory,
                optimization_mode="leaderboard",
            )
            with (
                patch.object(
                    Campaign,
                    "source_checks_enabled",
                    new_callable=PropertyMock,
                    return_value=True,
                ),
                patch(
                    "aka.light.workflow.bootstrap.production_kernel_violations",
                    return_value=[],
                ) as check,
            ):
                self.assertEqual(campaign._production_kernel_violations(), [])

        self.assertFalse(check.call_args.kwargs["specify_dsl"])

    def test_distinct_v0_implementation_still_uses_production_review(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self._framework_baseline_campaign(directory)
            patches = self._framework_baseline_decision_patches(
                directory,
                reference_blob="reference-blob",
                implementation_blob="implementation-blob",
            )
            reviewer_patch = patch.object(
                    Campaign, "_production_kernel_violations", return_value=[]
                )
            with ExitStack() as stack:
                reviewer = stack.enter_context(reviewer_patch)
                for current in patches:
                    stack.enter_context(current)
                action, reason = campaign._framework_baseline_decision()

        self.assertEqual(action, "pin")
        self.assertIn("already a compliant framework implementation", reason)
        reviewer.assert_called_once_with()

    def test_real_git_v0_reference_wrapper_skips_production_review(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self._framework_baseline_campaign(directory)
            workspace = campaign.workspace
            (workspace / "implementation").mkdir(parents=True)
            source = "import torch\n\ndef run(*args, **kwargs):\n    return torch.zeros(1)\n"
            (workspace / "kernel.py").write_text(source, encoding="utf-8")
            (workspace / "reference.py").write_text(source, encoding="utf-8")
            (workspace / "implementation" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "config", "user.name", "AKA Test"], cwd=workspace, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "aka-test@atrex.local"],
                cwd=workspace,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=workspace, check=True)
            subprocess.run(
                ["git", "-c", "commit.gpgSign=false", "commit", "-qm", "V0"],
                cwd=workspace,
                check=True,
            )
            from aka.light.workspace.sources import from_git

            committed_source = from_git(workspace, "HEAD")

            with (
                patch("aka.light.workflow.bootstrap.latest_version", return_value=0),
                patch(
                    "aka.light.workflow.bootstrap.resolve_framework_baseline_commit",
                    return_value=(None, None),
                ),
                patch.object(
                    Campaign,
                    "source_checks_enabled",
                    new_callable=PropertyMock,
                    return_value=True,
                ),
                patch(
                    "aka.light.workflow.bootstrap.production_structure_violations",
                    return_value=[],
                ),
                patch(
                    "aka.light.workspace.sources.snapshot",
                    return_value=committed_source,
                ),
                patch("aka.light.workflow.bootstrap.is_sol_op", return_value=False),
                patch.object(
                    Campaign,
                    "_production_kernel_violations",
                    side_effect=AssertionError(
                        "known V0 wrapper must not be reviewed"
                    ),
                ),
            ):
                action, reason = campaign._framework_baseline_decision()

        self.assertEqual(action, "run")
        self.assertIn("PyTorch reference wrapper", reason)

    def test_unlimited_numerical_review_keeps_nested_request_timeout(self):
        runtime = SimpleNamespace(closed=False, _live=lambda _parent: True)
        cancel = NumericalCancellation(
            runtime, SimpleNamespace(deadline=float("inf"))
        )

        with patch(
            "aka.light.acceptance.numerical.time.monotonic", return_value=100.0
        ):
            self.assertEqual(cancel.timeout(600), 600)

    def test_finite_numerical_review_caps_nested_request_timeout(self):
        runtime = SimpleNamespace(closed=False, _live=lambda _parent: True)
        cancel = NumericalCancellation(runtime, SimpleNamespace(deadline=150.9))

        with patch(
            "aka.light.acceptance.numerical.time.monotonic", return_value=100.0
        ):
            self.assertEqual(cancel.timeout(600), 50)

    def test_all_supervisor_review_sessions_are_unbounded(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "kernel.py").write_text(
                "class Model: pass\n", encoding="utf-8"
            )
            evidence = workspace / "evidence.txt"
            evidence.write_text("immutable evidence\n", encoding="utf-8")
            files = {"trusted/evidence.txt": evidence}
            digest = numerical_review._digest(files)
            observed: list[tuple[str, int | None]] = []

            def numerical_session(review_workspace, _prompt, *, timeout, **_kwargs):
                observed.append(("numerical", timeout))
                (review_workspace / "numerical_review.json").write_text(
                    json.dumps({
                        "schema_version": 1,
                        "evidence_digest": digest,
                        "action": "complete",
                        "summary": "No supplemental probe is required.",
                        "suite": None,
                    }),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    exit_status=0,
                    timed_out=False,
                    session_id="numerical-review",
                    stdout_tail="",
                    stderr_tail="",
                )

            numerical_campaign = SimpleNamespace(
                workspace=workspace,
                agent_cli="claude",
                _run_session=numerical_session,
                _account=lambda *_args: None,
                agent_boundary_environment=lambda _role: {},
            )
            numerical_review._request_review(
                numerical_campaign,
                workspace,
                files,
                digest,
                state_root=workspace / "review-state",
                cancel=SimpleNamespace(
                    check=lambda: None,
                    is_set=lambda: False,
                    wait=lambda _seconds: False,
                ),
            )

            campaign = Campaign(
                name="operator",
                kernel_demo=str(workspace / "kernel.py"),
                platform="cuda",
                framework="cuda",
                work_dir=str(root),
                optimization_mode="production",
            )
            result = SimpleNamespace(
                exit_status=0,
                timed_out=False,
                session_id="policy-review",
                stdout_tail="",
                stderr_tail="review stopped",
                tokens=0,
            )

            def policy_session(*_args, **kwargs):
                observed.append(("policy", kwargs["timeout"]))
                return result

            with (
                patch.object(
                    campaign, "_run_session", side_effect=policy_session, create=True
                ),
                patch.object(campaign, "_account"),
                patch.object(
                    campaign, "agent_boundary_environment", return_value={}
                ),
            ):
                with self.assertRaises(ReviewerOutputUnavailable):
                    campaign._review_production_candidate_once(
                        workspace,
                        "cuda",
                        False,
                        candidate_digest="a" * 64,
                    )

            context = campaign.workspace / "README.md"
            campaign.workspace.mkdir(parents=True, exist_ok=True)
            context.write_text("operator contract\n", encoding="utf-8")

            def baseline_session(*_args, **kwargs):
                observed.append(("baseline", kwargs["timeout"]))
                return result

            with (
                patch.object(
                    campaign,
                    "_framework_baseline_correctness_context",
                    return_value=(context,),
                ),
                patch.object(
                    campaign,
                    "_run_session",
                    side_effect=baseline_session,
                    create=True,
                ),
                patch.object(
                    campaign, "agent_boundary_environment", return_value={}
                ),
            ):
                campaign._run_framework_baseline_correctness_reviewer("codex")

            self.assertEqual(
                observed,
                [
                    ("numerical", None),
                    ("policy", None),
                    ("baseline", None),
                ],
            )

    def test_one_runtime_step_timeout_drives_gateway_request(self):
        self.assertEqual(derived_evaluation_timeout(), 270)
        command = RuntimeConfig("L20D").command(Path("/tmp/workspace"))
        self.assertEqual(
            command[command.index("--evaluation-step-timeout") + 1], "60"
        )
        parser = build_parser()
        arguments = parser.parse_args(["--hardware", "L20D"])
        self.assertEqual(arguments.evaluation_step_timeout, 60)
        with patch(
            "aka.light.runtime.server.configured_queue_wait_grace",
            return_value=120,
        ):
            runtime = SupervisorRuntime(RuntimeConfig("L20D"))
        try:
            self.assertEqual(runtime.request_timeout_seconds, 1980)
        finally:
            runtime.close()

    def test_eval_timeout_scales_with_actual_shape_count(self):
        # Only the sequential performance phase repeats.  Fixed candidate,
        # correctness, queue and cleanup budgets remain singletons.
        self.assertEqual(shape_request_timeout(270, 5), 510)
        self.assertEqual(shape_request_timeout(270, 3), 390)
        self.assertEqual(effective_shape_batch_size(270, 5, 120), 5)
        self.assertEqual(effective_shape_batch_size(3000, 5, 120), 5)
        self.assertEqual(
            effective_shape_batch_size(
                3000, 5, 120, performance_timeout=3000
            ),
            3,
        )

    def test_eval_outer_batches_use_native_shape_ids_and_one_ray_shard(self):
        self.assertEqual(SHAPE_BATCH_SIZE, 10)
        self.assertEqual(CampaignRuntimeSettings().shape_batch_size, 10)
        request = {
            "reference": {
                "shapes": {str(index): {"x": index} for index in range(21)},
                "metadata": {
                    "num_shapes": 21,
                    "shapes": {str(index): {"weight": index} for index in range(21)},
                    "cases": {str(index): {"label": index} for index in range(21)},
                },
                "roofline": {
                    "shapes": {str(index): {"flops": index} for index in range(21)},
                },
            },
            "options": {"timeout_s": 600},
            "runner_overrides": {"shape_ids": [str(index) for index in range(21)]},
            "lock_clocks": True,
        }

        batches = batch_shape_ids(request_shape_ids(request), SHAPE_BATCH_SIZE)
        requests = [shape_batch_request(request, ids) for ids in batches]

        self.assertEqual([len(ids) for ids in batches], [10, 10, 1])
        full = [str(index) for index in range(21)]
        self.assertEqual(
            [list(item["reference"]["shapes"]) for item in requests],
            [full, full, full],
        )
        self.assertEqual(
            [list(item["reference"]["metadata"]["cases"]) for item in requests],
            [full, full, full],
        )
        self.assertEqual(
            [list(item["reference"]["roofline"]["shapes"]) for item in requests],
            [full, full, full],
        )
        self.assertEqual(
            [item["reference"]["metadata"]["num_shapes"] for item in requests],
            [21, 21, 21],
        )
        self.assertEqual([item["options"]["shape_ids"] for item in requests], batches)
        self.assertTrue(all("shape_ids" not in item["runner_overrides"]
                            for item in requests))
        self.assertTrue(all(item["options"]["shape_shards"] == 1 for item in requests))
        self.assertTrue(all(item["lock_clocks"] is True for item in requests))
        self.assertEqual(len(request["reference"]["shapes"]), 21)
        self.assertEqual(len(request["runner_overrides"]["shape_ids"]), 21)

    def test_eval_delegates_locked_clock_frequency_to_agate(self):
        defaults = measurement_defaults("L20N")

        self.assertEqual(defaults["bench_iters"], 100)
        self.assertEqual(defaults["runner_overrides"]["warmup_iters"], 20)
        self.assertEqual(BENCH_ITERS, 100)
        self.assertEqual(WARMUP_ITERS, 20)
        self.assertIs(defaults["lock_clocks"], True)
        self.assertNotIn("gpu_clock_mhz", defaults["runner_overrides"])
        self.assertNotIn(
            "clock_lock_fail_on_deviation", defaults["runner_overrides"]
        )
        self.assertIsNone(resolved_gpu_clock_mhz("NVIDIA L20N"))
        self.assertIsNone(
            resolved_gpu_clock_mhz("NVIDIA RTX PRO 5000 Blackwell")
        )
        self.assertIsNone(resolved_gpu_clock_mhz("l20n", 2400))
        self.assertNotIn(
            "gpu_clock_mhz",
            runner_overrides(gpu_clock_mhz=2400),
        )

    def test_eval_timeout_rejects_one_shape_above_gateway_limit(self):
        with self.assertRaisesRegex(ValueError, "One Shape exceeds"):
            effective_shape_batch_size(10681, 5, 120)

    def test_only_request_performance_work_scales_by_shape(self):
        self.assertEqual(
            _evaluation_performance_timeout(["python3", "test_kernel.py"], 60),
            60,
        )
        self.assertEqual(
            _evaluation_performance_timeout(
                ["python3", "test_kernel.py", "--version", "v0"], 60
            ),
            60,
        )
        self.assertEqual(
            _evaluation_performance_timeout(
                ["python3", "test_kernel.py", "--baseline-path", "v0.py"],
                60,
            ),
            240,
        )
        self.assertEqual(
            _evaluation_performance_timeout(
                [
                    "python3", "test_kernel.py", "--baseline-path", "v0.py",
                    "--comparison-repeats", "4",
                ],
                500,
            ),
            4000,
        )

    def test_eval_retries_candidate_wall_clock_timeout_once(self):
        payload = {
            "name": "timeout-retry",
            "spec": {"target_hardware": ["L20D"]},
            "candidate": "class Model: pass\n",
            "reference": {
                "operator": "example",
                "reference_py": "class Model: pass\n",
                "input_py": "def generate_input(*args, **kwargs): return ()\n",
                "shapes": {"0": {"x": 1}},
            },
            "options": {"num_correctness_cases": 1, "bench_iters": 1, "timeout_s": 600},
            "runner_overrides": {"shape_ids": ["0"]},
            "mode": "full",
            "lock_clocks": True,
            "env_vars": {},
            "__atrex_eval_archives": {
                "candidate": {"kernel.py": "class Model: pass\n"}
            },
        }
        timeout_job = subprocess.CompletedProcess(
            ["agate-sdk", "get"],
            1,
            json.dumps({
                "job_id": "ev_timeout",
                "status": "succeeded",
                "result": {
                    "performance": {
                        "shapes": {
                            "0": {
                                "error": "candidate exceeded 60.0s wall-clock timeout"
                            }
                        }
                    }
                },
            }),
            "",
        )
        final = subprocess.CompletedProcess(
            ["agate-sdk", "get"],
            1,
            json.dumps({
                "job_id": "ev_timeout_retry",
                "status": "succeeded",
                "result": {
                    "performance": {
                        "shapes": {
                            "0": {
                                "error": "candidate exceeded 60.0s wall-clock timeout"
                            }
                        }
                    }
                },
            }),
            "",
        )
        args = SimpleNamespace(
            hardware="L20D", timeout=600, env=[], profiler="auto",
            profile_level="basic", profile_counter=[], kernel_regex=None,
            top_kernels=None, kernel_name=None, profile_source=None,
            launch_skip=None, launch_count=None, profile_shape_id=None,
            requirement=[], deps_mode="auto", evaluation_input_path=None,
            evaluation_shapes_path=None, evaluation_mode=None, version=None,
            multi_seed=None, shape_id=[], timed_runs=None, shape_batch_size=5,
            evaluation_step_timeout=60,
            dry_run=False, url=None, gateway_profile="prod",
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("aka.light.evaluation.gateway._is_generalized_workspace", return_value=False),
            patch("aka.light.workspace.sources.policy_at", return_value=None),
            patch("aka.light.evaluation.gateway.build_typed_request", return_value=payload),
            patch(
                "aka.light.evaluation.source_transport.run",
                side_effect=[timeout_job, final],
            ) as sdk_run,
        ):
            result = _run_typed_gateway(
                args, Path(directory), ["python3", "test_kernel.py"],
                "run", [], 120,
            )

        self.assertEqual(result, 1)
        self.assertEqual(sdk_run.call_count, 2)
        self.assertNotIn("identity_extra", sdk_run.call_args_list[0].kwargs)
        self.assertEqual(
            sdk_run.call_args_list[1].kwargs["identity_extra"],
            {"eval_wall_clock_retry": 1},
        )
        self.assertEqual(
            embedded_candidate_wall_clock_timeouts(json.loads(final.stdout)),
            ["candidate exceeded 60.0s wall-clock timeout"],
        )

    @staticmethod
    def _deadline_job(*, max_deadline_retries, failures):
        submitted = []

        def submit(wire_key):
            submitted.append(wire_key)
            return subprocess.CompletedProcess(
                [], 0, json.dumps({"job_id": f"job-{len(submitted)}"}), ""
            )

        def poll(process, _remaining, *, recovery_probe=False):
            attempt = int(json.loads(process.stdout)["job_id"].split("-")[-1])
            failed = attempt <= failures
            job = {
                "job_id": f"job-{attempt}",
                "status": "failed" if failed else "succeeded",
            }
            if failed:
                job["error"] = {
                    "error_class": "timeout",
                    "reason": "timeout",
                    "message": "job deadline exceeded",
                }
            else:
                job["result"] = {"ok": True}
            return subprocess.CompletedProcess(
                [], 1 if failed else 0, json.dumps(job), ""
            )

        with patch("aka.light.execution.gateway_jobs.time.sleep"):
            process = execute_job(
                {"request": "deadline-test"},
                submit,
                poll,
                wait_budget=60,
                idempotent_submission=True,
                max_deadline_retries=max_deadline_retries,
            )
        return process, submitted

    @staticmethod
    def _infra_job(*, max_infra_retries, failures, status="failed"):
        submitted = []

        def submit(wire_key):
            submitted.append(wire_key)
            return subprocess.CompletedProcess(
                [], 0, json.dumps({"job_id": f"job-{len(submitted)}"}), ""
            )

        def poll(process, _remaining, *, recovery_probe=False):
            attempt = int(json.loads(process.stdout)["job_id"].split("-")[-1])
            failed = attempt <= failures
            job = {
                "job_id": f"job-{attempt}",
                "status": status if failed else "succeeded",
            }
            if failed:
                job["error"] = {
                    "error_class": "infra",
                    "reason": "backend_unavailable",
                    "message": "temporary infrastructure failure",
                }
            else:
                job["result"] = {"ok": True}
            return subprocess.CompletedProcess(
                [], 1 if failed else 0, json.dumps(job), ""
            )

        with patch("aka.light.execution.gateway_jobs.time.sleep"):
            process = execute_job(
                {"request": "infra-test", "limit": max_infra_retries},
                submit,
                poll,
                wait_budget=60,
                idempotent_submission=True,
                max_infra_retries=max_infra_retries,
            )
        return process, submitted

    def test_ordinary_gateway_deadline_stops_after_two_retries(self):
        process, submitted = self._deadline_job(
            max_deadline_retries=DEFAULT_MAX_DEADLINE_RETRIES,
            failures=3,
        )
        self.assertEqual(process.returncode, 1)
        self.assertEqual(len(submitted), 3)
        self.assertEqual(
            json.loads(process.stdout)["error"],
            {
                "error_class": "timeout",
                "reason": "timeout",
                "message": "job deadline exceeded",
            },
        )
        self.assertEqual(
            gateway_error_from_process(process, "evaluate")["error"],
            {
                "error_class": "timeout",
                "reason": "timeout",
                "message": "job deadline exceeded",
            },
        )

    def test_terminal_gateway_deadline_has_no_retry_count_cap(self):
        process, submitted = self._deadline_job(
            max_deadline_retries=None,
            failures=3,
        )
        self.assertEqual(process.returncode, 0)
        self.assertEqual(len(submitted), 4)

    def test_deadline_timeout_is_added_to_agent_visible_error(self):
        process, submitted = self._deadline_job(
            max_deadline_retries=DEFAULT_MAX_DEADLINE_RETRIES,
            failures=3,
        )
        annotated = annotate_deadline_timeout(process, 600)

        self.assertEqual(len(submitted), 3)
        self.assertEqual(json.loads(annotated.stdout)["error"]["timeout_s"], 600)
        projected = gateway_error_from_process(annotated, "dev")
        self.assertEqual(projected["operation"], "dev")
        self.assertEqual(projected["error"]["timeout_s"], 600)
        self.assertIn("configured 600-second deadline", projected["next_action"])
        self.assertIn("command is too slow", projected["next_action"])
        self.assertIn("Do not resubmit the unchanged", projected["next_action"])
        self.assertEqual(
            projected["recovery"],
            [{"instruction": projected["next_action"]}],
        )

    def test_collection_timeout_returns_immediately_without_job_id_or_resubmission(self):
        submitted = []
        polled = []

        def submit():
            submitted.append("submit")
            return subprocess.CompletedProcess(
                [], 0, json.dumps({"job_id": "ev_private", "status": "queued"}), ""
            )

        def poll(process, _remaining, *, recovery_probe=False):
            polled.append(recovery_probe)
            return subprocess.CompletedProcess(
                [], 0, json.dumps({"job_id": "ev_private", "status": "running"}), ""
            )

        process = execute_job(
            {"request": "collection-timeout"},
            submit,
            poll,
            wait_budget=37,
        )

        self.assertEqual(submitted, ["submit"])
        self.assertEqual(polled, [False])
        self.assertEqual(process.returncode, 75)
        payload = json.loads(process.stdout)
        self.assertEqual(payload["status"], "timed_out")
        self.assertEqual(payload["outcome"], "timeout")
        self.assertEqual(payload["error"]["reason"], "gateway_collection_timeout")
        self.assertEqual(payload["error"]["timeout_s"], 37)
        self.assertNotIn("job_id", payload)
        self.assertNotIn("ev_private", process.stderr)

        projected = gateway_error_from_process(process, "check")
        self.assertNotIn("job_id", projected)
        self.assertIn("not a Kernel correctness or performance verdict", projected["next_action"])
        self.assertIn("current Runtime call has ended", projected["next_action"])

    def test_collection_transport_timeout_keeps_private_checkpoint_for_later_probe(self):
        submitted = []
        polls = []

        def submit():
            submitted.append("submit")
            return subprocess.CompletedProcess(
                [], 0, json.dumps({"job_id": "ev_private", "status": "queued"}), ""
            )

        def first_poll(_process, _remaining, *, recovery_probe=False):
            polls.append(("first", recovery_probe))
            raise EvidenceUnavailable("Gateway collection transport unavailable")

        def resumed_poll(_process, _remaining, *, recovery_probe=False):
            polls.append(("resumed", recovery_probe))
            return subprocess.CompletedProcess(
                [], 0, json.dumps({
                    "job_id": "ev_private",
                    "status": "succeeded",
                    "result": {"ok": True},
                }), ""
            )

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {JOB_ROOT_ENV: str(Path(directory).resolve())}
        ):
            timed_out = execute_job(
                {"request": "recover-private-job"},
                submit,
                first_poll,
                wait_budget=37,
            )
            recovered = execute_job(
                {"request": "recover-private-job"},
                submit,
                resumed_poll,
                wait_budget=37,
            )

            state_path = next(Path(directory).glob("jobs/*/state.json"))
            state = json.loads(state_path.read_text())

        self.assertEqual(json.loads(timed_out.stdout)["error"]["reason"], "gateway_collection_timeout")
        self.assertNotIn("job_id", json.loads(timed_out.stdout))
        self.assertEqual(submitted, ["submit"])
        self.assertEqual(polls, [("first", False), ("resumed", True)])
        self.assertEqual(json.loads(recovered.stdout)["status"], "succeeded")
        self.assertEqual(state["phase"], "terminal")

    def test_sdk_collection_short_polls_same_job_until_valid_terminal(self):
        from aka.light.evaluation.source_transport import run_sdk_job

        class GatewayError(Exception):
            def __init__(self, status=None):
                self.status = status

        class TransportError(GatewayError):
            pass

        class Client:
            def __init__(self):
                self.submissions = 0
                self.polls = []

            def submit_job(self, _kind, _request):
                self.submissions += 1
                return {"job_id": "ev_sdk", "status": "queued"}

            def get_job(self, job_id, *, wait=False, timeout=30):
                self.polls.append((job_id, wait, timeout))
                if len(self.polls) == 1:
                    raise TransportError()
                if len(self.polls) == 2:
                    return {"job_id": job_id, "status": "running"}
                return {"job_id": job_id, "status": "succeeded", "result": {}}

            def cancel_job(self, job_id):
                raise AssertionError(f"unexpected cancellation of {job_id}")

        client = Client()
        args = SimpleNamespace(
            url="https://gateway.invalid", gateway_profile=None, timeout=60
        )
        payload = {"spec": {"target_hardware": ["L20D"]}}
        with (
            patch(
                "aka.light.evaluation.source_transport._load_sdk_client",
                return_value=(
                    client,
                    "https://gateway.invalid",
                    GatewayError,
                    TransportError,
                ),
            ),
            patch("aka.light.evaluation.source_transport.time.sleep") as sleep,
            patch.dict(os.environ, {}, clear=True),
        ):
            process = run_sdk_job(None, args, payload, "dev", 0)

        self.assertEqual(process.returncode, 0)
        self.assertEqual(client.submissions, 1)
        self.assertEqual([item[0] for item in client.polls], ["ev_sdk"] * 3)
        self.assertTrue(all(item[1] is False for item in client.polls))
        self.assertTrue(all(1 <= item[2] <= 30 for item in client.polls))
        self.assertEqual(sleep.call_count, 2)

    def test_light_has_no_agate_cli_or_raw_gateway_transport(self):
        from aka.light.evaluation import agate_native, gateway

        for legacy_name in (
            "find_agate",
            "build_typed_agate_command",
            "run_agate_with_cancel_retry",
            "run_direct_job",
            "_run_direct_gateway",
            "_gateway_json",
            "_resume_interrupted_agate_wait",
        ):
            with self.subTest(legacy_name=legacy_name):
                self.assertFalse(hasattr(gateway, legacy_name))
        self.assertFalse(hasattr(agate_native, "cli_arguments"))
        self.assertTrue(hasattr(gateway, "_run_sdk_dev"))


    def test_only_terminal_eval_selects_unbounded_deadline_retries(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(TERMINAL_VALIDATION_ENV, None)
            self.assertEqual(
                configured_max_deadline_retries(terminal_evaluation=True),
                DEFAULT_MAX_DEADLINE_RETRIES,
            )
        with patch.dict(os.environ, {TERMINAL_VALIDATION_ENV: "1"}):
            self.assertIsNone(
                configured_max_deadline_retries(terminal_evaluation=True)
            )
            self.assertEqual(
                configured_max_deadline_retries(terminal_evaluation=False),
                DEFAULT_MAX_DEADLINE_RETRIES,
            )

    def test_exploration_infrastructure_retry_remains_bounded(self):
        process, submitted = self._infra_job(
            max_infra_retries=DEFAULT_MAX_INFRA_RETRIES,
            failures=3,
        )
        self.assertEqual(process.returncode, 1)
        self.assertEqual(len(submitted), 3)

    def test_terminal_infrastructure_retry_has_no_count_cap(self):
        process, submitted = self._infra_job(
            max_infra_retries=None,
            failures=3,
        )
        self.assertEqual(process.returncode, 0)
        self.assertEqual(len(submitted), 4)

    def test_terminal_infrastructure_policy_is_selected_by_private_environment(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(TERMINAL_VALIDATION_ENV, None)
            self.assertEqual(
                configured_max_infra_retries(), DEFAULT_MAX_INFRA_RETRIES
            )
        with patch.dict(os.environ, {TERMINAL_VALIDATION_ENV: "1"}):
            self.assertIsNone(configured_max_infra_retries())

    def test_infrastructure_cancel_and_failure_origin_are_retryable(self):
        self.assertEqual(
            retry_kind({
                "status": "cancelled",
                "error": {"error_class": "infra", "message": "worker lost"},
            }),
            "infra",
        )
        self.assertEqual(
            retry_kind({
                "status": "failed",
                "failure_origin": "infrastructure",
                "error": {"message": "ray node exited"},
            }),
            "infra",
        )
        self.assertEqual(
            retry_kind({
                "status": "failed",
                "error": {
                    "message": "job supervisor exited",
                    "details": {"failure_origin": "infrastructure"},
                },
            }),
            "infra",
        )

    def test_episode_report_candidate_eval_is_terminal_validation(self):
        runtime = SupervisorRuntime.__new__(SupervisorRuntime)
        runtime.config = SimpleNamespace(
            atrex_bench_root=Path("/private/atrex-bench"),
            optimization_mode="production",
        )
        runtime.numerical_validator = None
        policy_calls = []
        runtime.production_policy_validator = lambda workspace, source, **options: (
            policy_calls.append((workspace, source, options)) or []
        )
        runtime.measurements = SimpleNamespace(read=lambda _: {})
        runtime.journal_factory = SimpleNamespace(
            research_records=False,
            experiments_enabled=False,
        )
        runtime._live = lambda _: True
        captured = {}

        def validate_candidate(context, source):
            context.measure(["--kind", "run"], source, True)
            self.assertEqual(context.production_policy(source), [])
            return {"status": "ok"}

        runtime.acceptance = SimpleNamespace(validate_candidate=validate_candidate)
        runtime.execute = lambda *args, **kwargs: captured.update(kwargs) or {
            "exit_code": 0, "stdout": "", "stderr": "",
        }
        capability = SimpleNamespace(
            journal=SimpleNamespace(
                service=SimpleNamespace(git_workspace=Path("/candidate")),
                require_gluon=True,
            )
        )
        self.assertEqual(
            runtime._validate_candidate_correctness(capability, b"candidate"),
            {"status": "ok"},
        )
        self.assertIs(captured["terminal_validation"], True)
        self.assertEqual(policy_calls[0][0], Path("/candidate"))
        self.assertEqual(policy_calls[0][1], b"candidate")
        self.assertIs(policy_calls[0][2]["parent"], capability)
        self.assertIs(policy_calls[0][2]["require_gluon"], True)

    def test_production_review_rejection_is_returned_by_episode_report(self):
        source = b"candidate"
        numerical_calls = []
        context = CandidateValidation(
            atrex_bench=True,
            production=True,
            measure=lambda *_args, **_kwargs: {
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            },
            read_record=lambda _record_id: {},
            ensure_live=lambda: None,
            production_policy=lambda _source: [
                "production candidate rejected by independent reviewer: "
                "compute_provenance: PyTorch performs the core computation"
            ],
            supplemental=lambda _source, _correctness_passed: (
                numerical_calls.append(True) or ""
            ),
        )
        result = {
            "all_pass": True,
            "gateway_record_id": "gateway-" + "a" * 32,
        }
        with (
            patch(
                "aka.light.evaluation.provider.result_from_response",
                return_value=result,
            ),
            patch(
                "aka.light.acceptance.provider.authoritative_candidate_record",
                return_value=True,
            ),
            self.assertRaises(AgentRequestError) as raised,
        ):
            RecordedAcceptance().validate_candidate(context, source)

        response = raised.exception.response
        self.assertEqual(
            response["error"]["code"],
            "candidate_production_policy_failed",
        )
        self.assertIn("PyTorch performs", response["error"]["violations"][0])
        self.assertIn("resubmit episode-report", response["error"]["next_action"])
        self.assertEqual(numerical_calls, [True])
        self.assertEqual(response["checks"]["correctness"]["status"], "passed")
        self.assertEqual(response["checks"]["production_review"]["status"], "failed")
        self.assertEqual(response["checks"]["numerical"]["status"], "passed")

    def test_production_acceptance_gates_start_concurrently(self):
        acceptance = RecordedAcceptance()
        barrier = threading.Barrier(3)

        def gate(*_args):
            barrier.wait(timeout=2)
            return {"status": "passed"}

        context = CandidateValidation(
            atrex_bench=True,
            production=True,
            measure=lambda *_args, **_kwargs: {},
            read_record=lambda _record_id: {},
            ensure_live=lambda: None,
        )
        with (
            patch.object(acceptance, "_validate_correctness", side_effect=gate),
            patch.object(acceptance, "_validate_policy", side_effect=gate),
            patch.object(acceptance, "_validate_numerical", side_effect=gate),
        ):
            result = acceptance.validate_candidate(context, b"candidate")
        self.assertEqual(result["checks"]["correctness"]["status"], "passed")
        self.assertEqual(result["checks"]["production_review"]["status"], "passed")
        self.assertEqual(result["checks"]["numerical"]["status"], "passed")

    def test_multiple_acceptance_failures_are_returned_together(self):
        acceptance = RecordedAcceptance()
        context = CandidateValidation(
            atrex_bench=True,
            production=True,
            measure=lambda *_args, **_kwargs: {},
            read_record=lambda _record_id: {},
            ensure_live=lambda: None,
        )
        with (
            patch.object(
                acceptance,
                "_validate_correctness",
                side_effect=AgentRequestError(
                    "correctness failed",
                    code="candidate_correctness_failed",
                    next_action="Repair correctness.",
                ),
            ),
            patch.object(
                acceptance,
                "_validate_policy",
                side_effect=AgentRequestError(
                    "policy failed",
                    code="candidate_production_policy_failed",
                    next_action="Repair final source.",
                ),
            ),
            patch.object(
                acceptance,
                "_validate_numerical",
                return_value={"status": "passed"},
            ),
            self.assertRaises(AgentRequestError) as raised,
        ):
            acceptance.validate_candidate(context, b"candidate")

        response = raised.exception.response
        self.assertEqual(response["error"]["code"], "candidate_acceptance_failed")
        self.assertEqual(
            response["error"]["failed_checks"],
            ["correctness", "production_review"],
        )
        self.assertEqual(response["error"]["checks"]["numerical"]["status"], "passed")

    def test_output_limit_recovery_defaults_to_five_continuations(self):
        self.assertEqual(WorkflowOptions().output_limit_recovery_retries, 5)
        self.assertEqual(WORKFLOW_PLUGIN_DEFAULTS["output_limit_recovery_retries"], 5)
        campaign = SimpleNamespace(module_options=SimpleNamespace())
        self.assertEqual(
            session_execution_policy(campaign)["output_limit_recovery_retries"],
            5,
        )

    def test_island_search_is_the_default_and_can_be_disabled_explicitly(self):
        defaults = WorkflowOptions().search
        self.assertIsInstance(defaults, SearchOptions)
        self.assertEqual(defaults.trajectories, 3)
        self.assertEqual(defaults.max_parallel_episodes, 3)
        self.assertEqual(defaults.episodes_per_trajectory, 1)
        self.assertEqual(defaults.synchronization, "synchronous")
        self.assertFalse(defaults.role_specialization)
        self.assertEqual(WORKFLOW_PLUGIN_DEFAULTS["search"], {
            "epochs": 10,
            "trajectories": 3,
            "episodes_per_trajectory": 1,
            "max_parallel_episodes": 3,
            "synchronization": "synchronous",
            "role_specialization": False,
        })
        self.assertIsNone(WorkflowOptions(search={}).search)

    def test_full_and_shape_selected_evaluates_use_five_cases(self):
        self.assertEqual(
            evaluation_policy(["python3", "test_kernel.py"])["num_correctness_cases"],
            5,
        )
        self.assertEqual(
            evaluation_policy(
                ["python3", "test_kernel.py", "--shape-id", "6"],
                "correctness_only",
            ),
            {
                "schema_version": 1,
                "num_correctness_cases": 5,
                "mode": "correctness_only",
            },
        )

    def test_agent_cannot_override_correctness_case_count(self):
        with self.assertRaisesRegex(ValueError, "Runtime-owned"):
            evaluation_policy(
                ["python3", "test_kernel.py", "--multi-seed", "0"]
            )
        parser = build_parser()
        schema = gateway_schema("run", parser)
        self.assertNotIn("--multi-seed", schema["cli_options"])
        self.assertNotIn("--evaluation-step-timeout", schema["cli_options"])
        self.assertNotIn("--shape-batch-size", schema["cli_options"])
        self.assertIn("--comparison-run-timeout", schema["cli_options"])
        arguments = parser.parse_args(["--kind", "run", "--multi-seed", "0"])
        with self.assertRaisesRegex(ValueError, "Runtime-owned"):
            validate_evaluation_options(arguments)
        help_text = " ".join(parser.format_help().split())
        self.assertIn("Evaluate always uses 5 total correctness cases", help_text)
        self.assertNotIn("Additional correctness seeds (Atrex-Bench default: 5", help_text)

    def test_abba_implicit_run_timeout_fits_dynamic_schedule(self):
        arguments = SimpleNamespace(
            timeout=1230,
            comparison_repeats=8,
            evaluation_step_timeout=300,
            comparison_run_timeout=None,
        )
        self.assertEqual(comparison_run_timeout(arguments), 75)
        self.assertEqual(comparison_allocation_timeout(arguments), 1230)

        arguments.comparison_run_timeout = 240
        self.assertEqual(comparison_run_timeout(arguments), 240)
        self.assertEqual(comparison_allocation_timeout(arguments), 3870)

        arguments.comparison_run_timeout = 700
        with self.assertRaisesRegex(
            ValueError,
            r"700 seconds.*8.*11230-second ABBA allocation.*10800-second.*"
            r"did not clamp.*no Job was submitted",
        ):
            comparison_allocation_timeout(arguments)

        help_text = " ".join(build_parser().format_help().split())
        self.assertIn(
            "An explicit positive value is used unchanged and the ABBA allocation expands to "
            "fit it.",
            help_text,
        )
        self.assertIn("Runtime never clamps the supplied value", help_text)
        self.assertNotIn("Default: 60, matching Performance timeout", help_text)

    def test_managed_env_uses_campaign_hardware_only(self):
        parser = build_parser()
        schema = gateway_schema("env", parser)
        self.assertEqual(
            set(schema["cli_options"]),
            {"--kind", "--env-force", "--force", "--frameworks", "--counters", "--limits"},
        )
        self.assertNotIn("--env-gpu", schema["cli_options"])
        self.assertNotIn("--env-capabilities", schema["cli_options"])
        parsed = parse_gateway(["--kind", "env"])
        argv = managed_gateway_argv(["--kind", "env"], parsed, "L20D")
        self.assertEqual(
            argv,
            ["--kind", "env", "--env-gpu", "L20D", "--env-capabilities"],
        )
        effective = parse_gateway(argv)
        self.assertEqual(effective.env_gpu, "L20D")
        self.assertTrue(effective.env_capabilities)
        with self.assertRaisesRegex(ValueError, "Supervisor-owned"):
            filter_options(
                ["--kind", "env", "--env-gpu", "B200"],
                frozenset(),
                frozenset({"--env-gpu"}),
            )

    def test_managed_help_exposes_only_agent_controlled_options(self):
        response = managed_gateway_help(["--kind", "env", "--help"])
        self.assertIsNotNone(response)
        payload = json.loads(response["stdout"])
        schema = payload["managed_runtime_request_schema"]
        self.assertEqual(
            set(schema["cli_options"]),
            {"--kind", "--env-force", "--force", "--frameworks", "--counters", "--limits"},
        )
        self.assertNotIn("--hardware", response["stdout"])
        self.assertNotIn("--env-gpu", response["stdout"])
        self.assertIsNone(managed_gateway_help(["--kind", "env"]))

    def test_abba_names_the_git_commit_identity_explicitly(self):
        parser = build_parser()
        parsed = parser.parse_args([
            "--kind", "run",
            "--baseline-commit", "a" * 40,
        ])
        self.assertEqual(parsed.baseline_commit, "a" * 40)
        schema = gateway_schema("run", parser)
        self.assertIn("--baseline-commit", schema["cli_options"])
        self.assertNotIn("--baseline-path", schema["cli_options"])
        for retired in ("record-read", "kernel-read", "kernel-records"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--kind", retired])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--kind", "kernel-pareto-frontier"])

    def test_trusted_execution_can_use_a_separate_episode_journal_workspace(self):
        runtime = SupervisorRuntime.__new__(SupervisorRuntime)
        runtime.lock = threading.RLock()
        runtime.closed = False
        runtime.capabilities = {}
        runtime.journals = {}
        runtime.config = SimpleNamespace(request_timeout=30)
        runtime.request_lock = lambda *_args: ExitStack()
        runtime.revoke = lambda token: runtime.capabilities.pop(token, None)
        captured = {}

        def execute(capability, route, request, **kwargs):
            captured.update(
                capability=capability,
                route=route,
                request=request,
                kwargs=kwargs,
            )
            return {"exit_code": 0, "stdout": "", "stderr": ""}

        runtime.execute = execute
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execution_workspace = root / "staged"
            journal_workspace = root / "episode"
            execution_workspace.mkdir()
            journal_workspace.mkdir()
            binding = SimpleNamespace(view=SimpleNamespace())
            runtime.journals[journal_workspace.resolve()] = binding

            response = runtime.execute_trusted(
                execution_workspace,
                ["--kind", "run"],
                journal_workspace=journal_workspace,
                kernel_ids={"candidate": "a" * 40, "baseline": "b" * 40},
            )

        self.assertEqual(response["exit_code"], 0)
        self.assertEqual(captured["capability"].workspace, execution_workspace.resolve())
        self.assertIs(captured["capability"].journal, binding)
        self.assertEqual(captured["route"], "/v1/gateway/execute")
        self.assertEqual(
            captured["kwargs"]["trusted_kernel_ids"],
            {"candidate": "a" * 40, "baseline": "b" * 40},
        )
        self.assertEqual(runtime.capabilities, {})

    def test_trusted_execution_rejects_an_unregistered_journal_workspace(self):
        runtime = SupervisorRuntime.__new__(SupervisorRuntime)
        runtime.lock = threading.RLock()
        runtime.closed = False
        runtime.capabilities = {}
        runtime.journals = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execution_workspace = root / "staged"
            journal_workspace = root / "episode"
            execution_workspace.mkdir()
            journal_workspace.mkdir()
            with self.assertRaisesRegex(RuntimeError, "not registered with Agent Git"):
                runtime.execute_trusted(
                    execution_workspace,
                    ["--kind", "run"],
                    journal_workspace=journal_workspace,
                )
        self.assertEqual(runtime.capabilities, {})

    def test_recorded_abba_keeps_execution_and_journal_workspaces_separate(self):
        captured = {}

        def execute(workspace, _argv, **kwargs):
            captured["execution_workspace"] = workspace
            captured.update(kwargs)
            raise RuntimeError("stop after observing trusted request")

        verifier = RecordedABBAValidator(execute=execute, hardware="L20D")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "config", "user.name", "AKA Test"], cwd=workspace, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "aka-test@atrex.local"],
                cwd=workspace,
                check=True,
            )
            (workspace / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "-c", "commit.gpgSign=false", "commit", "-qm", "baseline"],
                cwd=workspace,
                check=True,
            )
            baseline = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
            ).strip()
            (workspace / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "-c", "commit.gpgSign=false", "commit", "-qm", "candidate"],
                cwd=workspace,
                check=True,
            )
            candidate = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
            ).strip()

            result = verifier.verify(
                workspace,
                base_commit=baseline,
                candidate_commit=candidate,
                changed_paths=["kernel.py"],
                candidate_kernel_id=candidate,
                baseline_kernel_id=baseline,
            )

            self.assertEqual(result.gate, "ERROR")
            self.assertEqual(captured["journal_workspace"], workspace)
            self.assertNotEqual(captured["execution_workspace"], workspace)
            self.assertEqual(
                captured["kernel_ids"],
                {"candidate": candidate, "baseline": baseline},
            )

    def test_command_round_trips_agent_workspace_through_scratch(self):
        parser = build_parser()
        args = parser.parse_args([
            "--kind", "command", "--", "python3", "scratch/probe.py",
        ])
        self.assertEqual(_requested_gateway_kind(args.kind, args.command), "dev")
        self.assertEqual(requested_sync_paths(args), ("scratch",))
        schema = gateway_schema("command", parser)
        self.assertEqual(schema["cli_options"]["--kind"], {"const": "command"})
        self.assertIn("Agent-visible workspace", schema["command"])

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary)
            (workspace / "scratch").mkdir()
            (workspace / "tools").mkdir()
            (workspace / "scratch/probe.py").write_text("print('probe')\n")
            (workspace / "tools/helper.py").write_text("VALUE = 1\n")
            (workspace / "README.md").write_text("# task\n")
            (workspace / "private.json").write_text("{}\n")
            (workspace / COMMAND_WORKSPACE_MANIFEST).write_text(json.dumps([
                "README.md", "scratch/probe.py", "tools/helper.py",
            ]))

            selected = _command_input_paths(
                workspace,
                args.command,
                full_workspace=True,
            )
            self.assertIn("README.md", selected)
            self.assertIn("scratch/probe.py", selected)
            self.assertIn("tools/helper.py", selected)
            self.assertNotIn("private.json", selected)
            self.assertNotIn(COMMAND_WORKSPACE_MANIFEST, selected)

            request, frozen = measurement_inputs(
                args,
                workspace,
                {},
                identity_key=b"command-test-key",
            )
            self.assertEqual(request["operation"], "dev")
            self.assertEqual(request["options"]["workspace_mode"], "full_roundtrip")
            self.assertIn("scratch/probe.py", request["inputs"])
            self.assertIn("tools/helper.py", request["inputs"])
            self.assertIn("README.md", frozen)
            self.assertNotIn("private.json", request["inputs"])

            encoded, _, _ = _make_input_bundle(
                workspace,
                16 * 1024 * 1024,
                selected,
                include_documentation=True,
            )
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)), mode="r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("README.md", names)
            self.assertIn("scratch/probe.py", names)
            self.assertIn("tools/helper.py", names)
            self.assertNotIn("private.json", names)

        args = parser.parse_args([
            "--kind", "command", "--sync", "profiles/probe", "--", "true",
        ])
        self.assertEqual(requested_sync_paths(args), ("scratch", "profiles/probe"))
        args = parser.parse_args([
            "--kind", "command", "--no-sync", "--", "true",
        ])
        self.assertEqual(requested_sync_paths(args), ())

    def test_gateway_request_schemas_describe_each_command_route_accurately(self):
        parser = build_parser()
        run = gateway_schema("run", parser)["command"]
        profile = gateway_schema("profile", parser)["command"]
        dev = gateway_schema("dev", parser)["command"]
        command = gateway_schema("command", parser)["command"]

        self.assertIn("test_kernel.py", run)
        self.assertNotIn("profile_driver.py", run)
        self.assertIn("profile_driver.py", profile)
        self.assertNotIn("test_kernel.py", profile)
        self.assertIn("through stateless Dev", dev)
        self.assertIn("For a custom probe", dev)
        self.assertIn("not official Evaluate evidence", dev)
        self.assertIn("--input", dev)
        self.assertIn("Required argv", command)
        self.assertIn("synchronizes scratch/ back", command)
        with self.assertRaisesRegex(ValueError, "requires argv after --"):
            parse_gateway(["--kind", "command"])
        self.assertEqual(
            parse_gateway(["--kind", "command", "--", "true"]).command,
            ["--", "true"],
        )

    def test_direction_only_schema_hides_experiment_association_fields(self):
        from aka.light.runtime.request_feedback import journal_schema

        proposal = journal_schema(
            "direction_update",
            {"action": "propose"},
            research_records=True,
            directions_enabled=True,
            experiments_enabled=False,
        )
        self.assertNotIn(
            "derived_from_experiment_ids", proposal["properties"]
        )
        closure = journal_schema(
            "direction_update",
            {"action": "complete"},
            research_records=True,
            directions_enabled=True,
            experiments_enabled=False,
        )
        self.assertNotIn(
            "supporting_experiment_ids", closure["properties"]
        )

    def test_supervisor_gates_do_not_expose_multi_seed(self):
        framework = framework_baseline_validation_command(1, atrex_bench=True)
        candidate = candidate_acceptance_argv(atrex_bench=True)
        verifier = RecordedABBAValidator(execute=lambda *_: None, hardware="L20D")
        self.assertNotIn("--multi-seed", framework)
        self.assertNotIn("--multi-seed", candidate)
        self.assertNotIn("--multi-seed", verifier.request_argv("a" * 40))

    def test_abba_prompt_separates_research_baseline_from_terminal_incumbent(self):
        verifier = RecordedABBAValidator(execute=lambda *_: None, hardware="L20D")
        prompt = verifier.agent_instructions(measurement_repetitions=1)
        self.assertIn("--baseline-commit <FULL_GIT_COMMIT>", prompt)
        self.assertIn("you choose its baseline", prompt)
        self.assertIn("terminal baseline is not Agent-selectable", prompt)
        self.assertIn("Supervisor automatically compares", prompt)
        self.assertNotIn("run this exact command", prompt)

    def test_supervisor_state_migrates_promoted_agent_kernel_identity(self):
        from aka.light.workflow.models import SupervisorState

        promoted = "a" * 40
        state = SupervisorState.from_dict({
            "attempts": [
                {"accepted": False, "candidate_kernel_commit_id": "b" * 40},
                {"accepted": True, "candidate_kernel_commit_id": promoted},
            ],
        })
        self.assertEqual(state.incumbent_kernel_commit_id, promoted)

    def test_framework_baseline_uses_episode_evaluator_policy(self):
        native = framework_baseline_validation_command(1, atrex_bench=True)
        sol = framework_baseline_validation_command(1, atrex_bench=False)
        self.assertEqual(
            native[native.index("--timed-runs") + 1], str(BENCH_ITERS)
        )
        self.assertEqual(
            sol[sol.index("--multi-seed") + 1], str(CORRECTNESS_CASES - 1)
        )

    def test_framework_baseline_and_episode_share_session_execution_policy(self):
        campaign = SimpleNamespace(
            module_options=SimpleNamespace(
                episode_timeout_seconds=None,
                output_limit_recovery_retries=2,
                dependency_policy_recovery_retries=2,
            )
        )
        self.assertEqual(
            session_execution_policy(campaign),
            {
                "episode_timeout_seconds": None,
                "reasoning_effort": "max",
                "output_limit_recovery_retries": 2,
                "dependency_policy_recovery_retries": 2,
            },
        )

    def test_framework_baseline_prompt_delegates_runtime_contract_to_claude_and_skills(self):
        template = (
            Path(__file__).resolve().parents[1]
            / "src/aka/light/workflow/prompts/framework_baseline.md"
        ).read_text(encoding="utf-8")
        for removed in (
            "{{RUNTIME_WAIT_POLICY}}",
            "{{AGENT_RUNTIME}}",
            "{{SANDBOX}}",
            "{{EVALUATOR}}",
        ):
            self.assertNotIn(removed, template)

        bootstrap_source = (
            Path(__file__).resolve().parents[1]
            / "src/aka/light/workflow/bootstrap.py"
        ).read_text(encoding="utf-8")
        self.assertIn("by WORKSPACE.md and the active SourceProvider", bootstrap_source)
        self.assertNotIn("Finish the preserved V1 implementation inside kernel.py", bootstrap_source)
        self.assertIn("run or repeat the configured authoritative gates", bootstrap_source)
        self.assertNotIn("already run the authoritative full-workload gates", bootstrap_source)

    def test_framework_baseline_workspace_contract_uses_active_sourceprovider(self):
        from aka.light.workspace.sources import default_tree, install, pack

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            policy, files = default_tree(
                {"kernel.py": b"def run(*args):\n    return args\n"}
            )
            install(workspace, pack(files, policy), bind=True)

            contract = framework_baseline_workspace_contract(workspace).decode()

        self.assertIn("# Framework Baseline workspace contract", contract)
        self.assertIn("Editable kernel.py and optional impl/ form one Kernel", contract)
        self.assertIn("same Runtime measurement operations", contract)
        self.assertIn("Use local Git for candidate checkpoints", contract)
        self.assertIn("ordinary episode-report contract", contract)
        self.assertIn("without ABBA", contract)
        self.assertNotIn("trajectory-private local Git repository", contract)

    def test_dependency_recovery_preserves_sourceprovider_and_mode_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = dependency_policy_recovery_prompt(
                Path(directory),
                "Continue the task.",
                "",
                "session",
                "blocked command",
                1,
                2,
                stage_name="Episode",
            )
        self.assertIn("existing candidate source", prompt)
        self.assertIn("current optimization-mode policy", prompt)
        self.assertIn("candidate_source_snapshot_sha256", prompt)
        self.assertIn('"reason": "local_execution_boundary_violation"', prompt)
        self.assertNotIn('"reason": "dependency_policy_violation"', prompt)
        self.assertNotIn("Use only provisioned dependencies", prompt)

    def test_light_policy_defers_correctness_rules_to_task_configuration(self):
        directive = optimization_mode_directive("production", "CUDA")
        self.assertNotIn("relative L2 error <= 0.2 for NVFP4", directive)
        self.assertNotIn("elementwise atol/rtol", directive)
        self.assertNotIn("mismatch-fraction", directive)
        self.assertIn("read workspace `metadata.json`", directive)
        self.assertIn("benchmark correctness contract is authoritative", directive)
        self.assertIn("Do not assume, invent, or override", directive)
        self.assertIn("one passing check never overrides another reported failure", directive)

    def test_light_policy_describes_disabled_numerical_validation_truthfully(self):
        self.assertFalse(WorkflowOptions().numerical_validation)
        self.assertFalse(WORKFLOW_PLUGIN_DEFAULTS["numerical_validation"])
        directive = optimization_mode_directive("production", "CUDA")
        self.assertNotIn("Numerical review produces targeted", directive)
        self.assertIn("Supplemental numerical review is disabled", directive)
        open_dsl = optimization_mode_directive("production", "", specify_dsl=False)
        self.assertIn("Supplemental numerical review is disabled", open_dsl)

        enabled = optimization_mode_directive(
            "production", "CUDA", numerical_validation_enabled=True
        )
        self.assertIn("Numerical review produces targeted", enabled)

    def test_pinned_production_dsl_only_constrains_submitted_candidate(self):
        cuda = optimization_mode_directive("production", "Cuda")
        self.assertIn(
            "submitted candidate's implementation framework is exactly **Cuda**",
            cuda,
        )
        self.assertIn("During development, you may use any language or DSL", cuda)
        self.assertIn("PyTorch/ATen compute", cuda)
        self.assertIn("third-party compute libraries", cuda)
        self.assertIn("dynamically loaded external implementations", cuda)
        self.assertIn("temporary prototype", cuda)
        self.assertIn("Remove them from the submitted Kernel source tree", cuda)
        self.assertIn("wrapping another DSL as the selected framework", cuda)

        triton = optimization_mode_directive("production", "Triton")
        self.assertIn("submitted candidate's implementation framework is exactly **Triton**", triton)
        self.assertIn("Do not submit a candidate that switches early", triton)

    def test_dsl_context_distinguishes_final_gate_from_development_tools(self):
        production = object.__new__(Campaign)
        production.specify_dsl = True
        production.optimization_mode = "production"
        production.framework = "Cuda"
        self.assertIn("Required final candidate DSL: `Cuda`", production.dsl_context)
        self.assertIn("prototypes may use any language or DSL", production.dsl_context)

        leaderboard = object.__new__(Campaign)
        leaderboard.specify_dsl = True
        leaderboard.optimization_mode = "leaderboard"
        leaderboard.framework = "Cuda"
        self.assertEqual(
            leaderboard.dsl_context,
            "- Recommended candidate DSL: `Cuda`",
        )

    def test_generalized_agent_workspace_exposes_metadata_but_not_exact_shapes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            operator = Path(directory)
            visible = agent_visible_operator_files(operator, generalized=True)
        self.assertIn("metadata.json", visible)
        self.assertNotIn("shapes.json", visible)

    def test_generalized_workspace_privacy_guard_allows_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            operator = root / "operator"
            operator.mkdir()
            (operator / "reference.py").write_text("class Model: pass\n")
            (operator / "shapes.json").write_text(
                json.dumps({"0": {"init_kwargs": {}, "input_kwargs": {"x": 1}}})
            )
            public_problem = {
                "schema_version": "atrex.agent_problem.v1",
                "objective": "Optimize the operator.",
                "evaluation": {
                    "exact_cases": "private",
                    "development_cases_are_evaluation_cases": False,
                },
                "operator_contract": {},
                "shape_domain": {},
                "invariants": [],
                "coverage_regimes": [],
            }
            (operator / "agent_problem.json").write_text(json.dumps(public_problem))
            campaign = Campaign(
                name="operator",
                kernel_demo=str(operator / "reference.py"),
                platform="L20D",
                framework="Cuda",
                optimization_mode="production",
                atrex_bench_root=str(root),
                work_dir=str(root / "work"),
                expose_evaluator_shapes=False,
            )
            campaign.workspace.mkdir(parents=True)
            (campaign.workspace / "agent_problem.json").write_text(
                json.dumps(public_problem)
            )
            (campaign.workspace / "metadata.json").write_text(
                json.dumps({"benchmark_contract": {}})
            )

            campaign._assert_generalized_inputs_are_private()

            (campaign.workspace / "shapes.json").write_text(
                (operator / "shapes.json").read_text()
            )
            with self.assertRaisesRegex(RuntimeError, "shapes.json"):
                campaign._assert_generalized_inputs_are_private()

    def test_generalized_workspace_can_expose_all_or_no_shapes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            operator = root / "operator"
            operator.mkdir()
            (operator / "reference.py").write_text("class Model: pass\n")
            (operator / "shapes.json").write_text(
                json.dumps(
                    {
                        "2": {"init_kwargs": {}, "input_kwargs": {"x": 2}},
                        "1": {"init_kwargs": {}, "input_kwargs": {"x": 1}},
                    }
                )
            )
            problem = {
                "schema_version": "atrex.agent_problem.v1",
                "objective": "Optimize.",
                "evaluation": {
                    "exact_cases": "private",
                    "development_cases_are_evaluation_cases": False,
                },
                "operator_contract": {},
                "shape_domain": {},
                "invariants": [],
                "coverage_regimes": [],
            }
            (operator / "agent_problem.json").write_text(json.dumps(problem))

            hidden = Campaign(
                name="hidden",
                kernel_demo=str(operator / "reference.py"),
                platform="L20D",
                framework="Cuda",
                optimization_mode="production",
                atrex_bench_root=str(root),
                work_dir=str(root / "hidden-work"),
                expose_evaluator_shapes=False,
            )
            hidden.workspace.mkdir(parents=True)
            hidden._ensure_agent_problem()
            self.assertFalse((hidden.workspace / "public_shapes.json").exists())
            self.assertTrue(hidden.shape_ids_are_opaque)
            hidden._assert_generalized_inputs_are_private()
            self.assertIn("exact evaluator Shape parameters are private", hidden._shape_disclosure_prompt())

            complete = Campaign(
                name="complete",
                kernel_demo=str(operator / "reference.py"),
                platform="L20D",
                framework="Cuda",
                atrex_bench_root=str(root),
                work_dir=str(root / "all-work"),
                expose_evaluator_shapes=True,
            )
            complete.workspace.mkdir(parents=True)
            complete._ensure_agent_problem()
            self.assertFalse(complete.shape_ids_are_opaque)
            complete._assert_generalized_inputs_are_private()
            self.assertEqual(complete._shape_disclosure_prompt(), "")

    def test_hidden_shape_request_fails_closed_without_generalized_projection(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            operator = Path(directory)
            (operator / "reference.py").write_text(
                "class Model: pass\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                SystemExit,
                "requires Native Atrex-Bench shapes.json",
            ):
                LegacyProblemProvider().load(
                    operator,
                    optimization_mode="production",
                    expose_evaluator_shapes=False,
                )

    def test_typed_eval_forwards_metadata_without_light_tolerance_defaults(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("class Model: pass\n")
            (workspace / "reference.py").write_text("class Model: pass\n")
            (workspace / "input.py").write_text(
                "def generate_input(*args, **kwargs): return ()\n"
            )
            (workspace / "shapes.json").write_text(
                json.dumps({"0": {"init_kwargs": {}, "input_kwargs": {}}})
            )
            metadata = {
                "benchmark_contract": {
                    "correctness_tolerances": {
                        "default": {
                            "atol": 0.125,
                            "rtol": 0.25,
                            "max_rel_l2": 0.5,
                        }
                    }
                }
            }
            (workspace / "metadata.json").write_text(json.dumps(metadata))

            request = build_typed_request(
                workspace,
                "L20N",
                600,
                [],
                ["python3", "test_kernel.py"],
                "run",
            )

        self.assertEqual(request["reference"]["metadata"], metadata)
        self.assertNotIn("atol", request["options"])
        self.assertNotIn("rtol", request["options"])
        self.assertNotIn("correctness_max_rel_l2", request["options"])
        self.assertIs(request["lock_clocks"], True)
        self.assertNotIn("gpu_clock_mhz", request["runner_overrides"])

    def test_v0_and_candidates_share_60_second_performance_timeout(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("class Model: pass\n")
            (workspace / "reference.py").write_text("class Model: pass\n")
            (workspace / "input.py").write_text(
                "def generate_input(*args, **kwargs): return ()\n"
            )
            (workspace / "shapes.json").write_text(
                json.dumps({"0": {"init_kwargs": {}, "input_kwargs": {}}})
            )

            v0 = build_typed_request(
                workspace,
                "L20D",
                1800,
                [],
                ["python3", "test_kernel.py", "--version", "v0"],
                "run",
            )
            candidate = build_typed_request(
                workspace,
                "L20D",
                1800,
                [],
                ["python3", "test_kernel.py"],
                "run",
            )

        self.assertEqual(v0["runner_overrides"]["perf_timeout_s"], 60)
        self.assertEqual(candidate["runner_overrides"]["perf_timeout_s"], 60)

    def test_v0_baseline_uses_terminal_infrastructure_recovery(self):
        result = {
            "all_pass": True,
            "latency_us_geomean": 1.0,
            "latency_us_by_shape": {"0": 1.0},
        }
        completed = subprocess.CompletedProcess(
            ["python", "test_kernel.py"],
            0,
            "[test_kernel] RESULT_JSON=" + json.dumps(result) + "\n",
            "",
        )
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo="reference.py",
                platform="cuda",
                framework="cuda",
                work_dir=directory,
                evaluation_step_timeout=317,
            )
            with (
                patch(
                    "aka.light.workflow.bootstrap._sandbox_command",
                    return_value=completed,
                ) as sandbox,
                patch.object(Campaign, "_print_v0_evaluator_output"),
            ):
                measured = campaign._run_v0_evaluator()

            from aka.light.workspace.baseline import baseline_record_path
            private_record = json.loads(
                baseline_record_path(campaign.workspace).read_text(encoding="utf-8")
            )
            self.assertEqual(private_record["version"], "v0")
            self.assertFalse((campaign.workspace / "memory/v0.json").exists())
            self.assertFalse((campaign.workspace / ".atrex_versions/v0.json").exists())

        self.assertEqual(measured, result)
        self.assertIs(sandbox.call_args.kwargs["terminal_validation"], True)
        self.assertEqual(sandbox.call_args.kwargs["evaluation_step_timeout"], 317)

    def test_v0_terminal_recovery_does_not_retry_incorrect_result(self):
        result = {
            "all_pass": False,
            "failures": ["shape 0 failed correctness"],
        }
        completed = subprocess.CompletedProcess(
            ["python", "test_kernel.py"],
            1,
            "[test_kernel] RESULT_JSON=" + json.dumps(result) + "\n",
            "",
        )
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo="reference.py",
                platform="cuda",
                framework="cuda",
                work_dir=directory,
            )
            with (
                patch(
                    "aka.light.workflow.bootstrap._sandbox_command",
                    return_value=completed,
                ) as sandbox,
                patch.object(Campaign, "_print_v0_evaluator_output"),
            ):
                with self.assertRaisesRegex(RuntimeError, "V0 baseline failed"):
                    campaign._run_v0_evaluator()

            from aka.light.workspace.baseline import baseline_failure_path
            failure = json.loads(
                baseline_failure_path(campaign.workspace).read_text(encoding="utf-8")
            )
            self.assertEqual(failure["quality_gate"]["result"], "FAIL")
            self.assertFalse((campaign.workspace / ".atrex_versions/v0.failed.json").exists())

        sandbox.assert_called_once()
        self.assertIs(sandbox.call_args.kwargs["terminal_validation"], True)

    def test_legacy_v0_record_migrates_to_private_supervisor_state(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            legacy = workspace / ".atrex_versions" / "v0.json"
            legacy.parent.mkdir(parents=True)
            record = {
                "version": "v0",
                "performance": {"latency_us_geomean": 2.0},
                "correctness": {"status": "PASS"},
            }
            legacy.write_text(json.dumps(record), encoding="utf-8")

            from aka.light.workspace.baseline import baseline_record_path
            from aka.light.workspace.state import latest_version, read_memory

            self.assertEqual(read_memory(workspace, 0), record)
            self.assertEqual(latest_version(workspace), 0)
            self.assertEqual(
                json.loads(baseline_record_path(workspace).read_text(encoding="utf-8")),
                record,
            )

    def test_v0_progress_exposes_gateway_record_but_keeps_backend_jobs_private(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            (workspace / "kernel.py").write_text(
                "class Model:\n    pass\n", encoding="utf-8"
            )
            subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "config", "user.name", "test"], cwd=workspace, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=workspace,
                check=True,
            )
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "V0"], cwd=workspace, check=True
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            from aka.light.experiment.store import MeasurementStore
            from aka.light.runtime.paths import supervisor_campaign_root
            from aka.light.workflow.journal_service import (
                _git_visible_record_ids,
                _progress_markdown,
            )
            from aka.light.workspace.baseline import (
                ensure_baseline_gateway_record,
                read_baseline_record,
                write_baseline_record,
            )

            write_baseline_record(
                workspace,
                {
                    "version": "v0",
                    "git_commit_hash": commit,
                    "backend_job_ids": ["ev_part_a", "ev_part_b"],
                    "performance": {
                        "latency_us_geomean": 5.86,
                        "latency_us_arith_mean": 6.0,
                        "latency_us_by_shape": {"0": 5.86},
                        "measured_shape_count": 1,
                        "performance_objective": "latency_us_geomean",
                    },
                    "correctness": {
                        "status": "PASS",
                        "max_abs_err": 0.0,
                        "max_rel_err": 0.0,
                    },
                    "quality_gate": {"result": "PASS"},
                },
            )
            gateway_record_id = ensure_baseline_gateway_record(workspace)

            self.assertIsNotNone(gateway_record_id)
            self.assertTrue(gateway_record_id.startswith("gateway-"))
            private = read_baseline_record(workspace)
            self.assertEqual(private["gateway_record_id"], gateway_record_id)
            self.assertEqual(private["backend_job_ids"], ["ev_part_a", "ev_part_b"])
            store = MeasurementStore(
                supervisor_campaign_root(workspace) / "measurements"
            )
            saved = store.read(gateway_record_id)
            self.assertEqual(
                saved["request"]["visibility"], "supervisor_private_v0"
            )
            self.assertIn("ev_part_a,ev_part_b", saved["response"]["stderr"])
            self.assertNotIn(
                gateway_record_id, _git_visible_record_ids(store, workspace)
            )

            progress = _progress_markdown(
                {
                    "v0_baseline": {
                        "kernel_commit_id": commit,
                        "latency_us_geomean": 5.86,
                        "measured_shape_count": 1,
                        "correctness_status": "PASS",
                        "gateway_record_id": gateway_record_id,
                    }
                }
            ).decode()
            self.assertIn(
                f"private Gateway Record: `{gateway_record_id}`", progress
            )
            self.assertNotIn("ev_part_a", progress)
            self.assertNotIn("ev_part_b", progress)

    def test_v0_finalization_keeps_git_at_source_commit(self):
        source_commit = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo="reference.py",
                platform="cuda",
                framework="cuda",
                work_dir=directory,
            )
            from aka.light.workspace.baseline import read_baseline_record, write_baseline_record
            write_baseline_record(
                campaign.workspace,
                {
                    "version": "v0",
                    "performance": {
                        "latency_us_geomean": 1.0,
                        "latency_us_by_shape": {"0": 1.0},
                        "measurement_status": "complete",
                        "measured_shape_count": 1,
                    },
                    "correctness": {"status": "PASS"},
                    "quality_gate": {"result": "PASS"},
                },
            )
            with (
                patch("aka.light.workflow.bootstrap.git_head", return_value=source_commit),
                patch("aka.light.workspace.sources.from_git", return_value=b"source"),
                patch("aka.light.workspace.sources.snapshot", return_value=b"source"),
            ):
                campaign._finalize_v0_measurement({}, source_commit)

            self.assertEqual(
                read_baseline_record(campaign.workspace)["git_commit_hash"],
                source_commit,
            )
            self.assertFalse((campaign.workspace / "memory/v0.json").exists())
            self.assertFalse((campaign.workspace / ".atrex_versions/v0.json").exists())

    def test_native_harness_has_no_light_owned_tolerance_switches(self):
        args = native_harness_parser().parse_args([])
        self.assertFalse(hasattr(args, "atol"))
        self.assertFalse(hasattr(args, "rtol"))
        self.assertFalse(hasattr(args, "correctness_max_rel_l2"))

    def test_framework_baseline_preserves_structured_failure_diagnostics(self):
        result = {
            "all_pass": False,
            "failures": ["shape 6 failed correctness"],
            "actionable_diagnostics": [
                {
                    "stage": "correctness",
                    "shape_id": "6",
                    "message": "output mismatch",
                }
            ],
            "diagnostics": [
                {
                    "stage": "correctness",
                    "shape_id": "7",
                    "case_index": 4,
                    "check": "outputs",
                    "name": "output",
                    "error": "elementwise tolerance failed",
                    "max_elementwise_rel_diff": 0.125,
                    "relative_l2": 0.01,
                }
            ],
        }
        completed = subprocess.CompletedProcess(
            ["python", "test_kernel.py"],
            1,
            "[test_kernel] RESULT_JSON=" + json.dumps(result) + "\n",
            "",
        )
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo="reference.py",
                platform="cuda",
                framework="cuda",
                work_dir=directory,
            )
            with (
                patch("aka.light.workflow.bootstrap._sandbox_command", return_value=completed) as sandbox,
                patch.object(Campaign, "_print_v0_evaluator_output"),
            ):
                parsed, problem = campaign._validate_framework_baseline(1)
        self.assertIs(sandbox.call_args.kwargs["terminal_validation"], True)
        self.assertEqual(parsed, result)
        self.assertIn("shape 6 failed correctness", problem)
        self.assertIn('\"shape_id\": \"6\"', problem)
        self.assertIn('\"shape_id\": \"7\"', problem)
        self.assertIn("elementwise tolerance failed", problem)
        self.assertIn('\"max_elementwise_rel_diff\": 0.125', problem)
        self.assertNotEqual(problem, "combined validation command failed (exit=1)")

    def test_orchestrator_evaluator_log_uses_compact_success_summary(self):
        result = {
            "all_pass": True,
            "latency_us_geomean": 4.5,
            "latency_us_arith_mean": 5.0,
            "performance_score": 2.0,
            "latency_us_by_shape": {str(index): index + 0.5 for index in range(10)},
            "performance_by_shape": {
                str(index): {"timing_stats": {"runs": [index] * 1000}}
                for index in range(10)
            },
            "correctness": {"status": "PASS"},
            "gateway_record_id": "gateway-" + "4" * 32,
            "eval_id": "eval-123",
        }
        completed = subprocess.CompletedProcess(
            ["python", "test_kernel.py"],
            0,
            "[test_kernel] RESULT_JSON=" + json.dumps(result) + "\n",
            "large compatibility warning\n" * 1000,
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            Campaign._print_v0_evaluator_output(completed)

        rendered = stdout.getvalue()
        self.assertIn('"status": "passed"', rendered)
        self.assertIn('"correctness": "PASS"', rendered)
        self.assertIn('"latency_us_geomean": 4.5', rendered)
        self.assertIn('"measured_shape_count": 10', rendered)
        self.assertIn('"gateway_record_id": "gateway-', rendered)
        self.assertNotIn("performance_by_shape", rendered)
        self.assertNotIn("timing_stats", rendered)
        self.assertEqual(stderr.getvalue(), "")
        self.assertLess(len(rendered), 1000)

    def test_orchestrator_evaluator_log_uses_bounded_failure_summary(self):
        root_cause = "kernel.cu:91: error: no matching function for call"
        result = {
            "all_pass": False,
            "failures": ["shape 0 compile failed"],
            "actionable_diagnostics": [{
                "stage": "compile",
                "shape_id": "0",
                "message": "template note\n" * 1000 + root_cause,
            }],
        }
        completed = subprocess.CompletedProcess(
            ["python", "test_kernel.py"],
            1,
            "[test_kernel] RESULT_JSON=" + json.dumps(result) + "\n",
            "",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            Campaign._print_v0_evaluator_output(completed)

        rendered = stderr.getvalue()
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn('"status": "failed"', rendered)
        self.assertIn('"failure_count": 1', rendered)
        self.assertIn(root_cause, rendered)
        self.assertNotIn("RESULT_JSON", rendered)
        self.assertLess(len(rendered), 3000)

    def test_orchestrator_evaluator_log_summarizes_gateway_error(self):
        error = {
            "operation": "evaluate",
            "status": "failed",
            "gateway_record_id": "gateway-" + "5" * 32,
            "error": {
                "error_class": "timeout",
                "reason": "job_deadline_exceeded",
                "message": "job deadline exceeded after 600 seconds",
                "details": {"large": "x" * 100_000},
            },
        }
        completed = subprocess.CompletedProcess(
            ["python", "test_kernel.py"],
            75,
            "[sandbox] GATEWAY_ERROR_JSON=" + json.dumps(error) + "\n",
            "",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            Campaign._print_v0_evaluator_output(completed)

        rendered = stderr.getvalue()
        self.assertIn('"error_class": "timeout"', rendered)
        self.assertIn('"reason": "job_deadline_exceeded"', rendered)
        self.assertIn("job deadline exceeded after 600 seconds", rendered)
        self.assertNotIn('"details"', rendered)
        self.assertLess(len(rendered), 2000)

    def test_agent_visible_compiler_diagnostic_preserves_complete_tail(self):
        repeated = "template instantiation note from torch headers\n" * 40_000
        root_cause = (
            "ATen/core/List_inl.h:201: error: dependent name is not a type; "
            "prefix it with 'typename'"
        )
        compiler_error = repeated + root_cause
        raw = {
            "passed": {
                "compile": {
                    "0": {"status": "failed", "reason": compiler_error},
                },
                "correctness": {},
            },
            "correctness": {"shapes": {}},
        }
        result = optimizer_result_from_eval(
            raw, ["0"], {}, require_performance=False
        )
        self.assertEqual(
            result["actionable_diagnostics"][0]["message"], compiler_error
        )
        projected = project_evaluation(result)
        self.assertEqual(
            projected["actionable_diagnostics"][0]["message"], compiler_error
        )

        response = {
            "exit_code": 1,
            "stdout": "[test_kernel] RESULT_JSON=" + json.dumps(result) + "\n",
            "stderr": "",
        }
        self.assertEqual(bound_diagnostic_response(response), response)

        completed = subprocess.CompletedProcess(
            ["python", "test_kernel.py"], 1, response["stdout"], ""
        )
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo="reference.py",
                platform="cuda",
                framework="cuda",
                work_dir=directory,
            )
            with (
                patch(
                    "aka.light.workflow.bootstrap._sandbox_command",
                    return_value=completed,
                ),
                patch.object(Campaign, "_print_v0_evaluator_output"),
            ):
                parsed, problem = campaign._validate_framework_baseline(1)
        self.assertEqual(parsed, result)
        self.assertIn(root_cause, problem)
        self.assertNotIn("diagnostic truncated", problem)

    def test_performance_objective_is_computed_after_all_shape_shards_merge(self):
        metadata = {
            "performance_objective": "shape_speedup_geometric_mean",
            "shapes": {
                "0": {"production_performance": {"performance_us": 100.0}},
                "1": {"production_performance": {"performance_us": 100.0}},
            },
        }

        def shard(shape_id, latency_us):
            raw = {
                "passed": {
                    "compile": {"status": "passed"},
                    "correctness": {shape_id: {"status": "passed"}},
                },
                "correctness": {"shapes": {shape_id: {"cases": []}}},
                "performance": {
                    "shapes": {
                        shape_id: {
                            "samples": [
                                {"end_to_end_time_ms": latency_us / 1000.0}
                            ]
                        }
                    }
                },
            }
            return optimizer_result_from_eval(
                raw,
                [shape_id],
                metadata,
                finalize_performance=False,
            )

        first = shard("0", 10.0)
        second = shard("1", 100.0)
        self.assertIsNone(first["performance_score"])
        self.assertIsNone(second["performance_score"])

        merged = merge_optimizer_results(
            [first, second], ["0", "1"], metadata
        )
        self.assertEqual(
            merged["performance_objective"], "shape_speedup_geometric_mean"
        )
        self.assertAlmostEqual(merged["speedup_vs_ref_mean"], 5.5)
        self.assertAlmostEqual(
            merged["speedup_vs_ref_geomean"], math.sqrt(10.0)
        )
        self.assertAlmostEqual(merged["performance_score"], math.sqrt(10.0))

        arithmetic = merge_optimizer_results(
            [first, second],
            ["0", "1"],
            dict(metadata, performance_objective="shape_speedup_arithmetic_mean"),
        )
        self.assertAlmostEqual(arithmetic["performance_score"], 5.5)

        latency = merge_optimizer_results(
            [first, second],
            ["0", "1"],
            {
                "performance_objective": "shape_latency_geometric_mean",
                "shapes": {"0": {}, "1": {}},
            },
        )
        self.assertAlmostEqual(latency["performance_score"], math.sqrt(1000.0))

    def test_workspace_score_hydration_obeys_metadata_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "metadata.json").write_text(
                json.dumps(
                    {
                        "performance_objective": "shape_speedup_geometric_mean",
                        "shapes": {
                            "0": {
                                "production_performance": {
                                    "performance_us": 100.0
                                }
                            },
                            "1": {
                                "production_performance": {
                                    "performance_us": 100.0
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = _with_workspace_performance_score(
                workspace,
                {
                    "latency_us_by_shape": {"0": 10.0, "1": 100.0},
                    "performance_objective": "shape_speedup_arithmetic_mean",
                    "performance_score": 5.5,
                },
            )

        self.assertEqual(
            result["performance_objective"], "shape_speedup_geometric_mean"
        )
        self.assertAlmostEqual(result["performance_score"], math.sqrt(10.0))

    def test_abba_shape_shards_use_one_full_metadata_objective_score(self):
        schedule = [
            {"revision": "candidate", "repeat": 0},
            {"revision": "incumbent", "repeat": 0},
        ]
        metadata = {
            "performance_objective": "shape_speedup_geometric_mean",
            "shapes": {
                "0": {"production_performance": {"performance_us": 100.0}},
                "1": {"production_performance": {"performance_us": 100.0}},
            },
        }

        def payload(shape_id, candidate_latency):
            def row(step, latency):
                return {
                    **step,
                    "exit_code": 0,
                    "result": {
                        "all_pass": True,
                        "failures": [],
                        "latency_us_by_shape": {shape_id: latency},
                    },
                }

            return {
                "schema_version": 1,
                "runs": [
                    row(schedule[0], candidate_latency),
                    row(schedule[1], 100.0),
                ],
                "error": None,
            }

        merged = merge_abba_batch_payloads(
            [payload("0", 10.0), payload("1", 100.0)],
            schedule,
            ["0", "1"],
            metadata,
        )

        candidate = merged["runs"][0]["result"]
        incumbent = merged["runs"][1]["result"]
        self.assertAlmostEqual(candidate["performance_score"], math.sqrt(10.0))
        self.assertAlmostEqual(incumbent["performance_score"], 1.0)
        self.assertEqual(
            candidate["performance_objective"],
            "shape_speedup_geometric_mean",
        )

    def test_repeated_long_diagnostics_are_deduplicated_not_truncated(self):
        root_cause = "nvcc: fatal error: unsupported gpu architecture 'compute_103'"
        compiler_error = ("repeated include trace\n" * 20_000) + root_cause
        feedback = evaluation_feedback(
            {
                "compile": {
                    "shapes": {
                        "0": {"error": compiler_error},
                        "1": {"error": compiler_error},
                    }
                }
            },
            ["0", "1"],
        )
        self.assertEqual(len(feedback["diagnostics"]), 1)
        self.assertEqual(feedback["diagnostics"][0]["error"], compiler_error)
        self.assertEqual(feedback["diagnostics"][0]["shape_ids"], ["0", "1"])
        projected = project_feedback(feedback)
        self.assertEqual(projected["diagnostics"][0]["error"], compiler_error)
        self.assertEqual(projected["diagnostics"][0]["shape_ids"], ["0", "1"])
        self.assertNotIn("feedback_truncation", projected)

    def test_framework_baseline_gates_run_concurrently_and_return_together(self):
        barrier = threading.Barrier(3)

        def policy_gate():
            barrier.wait(timeout=2)
            return ["review rejected final dependency"]

        def validation_gate(_version):
            barrier.wait(timeout=2)
            return {"all_pass": False}, "shape 7 failed correctness"

        def numerical_gate(_workspace, **options):
            barrier.wait(timeout=2)
            self.assertFalse(options["standard_correctness_passed"]())
            return "supplemental probe found an unstable case"

        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo="reference.py",
                platform="cuda",
                framework="cuda",
                work_dir=directory,
                optimization_mode="production",
            )
            with (
                patch.object(
                    Campaign,
                    "source_checks_enabled",
                    new_callable=PropertyMock,
                    return_value=True,
                ),
                patch.object(campaign, "_production_kernel_violations", side_effect=policy_gate),
                patch.object(campaign, "_validate_framework_baseline", side_effect=validation_gate),
                patch.object(campaign, "supplemental_numerical_feedback", side_effect=numerical_gate),
            ):
                result, problem = campaign._framework_baseline_external_gates(1)

        self.assertEqual(result, {"all_pass": False})
        self.assertIn("review rejected final dependency", problem)
        self.assertIn("shape 7 failed correctness", problem)
        self.assertIn("supplemental probe found an unstable case", problem)
        self.assertIn("- production_review: failed", problem)
        self.assertIn("- correctness_performance: failed", problem)
        self.assertIn("- numerical: failed", problem)

    def test_framework_baseline_leaderboard_skips_production_review(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo="reference.py",
                platform="cuda",
                framework="cuda",
                work_dir=directory,
                optimization_mode="leaderboard",
            )
            with (
                patch.object(
                    Campaign,
                    "source_checks_enabled",
                    new_callable=PropertyMock,
                    return_value=True,
                ),
                patch.object(
                    campaign,
                    "_production_kernel_violations",
                ) as policy_gate,
                patch.object(
                    campaign,
                    "_validate_framework_baseline",
                    return_value=({"all_pass": True}, ""),
                ),
                patch.object(
                    campaign,
                    "supplemental_numerical_feedback",
                    return_value="",
                ),
            ):
                result, problem = campaign._framework_baseline_external_gates(1)

        self.assertEqual(result, {"all_pass": True})
        self.assertEqual(problem, "")
        policy_gate.assert_not_called()

    def test_framework_baseline_keeps_bounded_transport_error_without_result(self):
        completed = subprocess.CompletedProcess(
            ["python", "test_kernel.py"],
            1,
            "",
            "gateway rejected request: shape contract is invalid",
        )
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(
                name="operator",
                kernel_demo="reference.py",
                platform="cuda",
                framework="cuda",
                work_dir=directory,
            )
            with (
                patch("aka.light.workflow.bootstrap._sandbox_command", return_value=completed),
                patch.object(Campaign, "_print_v0_evaluator_output"),
            ):
                parsed, problem = campaign._validate_framework_baseline(1)
        self.assertIsNone(parsed)
        self.assertIn("shape contract is invalid", problem)


class ProjectedProcStartupPreflightTests(unittest.TestCase):
    @staticmethod
    def _config(*, proc_mode="projected"):
        return {
            "sandbox": "bwrap",
            "capture": True,
            "timeout_seconds": 28800,
            "bwrap_executable": "bwrap",
            "proc_mode": proc_mode,
            "read_only_paths": [],
        }

    def test_projected_light_sandbox_compiles_shim_before_campaign_launch(self):
        from aka.light.agent.plugin import validate_config

        with (
            patch("aka.light.agent.sandbox.sandbox_executable"),
            patch("aka.light.agent.preflight.check_projected_proc") as preflight,
        ):
            validate_config(self._config())

        preflight.assert_called_once_with()

    def test_projected_light_sandbox_reports_missing_compiler_as_startup_error(self):
        from aka.light.agent.plugin import validate_config

        with (
            patch("aka.light.agent.sandbox.sandbox_executable"),
            patch(
                "aka.light.agent.preflight.check_projected_proc",
                side_effect=FileNotFoundError("cc"),
            ),
            self.assertRaisesRegex(
                ValueError,
                r"working C compiler and development headers.*preflight",
            ),
        ):
            validate_config(self._config())

    def test_private_proc_does_not_require_compiler_preflight(self):
        from aka.light.agent.plugin import validate_config

        with (
            patch("aka.light.agent.sandbox.sandbox_executable"),
            patch("aka.light.agent.preflight.check_projected_proc") as preflight,
        ):
            validate_config(self._config(proc_mode="private"))

        preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
