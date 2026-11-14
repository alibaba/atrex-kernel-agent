"""Prepare compiled-candidate probes through typed Eval with one budget per case."""
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

from aka.contracts.evaluation import PreparedEvaluation, StepBudget, NumericalProbeResult
from aka.contracts.execution import CommandSpec
from aka.contracts.workspace import RevisionContent
from aka.contracts.measurement import MeasurementPoint, MeasurementAssessment, MeasurementDiagnostic
from .drivers import numerical_inputs
from .source_archives import prepare_source, source_parameters
from .gateway_request import DEFAULT_DEPENDENCY_POLICY, prepare_request
from .transport_results import _shape_batch_reference
from .measurement import gateway_identity


@dataclass(frozen=True)
class TypedNumericalPlan:
    request: object
    schedule: tuple
    step_plans: tuple
    identities: tuple
    source_artifact: object
    native: bool = False
    typed_numerical: bool = True


def input_source(original, case, seed):
    algorithms = Path(numerical_inputs.__file__).read_text().replace("from __future__ import annotations\n", "")
    return original + ("\n" + algorithms + "\n__aka_original_inputs = _make_inputs\n"
        "def _make_inputs(**kwargs):\n"
        f"    return numerical_inputs(__aka_original_inputs(**kwargs), {case!r}, {seed!r}, 0)\n").encode()


def prepare_typed(request, configuration, suite, shapes, schedule, trusted, limit, requirements):
    if suite['world_size'] != 1:
        raise ValueError('typed supplemental probes require one GPU')
    probe = request.numerical_probe
    source = RevisionContent({item.path: item.content for item in probe.candidate_files})
    descriptor, archive, artifact, _ = prepare_source(source)
    solution = json.loads(source.files['solution.json'])
    facts = source_parameters(source)
    steps, identities, plans, budgets = [], [], [], []
    for plan in schedule:
        if plan.get('status') == 'unsupported':
            continue
        case = next(case for case in suite['cases'] if case['id'] == plan['case_id'])
        budget = StepBudget('numerical:' + probe.rotation + ':' + plan['case_id'], configuration.timeout)
        for shape_id in plan['shape_ids']:
            for seed in plan['seeds']:
                content = {name: item.content for name, item in trusted.items()}
                content['input.py'] = input_source(content['input.py'], case, seed)
                content['shapes.json'] = json.dumps({shape_id: shapes[shape_id]}).encode()
                # vlong + multi-seed selects correctness before baseline validation;
                # the final typed request freezes exactly one fixed-seed probe.
                command = ('python3', 'test_kernel.py', '--version', 'vlong', '--multi-seed', '1')
                payload = prepare_request(request.workspace, configuration.hardware, configuration.timeout,
                    [], list(command), 'run', reference_dir=request.workspace, reference_content=content,
                    candidate_source='', candidate_solution=solution, requirements=requirements,
                    runtime_version=configuration.environment.get('ATREX_SANDBOX_ATREX_BENCH_VERSION', ''),
                    measurement_environment=configuration.environment)
                payload['candidate'] = descriptor
                payload['mode'] = 'correctness_only'
                # Supplemental probes are independent requests, not parent CLI
                # replays. Freeze their existing default before deriving identity.
                payload['deps_mode'] = DEFAULT_DEPENDENCY_POLICY
                payload['options'].update(num_correctness_cases=1, correctness_max_rel_l2=limit)
                files = {name: data.decode() for name, data in _shape_batch_reference(payload['reference']).items()}
                step = CommandSpec(request.workspace, command, timeout=configuration.timeout, operation='run',
                    hardware=configuration.hardware, environment=configuration.environment,
                    requirements=requirements, source_archives=(archive,) if archive else (),
                    submission=json.dumps({'schema_version': 1, 'kind': 'run', 'request': payload,
                                           'reference_files': files}, allow_nan=False).encode())
                configuration.execution_capabilities.require(step)
                identity = gateway_identity(payload, files, source_parameters=facts)
                identity = replace(identity, method='supplemental-numerical/typed-v1',
                    points=(MeasurementPoint('candidate', 'sha256:' + probe.candidate_identity, seed, shape_id),),
                    parameters={**identity.parameters, 'case_id': plan['case_id'], 'seed': seed,
                                'case_budget': {'key': budget.key, 'seconds': budget.seconds},
                                'rotation': probe.rotation, 'selection_digest': plan['selection_digest']})
                steps.append(step); identities.append(identity); budgets.append(budget)
                plans.append({'case_id': plan['case_id'], 'shape_id': shape_id, 'seed': seed})
    context = TypedNumericalPlan(request, schedule, tuple(plans), tuple(identities), artifact)
    return PreparedEvaluation(tuple(steps), 1, context, measurement_identities=tuple(identities),
                              step_budgets=tuple(budgets))


def probe_result(plan, index, result):
    if result is None or result.transport_status not in {'succeeded', 'failed'}:
        return NumericalProbeResult('needs_validation', 'execution_unavailable', 'missing', 1, 0)
    from .gateway_command import _job_response
    from .native_abba import complete_evaluation
    from .drivers.native_harness import result_from_eval
    from .feedback import evaluation_feedback
    job = _job_response(result.stdout)
    shape = plan.step_plans[index]['shape_id']
    raw = job.get('result') if job and job.get('status') in {'succeeded', 'failed'} else None
    if (result.facts.disposition != 'terminal' or not isinstance(raw, dict)
            or not complete_evaluation(raw, [shape], correctness_only=True)):
        return NumericalProbeResult('needs_validation', 'missing_receipt', 'missing', 1, 0)
    cases = ((raw.get('correctness') or {}).get('shapes') or {}).get(shape, {}).get('cases')
    if not isinstance(cases, list) or len(cases) != 1:
        return NumericalProbeResult('needs_validation', 'incomplete_probe_coverage', 'incomplete', 1, 0)
    payload = result_from_eval(raw, [shape], {}, require_performance=False)
    payload.update(evaluation_feedback(raw, [shape]))
    diagnostics = payload.get('diagnostics', [])
    nonfinite = tuple(payload.get('nonfinite_outputs', ()))
    failed = [d for d in diagnostics if isinstance(d, dict) and d.get('stage') == 'correctness'
              and d.get('check') in {'outputs', 'mutated_inputs', 'unexpected_mutations'}
              and any(type(d.get(k)) in (int, float) and math.isfinite(d[k])
                      for k in ('relative_l2', 'max_elementwise_abs_diff', 'max_elementwise_rel_diff'))]
    status = ('needs_validation' if {'reference', 'unknown'}.intersection(nonfinite)
              else 'passed' if result.returncode == 0 and payload.get('all_pass') is True
              else 'needs_repair' if failed and all(isinstance(d, dict) and d.get('stage') == 'correctness' for d in diagnostics)
              else 'needs_validation')
    metrics = {key: max(values) for key in (
        'relative_l2', 'max_elementwise_abs_diff', 'max_elementwise_rel_diff')
        if (values := [d[key] for d in failed if type(d.get(key)) in (int, float)
                       and math.isfinite(d[key])])}
    complete = status in {'passed', 'needs_repair'}
    return NumericalProbeResult(status, status, 'complete' if complete else 'incomplete', 1,
        int(complete), int(status == 'needs_repair'), metrics=metrics, nonfinite_outputs=nonfinite,
        exit_code=result.exit_code)


def assess(plan, index, result=None):
    fact = probe_result(plan, index, result)
    identity = plan.identities[index]
    complete = fact.status in {'passed', 'needs_repair'}
    return MeasurementAssessment(identity, 'complete' if complete else 'unknown',
        'valid' if complete else 'unknown', identity.points if complete else (),
        () if complete else identity.points, (MeasurementDiagnostic('numerical', fact.reason),),
        'comparable' if result and result.environment_identity else 'unknown',
        result.environment_identity if result else None)


def interpret(prepared, results):
    from .numerical_preparation import summarize_numerical
    plan = prepared.context
    if len(results) > len(plan.step_plans):
        raise ValueError('unexpected typed numerical results')
    probes = []
    assessments = []
    for index, item in enumerate(plan.step_plans):
        result = results[index] if index < len(results) else None
        fact = probe_result(plan, index, result)
        probes.append({**item, **fact.as_dict(), 'receipt_status': fact.receipt_status})
        assessments.append(assess(plan, index, result))
    for item in plan.schedule:
        if item.get('status') == 'unsupported':
            fact = NumericalProbeResult('unsupported', 'unsupported', 'missing', diagnosis=item['diagnosis'])
            probes.append({'case_id': item['case_id'], **fact.as_dict(), 'receipt_status': fact.receipt_status})
    return summarize_numerical(probes, tuple(assessments))
