"""Public sample snapshots for the explicitly selected dashboard preview."""
import json
from datetime import datetime, timezone
from pathlib import Path

from aka.contracts.workflow_export import (
    EXPORT_VERSION, CampaignRef, CandidateFact, EpisodeFact, ProgressFact, ResultFact,
    WorkflowExportSnapshot, snapshot_document,
)
from .reader import PHASES


def create_demo_workspace(root: Path) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    def token_usage(total, measurement='exact'):
        terminal = {'input_tokens': int(total * .8), 'output_tokens': total - int(total * .8),
                    'cache_read_tokens': int(total * .4), 'cache_write_tokens': 0,
                    'total_tokens': total, 'measurement': measurement}
        return {'total': total, 'control_tokens': total, 'raw_control_tokens': total,
                'measurement': measurement, 'terminal_usage': terminal,
                'attribution_measurement': measurement, 'reason_codes': []}
    def measured(score, passed=True):
        return ResultFact('pass' if passed else 'rejected', 'PASS', 'PASS' if passed else 'FAIL',
            True, 10 / score, {'sample-a': 10 / score, 'sample-b': 10 / score},
            ('sample-a', 'sample-b'), 'candidate', 'shape_speedup_arithmetic_mean', score)
    cuda = root / 'kernel_opt_demo_attention_cuda'
    cuda.mkdir(parents=True)
    baseline = CandidateFact(0, None, None, None, True, now, measured(1))
    candidates = [baseline]
    episodes = []
    rows = (
        ('fast', 'candidate_ready', True, 120000, 1.12, 'Vectorize contiguous memory loads'),
        ('fast', 'candidate_ready', False, 160000, 1.12, 'Larger tiles did not improve latency'),
        ('full', 'candidate_ready', True, 190000, 1.31, 'Fuse normalization with the reduction'),
        ('full', 'pivot', False, 200000, 1.31, 'Switch to register reuse'),
        ('full', 'candidate_ready', True, 180000, 1.49, 'Reduce synchronization between warps'),
        ('full', 'candidate_ready', False, 141000, 1.49, 'Extra unrolling increased register pressure'),
        ('full', 'candidate_ready', True, 190000, 1.54, 'Specialize the vectorized tail path'),
        ('goal', 'blocked', False, 99000, 1.54, 'Target requires a different launch strategy'))
    for number, (mode, status, accepted, total, score, summary) in enumerate(rows, 1):
        candidates.append(CandidateFact(number, number, None, None, accepted, now, measured(score, accepted), summary))
        phases = tuple({'phase': phase, 'usage': token_usage(total // len(PHASES))['terminal_usage']} for phase in PHASES)
        episodes.append(EpisodeFact(number, number, status, accepted, None, None, summary,
            'PASS' if accepted else 'FAIL', ({'name': summary, 'result': 'Candidate preserved for analysis'},),
            phases, mode, usage=token_usage(total, 'partial' if status == 'blocked' else 'exact'),
            verification={'gate': 'PASS' if accepted else 'FAIL', 'candidate_latency_us': 10 / score}, detail_available=True))
    latest = {'name': 'Overlap memory loads with computation', 'result': 'Candidate ready for the next evaluator pass'}
    episodes.append(EpisodeFact(9, 9, 'observed_active', False, None, None, None, None,
        (latest,), mode='full', detail_available=True))
    candidates.append(CandidateFact(9, 9, None, None, False, None,
        ResultFact('running', None, None, False, None, {}, ('sample-a', 'sample-b'), None, None, None)))
    incumbent = next(candidate for candidate in reversed(candidates) if candidate.promoted)
    total = sum(row[3] for row in rows)
    facts = WorkflowExportSnapshot(EXPORT_VERSION, CampaignRef('demo-cuda', cuda, cuda.name), 'Cuda',
        tuple(candidates), baseline, incumbent, tuple(episodes), (),
        ProgressFact('exploring', 9, 9, 8, total, now, False, 'full', 5, latest, pending_tokens=True), (),
        hardware={'platform': 'H20'}, token_summary={'total': total, 'control_total': total,
            'measurement': 'partial', 'receipt_count': 8, 'reason_codes': [], 'pending': True},
        counts={'episodes': 8, 'accepted': 4, 'rejected': 2})
    (cuda / 'workflow-export.json').write_text(json.dumps(snapshot_document(facts), indent=2))
    triton = root / 'kernel_opt_demo_attention_triton'; triton.mkdir()
    baseline = CandidateFact(0, None, None, None, True, now, measured(1))
    facts = WorkflowExportSnapshot(EXPORT_VERSION, CampaignRef('demo-triton', triton, triton.name), 'Triton',
        (baseline,), baseline, baseline, (), (), ProgressFact('waiting', None, 0, 0, None, now, False), (),
        hardware={'platform': 'H20'}, counts={'episodes': 0, 'accepted': 0, 'rejected': 0})
    (triton / 'workflow-export.json').write_text(json.dumps(snapshot_document(facts), indent=2))
    return root
