from copy import deepcopy

import pytest

from aka.workflow.light.host.journal import (
    initialize_journal, update_direction, _direction_views,
)


@pytest.fixture(params=[('directions',), ('directions', 'experiments')])
def direction(tmp_path, request):
    state = initialize_journal(tmp_path / 'state.json', episode=1,
                               base_commit='a' * 40, branch='work',
                               tool_modules=request.param)
    proposal = dict(action='propose', name='Memory traffic architecture',
                    change_scope='major_redesign', objective='Reduce memory traffic',
                    applicability='All vector sizes', expected_benefit='Lower latency',
                    plan=['Compare layouts', 'Measure launch families'])
    result = update_direction(state, tmp_path, proposal,
                              review_claim=lambda *_: {'claim_review_quality': 'supported'})
    revision = dict(proposal, action='revise', direction_id=result['direction_id'])
    return state, tmp_path, revision


def test_active_analysis_preserved_and_reviewed(direction):
    state, root, revision = direction
    reviewed = []

    def review(kind, event):
        reviewed.append(deepcopy(event))
        return {'claim_review_quality': 'supported'}

    update_direction(state, root, dict(action='start', direction_id=revision['direction_id'],
                                     analysis='Initial assessment'), review_claim=review)
    old_events = deepcopy(state['direction_events'])
    result = update_direction(state, root, dict(revision, analysis='Layout improved; launch work remains'),
                              review_claim=review)
    view = _direction_views(state, root)[revision['direction_id']]
    assert view['status'] == 'in_progress'
    assert view['analysis'] == reviewed[-1]['analysis'] == 'Layout improved; launch work remains'
    assert state['direction_events'][:-1] == old_events
    next_result = update_direction(state, root, revision, review_claim=review)
    assert next_result['revision'] == result['revision'] + 1
    assert reviewed[-1]['analysis'] == view['analysis']
    assert _direction_views(state, root)[revision['direction_id']]['analysis'] == view['analysis']


def test_proposed_revision_is_reviewed_without_experiments(direction):
    state, root, revision = direction
    reviewed = []
    def review(kind, event):
        reviewed.append(deepcopy(event))
        return {'claim_review_quality': 'supported'}
    result = update_direction(state, root, dict(revision, analysis='Preliminary assessment'), review_claim=review)
    view = _direction_views(state, root)[revision['direction_id']]
    assert view['status'] == 'proposed'
    assert view['analysis'] == reviewed[0]['analysis'] == 'Preliminary assessment'
    assert view['claim_review_quality'] == 'supported'
    assert result['revision'] == 2
    assert state['experiments'] == []


@pytest.mark.parametrize('quality', ['supported', 'partially_supported', 'unsupported'])
def test_propose_reviews_before_archival_without_experiments(direction, quality):
    state, root, revision = direction
    proposal = {k: v for k, v in revision.items() if k != 'direction_id'}
    proposal['action'] = 'propose'
    before = deepcopy(state)
    reviewed = []
    def review(kind, event):
        assert state == before  # Review is before mutation, not after publication.
        assert kind == 'direction'
        assert event['action'] == 'propose'
        assert event['revision'] == 1
        assert not event.get('supporting_experiment_ids')
        assert not event.get('derived_from_experiment_ids')
        reviewed.append(deepcopy(event))
        return {'claim_review_quality': quality}
    if quality == 'unsupported':
        with pytest.raises(ValueError, match='nothing was archived'):
            update_direction(state, root, proposal, review_claim=review)
        assert state == before
    else:
        result = update_direction(state, root, proposal, review_claim=review)
        assert result['review']['claim_review_quality'] == quality
        assert result['revision'] == 1
        view = _direction_views(state, root)[result['direction_id']]
        assert view['status'] == 'proposed'
        assert view['claim_review_quality'] == quality
    assert len(reviewed) == 1
    assert state['experiments'] == []


def test_start_and_restart_preserve_review_without_running_reviewer(direction):
    state, root, revision = direction
    def unexpected(*args):
        pytest.fail('Start must not trigger review')
    request = dict(action='start', direction_id=revision['direction_id'], analysis='Begin the investigation')
    result = update_direction(state, root, request, review_claim=unexpected)
    view = _direction_views(state, root)[revision['direction_id']]
    assert 'review' not in result
    assert result['revision'] == 2
    assert view['status'] == 'in_progress'
    assert view['claim_review_quality'] == 'supported'
    # A paused Direction also restarts without a new reviewer invocation.
    update_direction(state, root, dict(request, action='defer', reopen_condition='Resources available'),
                     review_claim=lambda *_: {'claim_review_quality': 'partially_supported'})
    result = update_direction(state, root, request, review_claim=unexpected)
    view = _direction_views(state, root)[revision['direction_id']]
    assert 'review' not in result
    assert result['revision'] == 4
    assert view['status'] == 'in_progress'
    assert view['claim_review_quality'] == 'partially_supported'
    assert state['experiments'] == []


@pytest.mark.parametrize('analysis', [None, '', 12, []])
def test_invalid_analysis_does_not_mutate(direction, analysis):
    state, root, revision = direction
    before = deepcopy(state)
    with pytest.raises(ValueError):
        update_direction(state, root, dict(revision, analysis=analysis))
    assert state == before


def test_rejected_revision_preserves_previous_analysis(direction):
    state, root, revision = direction
    update_direction(state, root, dict(action='start', direction_id=revision['direction_id'],
                                     analysis='Original assessment'),
                     review_claim=lambda *_: {'claim_review_quality': 'supported'})
    before = deepcopy(state)
    with pytest.raises(ValueError):
        update_direction(state, root, dict(revision, analysis='Unsupported assessment'),
                         review_claim=lambda *_: {'claim_review_quality': 'unsupported'})
    assert state == before
