from copy import deepcopy

import pytest

from aka.workflow.light.host.journal import (
    initialize_journal, update_direction, _direction_views,
)


@pytest.fixture
def direction(tmp_path):
    state = initialize_journal(tmp_path / 'state.json', episode=1,
                               base_commit='a' * 40, branch='work',
                               tool_modules=('directions',))
    proposal = dict(action='propose', name='Memory traffic architecture',
                    change_scope='major_redesign', objective='Reduce memory traffic',
                    applicability='All vector sizes', expected_benefit='Lower latency',
                    plan=['Compare layouts', 'Measure launch families'])
    result = update_direction(state, tmp_path, proposal)
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


def test_proposed_revision_stays_unreviewed(direction):
    state, root, revision = direction
    def unexpected(*args):
        pytest.fail('Draft revision must not trigger review')
    update_direction(state, root, dict(revision, analysis='Preliminary assessment'), review_claim=unexpected)
    view = _direction_views(state, root)[revision['direction_id']]
    assert view['status'] == 'proposed'
    assert view['analysis'] == 'Preliminary assessment'
    assert view['claim_review_quality'] == 'review_not_requested'


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
