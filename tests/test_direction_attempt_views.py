from aka.workflow.light.host.journal import _publish_research_library_tree


def entry(kind, identity, version, **fields):
    singular = 'direction' if kind == 'directions' else 'experiment'
    return (f'research-library/{kind}/{identity}/v{version}/{singular}.json',
            {f'{singular}_id': identity, **fields}, {}, set())


def test_attempt_projection_latest_visible_complete_and_refresh(tmp_path):
    versions = {('directions', 'd1', 1): entry('directions', 'd1', 1, name='Stores', supporting_experiment_ids=[])}
    # More than a recent-history shortlist; include unsupported and failed attempts.
    for n in range(80):
        eid = f'e{n:03d}'
        versions[('experiments', eid, 1)] = entry(
            'experiments', eid, 1, direction_id='d1', name='Failed trial',
            claim_review_quality='unsupported', implementation='A | B\n<probe>',
            interpretation='Regression; cause not established', expectations=[
                dict(scope='long', prediction='faster', observation='slower', assessment='missed'),
                dict(scope='short', prediction='no regression', observation='not measured', assessment='unresolved')])
    versions[('experiments', 'unlinked', 1)] = entry('experiments', 'unlinked', 1, name='unlinked')
    _publish_research_library_tree(tmp_path, versions, canonical=False)
    path = tmp_path / 'research-library/directions/d1/attempts.md'
    first = path.read_text()
    assert 'e079' in first and 'e000' in first and 'unlinked' not in first
    assert 'A &#124; B &lt;probe&gt;' in first
    assert 'slower | missed | retrospective | Not marked unresolved' in first
    assert 'not measured | unresolved | retrospective | no regression' in first
    # A later visible revision updates only the derived view, not old JSON.
    old = (tmp_path / 'research-library/experiments/e000/v1/experiment.json').read_bytes()
    versions[('experiments', 'e000', 2)] = entry('experiments', 'e000', 2, direction_id='d1', name='New result', implementation='reuse', expectations=[])
    _publish_research_library_tree(tmp_path, versions, canonical=False)
    assert '[e000 v2]' in path.read_text() and '[e000 v1]' not in path.read_text()
    assert old == (tmp_path / 'research-library/experiments/e000/v1/experiment.json').read_bytes()
    # A frozen visibility snapshot cannot leak the newer revision from disk.
    del versions[('experiments', 'e000', 2)]
    _publish_research_library_tree(tmp_path, versions, canonical=False)
    assert path.read_text() == first
    # Removed links and absent fields remain honest; empty Direction still has a view.
    versions = {('directions', 'd1', 1): versions[('directions', 'd1', 1)]}
    _publish_research_library_tree(tmp_path, versions, canonical=False)
    assert 'No linked Experiments are visible yet.' in path.read_text()
