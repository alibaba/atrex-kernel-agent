"""Launch policy, scoped tool help and independent research/submission paths."""
import argparse
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from aka.bootstrap.workflow_request import register_request_options, request_arguments
from aka.contracts.workflow import CampaignRuntimeSettings
from aka.contracts.workflow_launch import decode_launch, encode_launch
from aka.contracts.agate_tools import AGATE_TOOLS, DEFAULT_AGATE_TOOLS
from aka.workflow.light.host.errors import AgentRequestError
from aka.workflow.light.orchestration.supervisor_runtime import SupervisorRuntime


def runtime(tools=DEFAULT_AGATE_TOOLS):
    owner=object.__new__(SupervisorRuntime)
    owner.config=SimpleNamespace(agate_tools=tools, evaluation_step_timeout=60)
    owner.evaluation_configuration=SimpleNamespace(warmup_iters=10,bench_iters=20)
    return owner


def test_launch_selection_survives_resume(tmp_path):
    parser=argparse.ArgumentParser()
    register_request_options(parser)
    for extra, expected in (([],DEFAULT_AGATE_TOOLS),(['--agate-tools','all'],AGATE_TOOLS),
                            (['--agate-tools','evaluate','profile'],('evaluate','profile'))):
        args=parser.parse_args(['--problem',str(tmp_path),'--platform','L20N',*extra])
        launch=decode_launch(request_arguments(args,parser))
        assert launch.request.runtime.agate_tools == expected
        assert decode_launch(encode_launch(launch)).request.runtime.agate_tools == expected


@pytest.mark.parametrize('value',[['env','env'],['all','env'],['submit-kernel'],['wiki-query'],'env',None])
def test_invalid_selection(value):
    with pytest.raises(ValueError,match='agate_tools'):
        CampaignRuntimeSettings(agate_tools=value)


@pytest.mark.parametrize('kind',['profile','dev','check','disassemble'])
@pytest.mark.parametrize('suffix',[[],['--help']])
def test_disabled_tools_rejected_before_staging_or_execution(kind,suffix):
    owner=runtime()
    with pytest.raises(AgentRequestError) as caught:
        owner.execute(None,'/v1/gateway/execute',{'argv':['--kind',kind,*suffix]})
    assert caught.value.response['error']['code']=='runtime_tool_disabled'
    assert caught.value.response['error']['job_submitted'] is False
    assert caught.value.response['error']['available_kinds']==list(DEFAULT_AGATE_TOOLS)


def test_implicit_profile_cannot_bypass_selection():
    with pytest.raises(AgentRequestError):
        runtime().execute(None,'/v1/gateway/execute',{'argv':['--','python3','profile_driver.py']})


def test_help_lists_only_selected_tools_and_all_restores_optional_help():
    assert runtime().execute(None,'/v1/gateway/execute',{'argv':['--help']})['result']['available_kinds']==list(DEFAULT_AGATE_TOOLS)
    result=runtime(AGATE_TOOLS).execute(None,'/v1/gateway/execute',{'argv':['--kind=profile','--help']})
    assert '--profiler' in result['result']['request_schema']['cli_options']
    assert result['result']['request_schema']['cli_options']['--kind']=={'const':'profile'}
    assert 'baseline' in str(runtime().execute(None,'/v1/gateway/execute',{'argv':['--kind','evaluate','--help']}))


def test_research_submission_and_wiki_routes_are_independent(tmp_path):
    owner=runtime(())
    owner.execute_journal=Mock(return_value={'ok':True})
    for kind in ('direction_update','experiment_record','kernel_submit'):
        assert owner.execute(None,'/v1/journal/execute',{'operation':kind})=={'ok':True}
    owner.execute_plugin=Mock(return_value={'ok':True})
    assert owner.execute(None,'/v1/plugins/execute',{'action':'call','tool':'wiki.query'})=={'ok':True}
    owner.environment={}
    owner.root=tmp_path
    owner._wiki_command=Mock(return_value='wiki-command')
    owner.wiki=SimpleNamespace(execute=Mock(return_value={'ok':True}))
    assert owner.execute(SimpleNamespace(context={}),'/v1/wiki/query',{'tool':'query_nl','argv':[]})=={'ok':True}


def test_composed_custom_tools_with_research_and_submit():
    from test_integrated_campaign import ComposedCampaignTest
    case=ComposedCampaignTest()
    case.setUp()
    try:
        case.run_campaign('fast',('directions','experiments'),evaluate=True,agate_tools=('all',))
    finally:
        case.doCleanups()


@pytest.mark.parametrize('kind', ['run', 'preflight', 'same_allocation_abba', 'auto', 'typo'])
def test_invalid_kind_only_advertises_enabled_public_names(kind):
    from aka.workflow.light.orchestration.supervisor_runtime import RequestValidationError
    with pytest.raises(RequestValidationError) as caught:
        runtime().execute(None, '/v1/gateway/execute', {'argv': ['--kind', kind, '--help']})
    choices = str(caught.value).split('choose from', 1)[1]
    assert {part.strip(" '()") for part in choices.split(',')} == set(DEFAULT_AGATE_TOOLS)


def test_error_schema_does_not_restore_disabled_tools():
    from aka.evaluation.tool_requests import tool_argument_parser
    from aka.workflow.light.host.errors import error_response
    from aka.workflow.light.host.request_feedback import augment_error
    for kind in ('profile', 'run', 'auto'):
        result = augment_error(error_response('invalid request'), route='/v1/gateway/execute',
            request={'argv': ['--kind', kind]}, parser=tool_argument_parser(),
            enabled_kinds=DEFAULT_AGATE_TOOLS)
        assert set(result['request_schema']['supported_kinds']) == set(DEFAULT_AGATE_TOOLS)
        assert 'cli_options' not in result['request_schema']


@pytest.mark.parametrize('selected', [DEFAULT_AGATE_TOOLS, ('env',), ('evaluate', 'profile'), (), AGATE_TOOLS])
def test_installed_skill_and_discovery_match_runtime_help(tmp_path, monkeypatch, selected):
    import json
    import re
    from pathlib import Path
    from aka.contracts.agent_resources import AgentSkill
    from aka.resources.bundles import bundle_contributions
    from aka.workflow.light.runtime_resources import runtime_resource_registry

    resources, skills = bundle_contributions(('measurement-session', 'research-records', 'kernel-wiki'))
    execution_skills = Path(__file__).resolve().parents[1] / 'src/aka/execution/skills'
    skills += tuple(AgentSkill(name, execution_skills / name, 'Host operation manual')
                    for name in ('aka-execution', 'atrex-gpu-gateway'))
    monkeypatch.setattr('aka.resources.contributions.selected_resources', lambda: (resources, skills))
    monkeypatch.delenv('ATREX_PLUGIN_DIR', raising=False)
    monkeypatch.setenv('ATREX_PLUGIN_PATHS', '[]')
    registry = runtime_resource_registry(tmp_path / 'campaign', agate_tools=selected, wiki=True)
    workspace = tmp_path / 'workspace'
    installation = registry.install(workspace)
    names = {skill.name for skill in installation.module_skills}
    assert names == {'gpu-measurement', 'runtime-records', 'kernel-wiki'}
    assert not (workspace / '.claude/skills/aka-execution').exists()
    assert not (workspace / '.claude/skills/atrex-gpu-gateway').exists()
    skill_root = workspace / '.claude/skills/gpu-measurement'
    text = '\n'.join(path.read_text() for path in skill_root.rglob('*.md'))
    documented = set(re.findall(r'--kind ([a-z][a-z-]*)', text))
    help_kinds = runtime(selected).execute(None, '/v1/gateway/execute', {'argv': ['--help']})['result']['available_kinds']
    assert documented == set(help_kinds) == set(selected)
    assert all(word not in text for word in ('same_allocation_abba', 'preflight', 'aka execution', 'agate eval'))
    for kind in selected:
        schema = runtime(selected).execute(None, '/v1/gateway/execute', {'argv': ['--kind', kind, '--help']})
        assert schema['result']['request_schema']['cli_options']['--kind'] == {'const': kind}
    # The child-process resource selection must contain the same scoped Skill,
    # not rediscover the provider's unfiltered manuals.
    from aka.resources.contributions import SELECTION_ENV
    inherited = json.loads(registry.environment(workspace)[SELECTION_ENV])
    assert {row['name'] for row in inherited['skills']} == names
    assert next(row['source'] for row in inherited['skills'] if row['name'] == 'gpu-measurement') == str(skill_root.resolve())
    assert registry.install(workspace).module_skills == installation.module_skills
