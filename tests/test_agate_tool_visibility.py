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
