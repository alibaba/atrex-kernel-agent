import json
import unittest
from unittest.mock import Mock, patch
from aka.execution.gateway import sdk
from aka.execution.gateway.sandbox import GatewayHTTPError
from atrex_gateway_client.errors import GatewayError
from aka.execution.gateway import sandbox
from aka.contracts.execution import ExecutionDiagnostic, ExecutionFacts, JobResult
from aka.evaluation.response_policy import execution_error

class SDKIntegrationTest(unittest.TestCase):
    def test_public_sdk_upload_submit_poll_cancel_and_environment(self):
        api=Mock()
        with patch.object(sdk, 'client', return_value=api):
            sdk.request('endpoint','POST','/v1/uploads',{'kind':'eval','spec':{'target_hardware':['L20N'],'num_gpus':2},'files':[{'path':'candidate.tar.gz','bytes':4,'sha256':'abc'}]},30)
            api.prepare_uploads.assert_called_once_with('L20N',[{'path':'candidate.tar.gz','bytes':4,'sha256':'abc'}],kind='eval',num_gpus=2)
            sdk.request('endpoint','POST','/v1/jobs/eval',{'idempotency_key':'stable-key'},30)
            api.submit_job.assert_called_once_with('eval',{'idempotency_key':'stable-key'})
            sdk.request('endpoint','GET','/v1/jobs/ev_test?wait=true',None,7)
            api.get_job.assert_called_once_with('ev_test',wait=False,timeout=7)
            sdk.request('endpoint','POST','/v1/jobs/ev_test/cancel',None,7)
            api.cancel_job.assert_called_once_with('ev_test')
            sdk.request('endpoint','GET','/v1/env/L20N?force=true',None,7)
            api.get_env.assert_called_once_with('L20N',force=True)
            sdk.request('endpoint','GET','/v1/env/L20N/capabilities',None,7)
            api.get_capabilities.assert_called_once_with('L20N',force=False)

    def test_sdk_rejection_keeps_status_and_nested_details(self):
        api=Mock()
        payload={'reason':'request_validation_failed','details':{'validation_errors':[{'msg':'invalid field'}]}}
        error=GatewayError(422,payload)
        api.submit_job.side_effect=error
        with patch.object(sdk,'client',return_value=api):
            with self.assertRaises(GatewayHTTPError) as found:
                sdk.request('endpoint','POST','/v1/jobs/eval',{},30)
        self.assertEqual(found.exception.status,422)
        self.assertEqual(json.loads(found.exception.detail)['error'],payload)
        self.assertEqual(api.submit_job.call_count,1)

    def test_accepted_job_short_poll_recovers_transient_query_failure(self):
        replies=[GatewayHTTPError(503,'temporary'),{'job_id':'ev_test','status':'running'},
                 {'job_id':'ev_test','status':'succeeded','result':{'exit_code':0}}]
        with patch.object(sandbox,'_gateway_json',side_effect=replies) as query, \
             patch.object(sandbox.time,'monotonic',return_value=0), patch.object(sandbox.time,'sleep'):
            result=sandbox._wait_sdk_job(url='endpoint',kind='dev',job_id='ev_test',deadline=10,started=0)
        self.assertEqual(result.returncode,0)
        self.assertEqual(query.call_count,3)
        self.assertTrue(all(call.args[1]=='GET' and '/ev_test?wait=false' in call.args[2]
                            for call in query.call_args_list))

    def test_wait_budget_returns_timeout_without_claiming_remote_failure(self):
        with patch.object(sandbox.time,'monotonic',return_value=10), \
             patch.object(sandbox,'_gateway_json') as query, patch.object(sandbox,'emit_diagnostic') as diagnostic:
            process=sandbox._wait_sdk_job(url='endpoint',kind='dev',job_id='ev_private',deadline=5,started=0)
        query.assert_not_called()
        diagnostic.assert_called_once_with({'failure_class':'timeout','reason':'original-job-wait-incomplete'})
        self.assertEqual(json.loads(process.stdout)['status'],'unknown') # Private remote observation remains honest.
        receipt=JobResult(process.returncode,process.stdout,process.stderr,transport_status='timed_out',
            facts=ExecutionFacts(disposition='unknown'),
            diagnostics=(ExecutionDiagnostic('timeout','original-job-wait-incomplete'),))
        public=execution_error(receipt)
        self.assertEqual(public['outcome'],'timeout')
        self.assertEqual(public['error']['code'],'gateway_wait_timeout')
        self.assertNotIn('ev_private',json.dumps(public))
