"""CPU supervision/admission contracts; subprocess fixtures never use GPUs."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from h200_connector_capture_campaign import admission,profiled_command,run_profile_pair,PAIR_BOUND


class Controller(unittest.TestCase):
    def inputs(self,available=4000):
        now=time.time();cutoff=now+available;host='node.domain';job='123'
        sessions=[{'gpu_uuid':'GPU-'+str(i),'python':sys.executable,'cuda_home':'/cuda',
            'slurm_job_id':job,'expected_hostname':host,'frozen_source_identity':'same',
            'launch_cutoff':dt.datetime.fromtimestamp(cutoff,dt.timezone.utc).isoformat()} for i in (0,1)]
        end=dt.datetime.fromtimestamp(cutoff+900,dt.timezone.utc).isoformat()
        raw=f'JobId=123 JobState=RUNNING UserId=user({os.getuid()}) NodeList=node AllocTRES=gres/gpu:h200=2 EndTime={end}'
        pair={'job_id':job,'hostname':host,'protocol':'native_model_KV_long_context_functional_v2',
            'context':16384,'generated_tokens':64,'roles':{'producer':{'gpu_uuid':'GPU-0'},'consumer':{'gpu_uuid':'GPU-1'}}}
        return [sessions,[{'status':'idle'}]*2,job,host,os.getuid(),raw,
            {'status':'functional_prerequisites_passed'},
            {'status':'native_model_functional_review_passed','errors':[],'job_id':job},pair,now]

    def test_complete_budget_and_original_functional_gates_are_required(self):
        args=self.inputs();self.assertGreater(admission(*args)['available_s'],1800)
        with self.assertRaisesRegex(AssertionError,'budget does not fit'):admission(*self.inputs(1700))
        for change in ('busy','failed','wrong_job','wrong_pair'):
            args=self.inputs()
            if change=='busy':args[1]=[{'status':'running'},{'status':'idle'}]
            if change=='failed':args[6]={'status':'partial'}
            if change=='wrong_job':args[2]='124'
            if change=='wrong_pair':args[8]['roles']['consumer']['gpu_uuid']='GPU-foreign'
            with self.subTest(change=change),self.assertRaises(AssertionError):admission(*args)

    def test_profile_wrapper_preserves_supervisor_uuid_and_numeric_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'session.json').write_text(json.dumps({'cuda_home':'/cuda'}))
            def original(base,label,entry,args,timeout):
                return ['py','supervisor.py','--gpu','--timeout',str(timeout),'--','py','frozen/'+entry,*map(str,args)],{'CUDA_VISIBLE_DEVICES':'GPU-owned'}
            with patch('h200_connector_capture_campaign.supervised_command',side_effect=original):
                command,env=profiled_command(root,root/'pair.json','producer','GPU-owned')
            split=command.index('--')+1
            self.assertEqual(command[split:split+2],['/cuda/bin/nsys','profile'])
            self.assertIn('--capture-range=cudaProfilerApi',command)
            self.assertIn('--cuda-graph-trace=node',command)
            self.assertEqual(command[command.index('--gpu-uuid')+1],'GPU-owned')
            self.assertIn('frozen/h200_native_connector_capture.py',command)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-owned')
            self.assertEqual(command[command.index('--timeout')+1],str(PAIR_BOUND))

    def test_short_pair_bound_never_spawns(self):
        with patch('h200_connector_capture_campaign.subprocess.Popen') as spawn:
            with self.assertRaises(AssertionError):run_profile_pair(Path('/unused'),{},time.time()+500)
            spawn.assert_not_called()

    def test_failed_actor_stops_its_waiting_peer(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'logs').mkdir()
            for role in ('producer','consumer'):(root/role/'results').mkdir(parents=True)
            pair={'roles':{'producer':{'gpu_uuid':'GPU-a'},'consumer':{'gpu_uuid':'GPU-b'}}}
            children=[];real_spawn=subprocess.Popen
            def command(base,*args):
                code='raise SystemExit(9)' if base.name=='producer' else 'import time; time.sleep(60)'
                return [sys.executable,'-c',code],{}
            def spawn(*args,**kwargs):
                child=real_spawn(*args,**kwargs);children.append(child);return child
            started=time.time()
            try:
                with patch('h200_connector_capture_campaign.profiled_command',side_effect=command),patch('h200_connector_capture_campaign.subprocess.Popen',side_effect=spawn):
                    with self.assertRaisesRegex(AssertionError,'profiled role failed'):run_profile_pair(root,pair,time.time()+1000)
                self.assertLess(time.time()-started,5)
                self.assertTrue(all(c.poll() is not None for c in children))
            finally:
                for child in children:
                    if child.poll() is None:child.kill();child.wait()


if __name__=='__main__':unittest.main()
