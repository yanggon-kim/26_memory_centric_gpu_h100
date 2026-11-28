#!/usr/bin/env python3
"""CPU contracts for targeted H200 counter-scope and concurrency repairs."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

import bootstrap_h200_batch as batch
import shoreline_queue
from shoreline_timeline import competition_gate


class GapContracts(unittest.TestCase):
    def test_remaining_pressure_only_collects_three_new_captures(self):
        import shoreline_jobs as jobs
        plan=json.loads((Path(__file__).resolve().parents[1]/'configs/h200_batch_gap_remaining_v2.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch.write(root/'session.json',plan['session_overrides'])
            calls=[]
            def counter(*args,**kw):
                calls.append((args,kw));return {'status':'complete'}
            with patch.multiple(jobs,ROOT=root,campaign=SimpleNamespace(update_job=lambda *a:None),
                reference=lambda *a:Path('functional-proof'),counter_case=counter,
                treatment_case=lambda *a,**kw:self.fail('repeated ordinary timing'),
                dump=batch.write,publish=lambda *a:None,disposition=lambda *a:None,create=True):
                jobs.retention()
            self.assertEqual(len(calls),3)
            self.assertEqual([a[0][4] for a in calls],[0,1,2])
            self.assertTrue(all(a[0][2:4]==(8,4096) and a[1]['pressure']=='pressure' for a in calls))
            self.assertEqual(plan['session_overrides']['gpu_job_order'],['J7'])

    def test_failed_first_repaired_counter_stops_expansion_and_skips_timing(self):
        import shoreline_jobs as jobs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch.write(root/'session.json',{'retention_components':['traffic'],
                'retention_shapes':[[1,1024],[8,4096]]})
            calls=[];dispositions=[]
            def counter(*args,**kw):calls.append((args,kw));return {'status':'failed'}
            with patch.multiple(jobs,ROOT=root,campaign=SimpleNamespace(update_job=lambda *a:None),
                reference=lambda *a:Path('functional-proof'),counter_case=counter,
                treatment_case=lambda *a,**kw:self.fail('traffic-only repair launched ordinary timing'),
                dump=batch.write,publish=lambda *a:None,disposition=lambda *a:dispositions.append(a),create=True):
                jobs.retention()
            self.assertEqual(len(calls),1)
            self.assertEqual(dispositions[-1][0:2],('J7','partial'))

    def test_explicit_repair_queue_does_not_repeat_completed_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('environment/source_snapshots','logs','results'):(root/name).mkdir(parents=True)
            batch.write(root/'session.json',{'launch_cutoff':'2099-01-01T00:00:00+00:00',
                'gpu_job_order':['J6','J7'],'queue_reused_jobs':{'J5':{},'J8':{},'J10':{},'J11_gpu':{}}})
            calls=[]
            with patch.object(shoreline_queue.subprocess,'call',side_effect=lambda c,**kw:calls.append(c) or 0):
                shoreline_queue.main(SimpleNamespace(session=root/'session.json',start_at='J5',schedule='gpu-first',phase='all'))
            self.assertEqual([c[c.index('--job')+1] for c in calls if '--job' in c],['J6','J7'])

    def test_v5_replay_rejects_warmup_interval_attribution(self):
        from shoreline_evidence import qualify_replays
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'run';output=base/'pass000';output.mkdir(parents=True)
            expected={'prompt_sha256':'prompt','trials':[{'output_ids':[[1,2,3]]}]}
            batch.write(base/'pass000.json',{'status':'complete','output':str(output)})
            batch.write(output/'run.json',{'status':'complete','config':{'shoreline_protocol':{'counter_range_protocol':'ticks_only_v5'}},
                'cases':[{'prompt_sha256':'prompt','trials':expected['trials']*3}]})
            interval={'status':'complete','ticks':[4,5,6],'trial':'B1C1_warmup0'}
            batch.write(output/'selected_interval.json',interval)
            self.assertFalse(qualify_replays(base,expected,root)['passed'])
            interval['trial']='B1C1_measured0';batch.write(output/'selected_interval.json',interval)
            self.assertTrue(qualify_replays(base,expected,root)['passed'])

    def test_competition_requires_clean_exit_exact_outputs_and_each_kernel_role(self):
        a={'decoder':{'green_context_id':2},'competitor':{'green_context_id':3},
           'disjoint_api_resources':True,'observed_smids':[[0,1],[2,3]]}
        t={'windows':[{'green_context_ids':[2,3],'distinct_green_context_kernel_overlap_ms':1,
            'kernels':{'transfer':{'green_context_ids':[3]},'decoder_matmul':{'green_context_ids':[2]}}}]}
        self.assertTrue(competition_gate(t,a,True,True))
        self.assertFalse(competition_gate(t,a,False,True))
        self.assertFalse(competition_gate(t,a,True,False))
        for symbol,ids in [('transfer',[2]),('decoder_matmul',[0]),('decoder_matmul',[2,3])]:
            broken=copy.deepcopy(t);broken['windows'][0]['kernels'][symbol]['green_context_ids']=ids
            self.assertFalse(competition_gate(broken,a,True,True))
        broken=copy.deepcopy(t);broken['windows'][0]['distinct_green_context_kernel_overlap_ms']=0
        self.assertFalse(competition_gate(broken,a,True,True))
        broken=copy.deepcopy(a);broken['observed_smids']=[[0,1],[1,2]]
        self.assertFalse(competition_gate(t,broken,True,True))

    def test_repair_reopens_old_dispositions_but_preserves_completed_repair(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(batch,'PACKAGE',Path(directory)):
            root=Path(directory)/'h200/old';batch.write(root/'session.json',{'collection_scope_version':'old'})
            proof={'path':'h200/old/results/campaign_queue.json'}
            plan={'reopen_gpu_jobs':['J6','J7'],'session_overrides':{'collection_scope_version':'repair_v2'}}
            reused={'J6':proof,'J7':proof,'J8':proof}
            self.assertEqual(set(batch.apply_reopen_policy(reused,plan)),{'J6','J7'})
            self.assertEqual(set(reused),{'J8'})
            batch.write(root/'session.json',{'collection_scope_version':'repair_v2'})
            reused={'J6':proof};self.assertEqual(batch.apply_reopen_policy(reused,plan),{})
            self.assertEqual(set(reused),{'J6'})

    def test_pressure_profiles_only_first_measured_trial_not_warmups_or_later_trials(self):
        import shoreline_decode
        pushes=[];pops=[];published=[]
        class Event:
            def __init__(self,**kw):pass
            def record(self):pass
            def elapsed_time(self,other):return 1
        cuda=SimpleNamespace(Event=Event,synchronize=lambda:None,
            nvtx=SimpleNamespace(range_push=pushes.append,range_pop=lambda:pops.append(1)),
            profiler=SimpleNamespace(start=lambda:None,stop=lambda:None))
        torch=ModuleType('torch');torch.cuda=cuda
        baseline=ModuleType('controlled_vllm_experiments')
        worker=ModuleType('shoreline_worker');worker.ACTIVE_WORKERS=[];worker.install_ranges=lambda *a:None
        metadata=ModuleType('h200_worker_metadata')
        software=ModuleType('software_decode_experiments');software.dump=lambda p,d:published.append((str(p),copy.deepcopy(d)))
        vllm=ModuleType('vllm');vllm.LLM=type('LLM',(),{})
        class Engine:
            def __init__(self):self.tick=0
            def step(self):
                self.tick+=1
                return [SimpleNamespace(request_id='one',outputs=[SimpleNamespace(token_ids=[1]*self.tick)])]
        def original(engine,prompts,sampling,*args,**kwargs):
            for _ in range(64):engine.step()
            return {}
        baseline.run_trial=original;baseline.cleanup_owned_engine=lambda *a:None
        def run(a):
            for index in range(3):
                baseline.run_trial(Engine(),[[1]],SimpleNamespace(max_tokens=64),f'B1C1_warmup{index}',torch,a.output,profile_capture=False)
            for index in range(3):
                baseline.run_trial(Engine(),[[1]],SimpleNamespace(max_tokens=64),f'B1C1_measured{index}',torch,a.output,profile_capture=index==0)
        baseline.main=run
        modules={'torch':torch,'controlled_vllm_experiments':baseline,'shoreline_worker':worker,
                 'h200_worker_metadata':metadata,'software_decode_experiments':software,'vllm':vllm}
        with tempfile.TemporaryDirectory() as directory,patch.dict('sys.modules',modules),patch.dict('os.environ'):
            shoreline_decode.main(SimpleNamespace(output=Path(directory),pressure='normal',external_profiler=True))
        self.assertEqual(pushes.count('shoreline_ticks_4_6'),1)
        self.assertEqual(len(pushes),len(pops))
        self.assertEqual(published[-1][1]['trial'],'B1C1_measured0')
        self.assertEqual(published[-1][1]['status'],'complete')


if __name__=='__main__':unittest.main()
