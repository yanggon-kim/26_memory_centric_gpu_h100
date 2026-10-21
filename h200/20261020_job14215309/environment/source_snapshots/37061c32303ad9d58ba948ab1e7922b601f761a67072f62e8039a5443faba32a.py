#!/usr/bin/env python3
"""CPU gates for the separately labelled failure-recovery protocol."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch

from decode_precondition import create,validate
from diagnose_decode_correctness import first_difference
from recover_h200_failures import exact


class FailureContracts(unittest.TestCase):
    def test_memory_workload_is_bounded_and_uses_sm_kernel(self):
        class Kernel:
            def __getitem__(self,grid):return lambda *a:calls.append((grid,a))
        calls=[]
        event=SimpleNamespace(record=lambda:None,elapsed_time=lambda other:1000.0)
        stream=SimpleNamespace(synchronize=lambda:None)
        torch=SimpleNamespace(float32='f32',int32='i32',ones=lambda *a,**kw:object(),empty_like=lambda *a:object(),
            empty=lambda *a,**kw:object(),cuda=SimpleNamespace(synchronize=lambda:None,
            current_stream=lambda:stream,Event=lambda **kw:event))
        config={'protocol':'memory_settle_v3','seconds':1.0,'buffer_bytes':256*1024**2,'scope':'conditioned_decode_only'}
        with patch.dict('sys.modules',{'triton':SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),
            'shoreline_kernels':SimpleNamespace(transfer=Kernel())}):
            condition=create(torch,config)
            with patch('decode_precondition.time.perf_counter',side_effect=[0,0,1.1,1.2]):row=condition()
        self.assertEqual(row['copies'],32);self.assertEqual(len(calls),33)
        self.assertIsNot(calls[0][1][0],calls[0][1][1])
        self.assertIn('changes cache history',row['scope'])
        with self.assertRaises(AssertionError):validate({**config,'seconds':0.05})

    def test_prefill_failure_prevents_conditioning(self):
        import controlled_vllm_experiments as baseline
        condition=Mock()
        engine=SimpleNamespace(has_unfinished_requests=lambda:False,add_request=lambda *a:'id',
            step=lambda:[],_h200_predecode_condition=condition)
        torch=SimpleNamespace(cuda=SimpleNamespace(nvtx=SimpleNamespace(range_push=lambda *a:None,range_pop=lambda:None)))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError,'prefill barrier'):
                baseline.run_trial(engine,[[1]],SimpleNamespace(),'bad',torch,Path(directory))
        condition.assert_not_called()

    def test_clock_gate_not_relaxed_by_new_protocol(self):
        from shoreline_campaign import clock_qualification
        trial={'hardware_before':{'sm_clock_mhz':1830,'memory_clock_mhz':3201},
            'hardware_after':{'sm_clock_mhz':1980,'memory_clock_mhz':3201},
            'wall_start_unix_ns':1,'wall_end_unix_ns':2}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'monitor';p.write_text('')
            self.assertFalse(clock_qualification({'cases':[{'trials':[trial]}]},p)['passed'])

    def test_correctness_gate_requires_trials_and_warmup_identity(self):
        self.assertFalse(exact({'status':'complete','cases':[]},[[4,5]]))
        run={'status':'complete','cases':[{'trials':[{'output_ids':[[4,5]]}],
            'warmup_output_ids':[[[4,6]]]}]}
        self.assertFalse(exact(run,[[4,5]]))
        self.assertEqual(first_difference([[1,2,3]],[[1,4,3]])['token_index'],1)
        with self.assertRaises(AssertionError):first_difference([[1,2]],[[1]])

    def test_new_plan_retains_original_cohorts_and_gates(self):
        plan=json.loads((Path(__file__).resolve().parents[1]/'configs/h200_batch_failure_recovery_v3.json').read_text())
        cfg=plan['session_overrides'];validate(cfg['decode_conditioning'])
        self.assertEqual(cfg['competition_shapes'],[[8,4096]])
        self.assertEqual(cfg['rejected_acquisition_retry_limit'],0)
        self.assertEqual(plan['reopen_gpu_jobs'],[])
        self.assertEqual(len(cfg['retained_repair_sessions']),2)

    def test_compact_energy_keeps_raw_parent_and_estimators(self):
        import consolidate_h200 as consolidate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'results').mkdir()
            p=root/'results/shoreline_energy_qualification.json'
            p.write_text(json.dumps({'cases':[{'raw_sensors':[{'samples':[{'power_w':3}],
                'energy_counter_joules_per_token':2.0,'unavailable_reason':'clock rejected'}]}]}))
            with patch.object(consolidate,'PACKAGE',root.parent):target=consolidate.compact_energy(root)
            compact=json.loads(target.read_text());sensor=compact['cases'][0]['raw_sensors'][0]
            self.assertEqual(sensor['energy_counter_joules_per_token'],2.0)
            self.assertNotIn('samples',sensor)
            self.assertIn('samples',json.loads(p.read_text())['cases'][0]['raw_sensors'][0])
            self.assertEqual(compact['parent']['sha256'],consolidate.sha(p))


if __name__=='__main__':unittest.main()
