#!/usr/bin/env python3
"""Calibration leakage, qualification, scope and uncertainty negative controls."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from decoder_calibration_model import CONDITIONS,fit,predict,validate,roles,parameter_uncertainty,full_decode_publication


def examples():
    result=[]
    for i,(name,b,c,split) in enumerate(CONDITIONS):
        compute=.3+.25*i;memory=1.2+.1*i
        t=compute+memory-.37*min(compute,memory)+.2
        r={'id':name,'split':split,'qualified':True,'C_ms':compute,'M_ms':memory,'T_ms':t}
        for k,v in [('C',compute),('M',memory),('T',t)]:r[k+'_acquisitions']=[[v]*10 for _ in range(3)]
        result.append(r)
    return result


class CalibrationContracts(unittest.TestCase):
    def test_frozen_train_only_model_predicts_new_conditions(self):
        rows=examples();model=fit(rows[:4])
        self.assertAlmostEqual(model['eta'],.37,places=3)
        self.assertAlmostEqual(model['O_ms'],.2,places=3)
        self.assertTrue(model['training_gate'])
        self.assertEqual(validate(model,rows[4:],'frozen-sha')['status'],'validated_for_declared_conditioned_graph_family')
        with self.assertRaises(AssertionError):fit(rows)
        with self.assertRaises(AssertionError):validate(model,rows[4:5],'frozen-sha')
        with self.assertRaises(AssertionError):validate(model,rows[4:],'')

    def test_heldout_failure_and_qualification_rejected(self):
        rows=examples();model=fit(rows[:4]);rows[4]['T_ms']*=1.3
        self.assertEqual(validate(model,rows[4:],'sha')['status'],'rejected')
        rows=examples();rows[4]['qualified']=False
        self.assertEqual(validate(model,rows[4:],'sha')['status'],'rejected')
        rows[0]['qualified']=False
        with self.assertRaises(AssertionError):fit(rows[:4])
        rows=examples();rows[0]['T_ms']=10
        self.assertFalse(fit(rows[:4])['training_gate'])

    def test_reference_work_covers63_steps_and_full_output_head(self):
        cfg={'hidden_size':4,'intermediate_size':8,'num_attention_heads':2,'num_key_value_heads':1,
            'num_hidden_layers':2,'vocab_size':100}
        work=roles(cfg,2,10)
        linear=sum(2*m*k*n*count for role,m,k,n,count in work if not role.startswith('attention_'))
        self.assertEqual(linear,63*(2*(128+64+256+128)+1600))
        attention=sum(2*m*k*n*count for role,m,k,n,count in work if role.startswith('attention_'))
        self.assertEqual(attention,63*4*2*2*2*2*(10+32))

    def test_uncertainty_reproducible_and_short_trials_rejected(self):
        rows=examples()[:4]
        a=parameter_uncertainty(rows,draws=10);b=parameter_uncertainty(rows,draws=10)
        self.assertEqual(a,b)
        self.assertTrue(all(0<=p['eta']<=1 and p['O_ms']>=0 for p in a['draws']))
        rows[0]['C_acquisitions'][0].pop()
        with self.assertRaises(AssertionError):parameter_uncertainty(rows,draws=1)

    def test_three_tick_or_wrong_output_counter_publication_cannot_join(self):
        reference={'prompt_sha256':'prompt','trials':[{'output_ids':[[1]*64]}]}
        p={'full_decode_calibration':True,'decode_iterations':63,'timed_generated_tokens':63,
            'prompt_sha256':'prompt','output_ids':[[1]*64],'prefix':'B1_measured0','prefill_barrier':{'passed':True},
            'post_prefill_conditioning':{'protocol':'memory_settle_v3'}}
        self.assertTrue(full_decode_publication(p,reference,1))
        self.assertFalse(full_decode_publication({**p,'decode_iterations':3},reference,1))
        self.assertFalse(full_decode_publication({**p,'prefix':'B1_warmup0'},reference,1))
        self.assertFalse(full_decode_publication({**p,'output_ids':[[2]*64]},reference,1))
        self.assertFalse(full_decode_publication({**p,'post_prefill_conditioning':None},reference,1))

    def test_full_counter_range_once_and_complete_native_outputs(self):
        import controlled_vllm_experiments as baseline
        pushes=[];pops=[]
        class Event:
            def __init__(self,**kw):pass
            def record(self):pass
            def elapsed_time(self,other):return 63.0
        cuda=SimpleNamespace(Event=Event,synchronize=lambda:None,
            nvtx=SimpleNamespace(range_push=pushes.append,range_pop=lambda:pops.append(1)),
            profiler=SimpleNamespace(start=lambda:None,stop=lambda:None))
        class Engine:
            _h200_calibration_full_range=True
            _h200_predecode_condition=staticmethod(lambda:{'protocol':'memory_settle_v3'})
            def __init__(self):self.active=False;self.tick=0
            def has_unfinished_requests(self):return self.active
            def add_request(self,rid,prompt,sampling):self.rid=rid;self.prompt=prompt['prompt_token_ids'];self.active=True;return rid
            def step(self):
                self.tick+=1;self.active=self.tick<64
                return [SimpleNamespace(request_id=self.rid,num_cached_tokens=0,finished=not self.active,
                    prompt_token_ids=self.prompt,outputs=[SimpleNamespace(token_ids=list(range(self.tick)),finish_reason='length')])]
        torch=SimpleNamespace(cuda=cuda);sampling=SimpleNamespace(max_tokens=64,logprobs=None)
        with tempfile.TemporaryDirectory() as directory:
            for i in range(6):
                prefix='B1C128_'+('warmup'+str(i) if i<3 else 'measured'+str(i-3))
                baseline.run_trial(Engine(),[[5,6]],sampling,prefix,torch,Path(directory),profile_capture=i==3,
                    hardware_sampler=lambda:{'sm_clock_mhz':1980,'memory_clock_mhz':3201})
            p=json.loads((Path(directory)/'profile_interval_output.json').read_text())
        self.assertEqual(pushes.count('calibration_decode_selected'),1)
        self.assertEqual(len(pushes),len(pops))
        self.assertEqual(p['decode_iterations'],63);self.assertEqual(len(p['output_ids'][0]),64)
        self.assertTrue(p['full_decode_calibration']);self.assertTrue(p['prefix'].endswith('_measured0'))


if __name__=='__main__':unittest.main()
