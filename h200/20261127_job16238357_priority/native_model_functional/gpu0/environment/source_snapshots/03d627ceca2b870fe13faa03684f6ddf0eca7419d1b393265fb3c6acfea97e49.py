#!/usr/bin/env python3
"""CPU regressions for mask semantics and cross-request common-prefix replay."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import tempfile

import numpy as np

from diagnose_decode_correctness_v4 import discrepancy,forced_logits
from software_decode_experiments import make_decode


class Tensor(np.ndarray):
    @property
    def device(self):return 'cuda'
    def float(self):return self
    def cpu(self):return self
    def masked_fill(self,mask,value):
        result=self.copy();result[mask]=value;return result
    def unsqueeze(self,index):return np.expand_dims(self,index)


def tensor(value,**kwargs):return np.asarray(value).view(Tensor)


class RecoveryContracts(unittest.TestCase):
    def torch(self):
        return SimpleNamespace(arange=lambda n,**kw:tensor(range(n)),tensor=tensor,long=np.int64,
            zeros=lambda shape,**kw:np.zeros(shape,dtype=kw['dtype']).view(Tensor),finfo=np.finfo)

    def test_eager_mask_excludes_unused_slots_and_sdpa_stays_boolean(self):
        for backend in ('eager','sdpa'):
            calls=[]
            class Model:
                dtype=np.float32
                config=SimpleNamespace(_attn_implementation=backend)
                def __call__(self,**kw):
                    calls.append(kw)
                    return SimpleNamespace(logits=tensor([[[0.,1.]]]))
            forward=make_decode(Model(),object(),4,self.torch(),False)
            forward(tensor([[3]]),tensor([1]))
            mask=calls[0]['attention_mask'].reshape(-1)
            if backend=='sdpa':
                self.assertEqual(mask.dtype,np.bool_);self.assertEqual(mask.tolist(),[True,True,False,False])
            else:
                # Large values in unused cache slots must have zero softmax mass.
                score=tensor([0.,0.,100.,100.])+mask
                weights=np.exp(score-score.max());weights/=weights.sum()
                np.testing.assert_array_equal(weights,tensor([.5,.5,0.,0.]))

    def test_builtin_mask_is_delegated(self):
        class Model:
            def __call__(self,**kw):
                self.mask=kw['attention_mask'];return SimpleNamespace(logits=tensor([[[0.,1.]]]))
        model=Model();make_decode(model,object(),4,self.torch(),False,mask_kind='builtin')(tensor([[1]]),tensor([2]))
        self.assertIsNone(model.mask)

    def test_discrepancy_is_earliest_across_requests(self):
        self.assertEqual(discrepancy([[1,2,3],[4,5,6]],[[1,2,0],[4,0,6]]),
                         {'request':1,'token_index':1,'reference':5,'actual':0})
        with self.assertRaises(AssertionError):discrepancy([[1]],[[1,2]])

    def test_forced_replay_uses_all_reference_tokens_not_own_argmax(self):
        calls=[];prefills=[]
        class Model:
            def __call__(self,**kw):
                prefills.append(kw)
                return SimpleNamespace(past_key_values='cache',logits=tensor([[[99.,0.]],[[99.,0.]]]))
        class Cache:
            resets=0
            def reset(self):self.resets+=1
        def decode(tokens,position):
            calls.append((tokens.tolist(),position.tolist()))
            return tensor([[0],[0]]),tensor([[[99.,0.]],[[99.,0.]]])
        cache=Cache()
        forced_logits(Model(),cache,decode,tensor([[8,9],[6,7]]),[[3,4,5],[11,12,13]],2,self.torch())
        self.assertEqual(calls,[([[3],[11]],[2]),([[4],[12]],[3])]);self.assertEqual(cache.resets,1)
        calls.clear()
        forced_logits(Model(),cache,decode,tensor([[8,9],[6,7]]),[[3,4,5],[11,12,13]],0,self.torch())
        self.assertEqual(calls,[])

    def test_plan_has_single_allocation_prerequisite_and_no_retries(self):
        root=Path(__file__).resolve().parents[1]
        plan=json.loads((root/'configs/h200_batch_failure_recovery_v4.json').read_text())
        s=plan['session_overrides']
        self.assertEqual(s['failure_diagnostics_version'],'failure_recovery_v4')
        self.assertIsNone(s['decoder_calibration_version'])
        self.assertTrue(s['required_graph_pilot_current_allocation'])
        self.assertEqual(s['required_graph_pilot_label'],'D4_graph_clock')
        self.assertEqual(s['rejected_acquisition_retry_limit'],0)

    def test_failed_diagnostic_and_exact_gate_block_calibration(self):
        from recover_h200_failures import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);session=root/'session.json';session.write_text('{}')
            def dump(path,data):
                path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(data))
            env=SimpleNamespace(SESSION={'failure_diagnostics_version':'failure_recovery_v4','frozen_source_root':directory},
                PACKAGE=root,PYTHON='python',UUID='test',CUDA=root,dump=dump,
                run=Mock(return_value={'status':'failed'}),python_command=lambda *args:list(args),verify_allocation=lambda:None)
            campaign=SimpleNamespace(checkpoint=Mock(),make_config=lambda *a:root/'config.json',
                acquisition=lambda *a,**kw:{'status':'passed'},update_job=Mock())
            jobs=SimpleNamespace(load=lambda *a:{},functional_smoke_reference=Mock(side_effect=AssertionError('HF/engine smoke output mismatch')))
            calibration=Mock()
            modules={'h200_session':env,'shoreline_campaign':campaign,'shoreline_jobs':jobs,
                'shoreline_treatment_analysis':SimpleNamespace(analyze=Mock()),
                'shoreline_analyze':SimpleNamespace(main=Mock()),
                'calibrate_h200_decoder':SimpleNamespace(main=calibration)}
            with patch.dict('sys.modules',modules),patch.dict('os.environ',{}),patch('recover_h200_failures.subprocess.run'):
                main(session)
            calibration.assert_not_called()
            disposition=json.loads((root/'results/decoder_calibration_disposition.json').read_text())
            self.assertEqual(disposition['scientific_status'],'blocked_before_acquisition')
            self.assertEqual(disposition['calibration_acquisitions'],0)
            self.assertEqual(env.run.call_count,2)  # bounded B8/B1 diagnostics; no clock/sweep launches


if __name__=='__main__':unittest.main()
