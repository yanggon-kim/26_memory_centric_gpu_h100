import tempfile
import json
import copy
from pathlib import Path
import types
import unittest
from unittest.mock import patch
from h200_resident_campaign import acquire
from h200_resident_sweep import conditions


class Failure(unittest.TestCase):
    def test_failed_conditioned_producer_preserves_config_and_all_missing_points(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            worker=types.SimpleNamespace(root=root,script=Path('/CPU-fake/scripts'),py='CPU-fake',
                uuid='GPU-CPU-fake',settings={'frozen_source_identity':'CPU-fake'},expected=None,
                execute=lambda *args,**kwargs:{'status':'failed','elapsed_s':1.})
            with patch('h200_resident_campaign.limitation'):
                rows,proc,good=acquire(worker,0,conditioned=True)
            self.assertFalse(good);self.assertEqual(len(rows),9)
            self.assertTrue(all(r['status']=='failed' and not r['qualified'] for r in rows))
            self.assertTrue(all(r['conditioning']['protocol']=='memory_settle_v3' for r in rows))
            self.assertTrue((root/'configs/qwen32b_eager_b1_c4096_t10_memory_settle_v3.json').exists())

    def captured(self,corrupt=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            receipt={'protocol':'memory_settle_v3','copies':64,'buffer_bytes':256*1024**2,'wall_ms':1001.,'gpu_event_ms':999.}
            trial={'output_ids':[[7]*64],'wall_ms':1260.,'cuda_event_ms':1250.,'tokens_per_s':50.,
                'decode_iterations':63,'prefill_barrier':{'passed':True},'post_prefill_conditioning':receipt}
            def execute(label,command,**kwargs):
                output=Path(command[command.index('--output')+1]);output.mkdir(parents=True)
                cfg=json.loads(Path(command[command.index('--config')+1]).read_text())
                records={}
                for name,fraction,green in conditions():
                    sub=output/name;sub.mkdir()
                    case={'generated_tokens':64,'warmup_output_ids':[[[7]*64] for _ in range(3)],
                          'output_digests_identical':True,'trials':[copy.deepcopy(trial) for _ in range(10)]}
                    (sub/'run.json').write_text(json.dumps({'status':'complete','config':cfg,'cases':[case]}))
                    records[name]={'allocation':{'actual_sm_count':{1.:132,.85:112,.75:96,.6:80,.45:56,.3:40}[fraction]},
                                   'physical_sm_count':132,'warmups':[copy.deepcopy(trial) for _ in range(3)]}
                    if corrupt and name=='green40':records[name]['warmups'][-1].pop('post_prefill_conditioning')
                (output/'resident_sweep.json').write_text(json.dumps({'status':'complete','conditions':records}))
                return {'status':'complete','elapsed_s':1.,'monitor':{'source':'CPU-fake'}}
            worker=types.SimpleNamespace(root=root,script=Path('/CPU-fake/scripts'),py='CPU-fake',uuid='GPU-CPU-fake',
                settings={'frozen_source_identity':'CPU-fake'},expected=None,execute=execute)
            with patch('h200_resident_campaign.limitation'),patch('shoreline_campaign.clock_qualification',return_value={'passed':True}):
                rows,proc,good=acquire(worker,0,conditioned=True)
            return rows,good

    def test_complete_conditioned_acquisition_requires_all_warm_and_timed_receipts(self):
        rows,good=self.captured()
        self.assertTrue(good);self.assertTrue(all(r['conditioning_scope_verified'] and r['qualified'] for r in rows))
        rows,good=self.captured(corrupt=True)
        self.assertFalse(good)
        failed=next(r for r in rows if r['actual_sm_count']==40)
        self.assertFalse(failed['conditioning_scope_verified']);self.assertFalse(failed['qualified'])


if __name__=='__main__':unittest.main()
