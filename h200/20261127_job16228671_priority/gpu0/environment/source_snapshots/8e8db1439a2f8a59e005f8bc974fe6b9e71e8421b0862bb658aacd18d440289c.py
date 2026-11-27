#!/usr/bin/env python3
"""Terminal accounting, retained corruption and interruption evidence contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h200_priority_campaign as campaign
import h200_review_allocation as review


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'environment').mkdir();(self.root/'analysis').mkdir();(self.root/'results').mkdir()
        self.account=self.root/'environment/final_slurm_accounting.psv'
        self.account.write_text('JobID|State|Start|End|ElapsedRaw|AllocTRES\n123|PREEMPTED|start|end|3600|cpu=16,gres/gpu=2,gres/gpu:h200=2\n')
        campaign.write(self.root/'environment/allocated_devices.json',{'slurm_job_id':'123','devices':[{'uuid':'GPU-a'},{'uuid':'GPU-b'}]})
        for i,uuid in enumerate(('GPU-a','GPU-b')):
            sub=self.root/f'gpu{i}'
            for folder in ('results/processes','runs/test','analysis'):(sub/folder).mkdir(parents=True)
            campaign.write(sub/'session.json',{'gpu_uuid':uuid,'frozen_source_identity':'frozen'})
        self.sub=self.root/'gpu0';self.run=self.sub/'runs/test/run.json'
        self.run.write_text('{"status":"complete"}\n')
        self.row={'id':'test','engine':'graph','fraction':1.,'batch':1,'context':4096,
                  'pilot':False,'stock':True,'acquisition':0,'suffix':'','status':'complete','qualified':True,
                  'gpu_uuid':'GPU-a','source_identity':'frozen','run':str(self.run),'run_sha256':campaign.sha(self.run)}
        campaign.write(self.sub/'results/test.json',self.row)
        self.process={'label':'test','status':'complete','source_freeze_identity':'frozen','monitor':{'failure':None}}
        campaign.write(self.sub/'results/processes/test.json',self.process)

    def tearDown(self):self.tmp.cleanup()

    def test_accounting_requires_terminal_and_actual_gpu_count(self):
        a=review.accounting(self.account,'123',2)
        self.assertEqual(a['actual_allocated_gpu_hours'],2.)
        self.account.write_text(self.account.read_text().replace('PREEMPTED','RUNNING'))
        with self.assertRaises(AssertionError):review.accounting(self.account,'123',2)
        self.account.write_text(self.account.read_text().replace('RUNNING','CANCELLED by 42'))
        self.assertEqual(review.accounting(self.account,'123',2)['state'],'CANCELLED')
        with self.assertRaises(AssertionError):review.accounting(self.account,'123',1)

    def test_checksum_change_rejects_without_discarding_record(self):
        records,issues,_=review.case_records(self.sub)
        self.assertTrue(records[0]['qualified']);self.assertFalse(issues)
        self.run.write_text('{"status":"changed"}\n')
        records,issues,_=review.case_records(self.sub)
        self.assertEqual(len(records),1);self.assertFalse(records[0]['qualified']);self.assertTrue(issues)
        self.assertTrue(campaign.load(self.sub/'results/test.json')['qualified'])

    def test_interruption_preserves_original_producer(self):
        self.process.pop('status');campaign.write(self.sub/'results/processes/test.json',self.process)
        original=(self.sub/'results/processes/test.json').read_bytes()
        records,issues,interrupted=review.case_records(self.sub)
        self.assertFalse(records[0]['qualified']);self.assertTrue(issues)
        self.assertEqual(interrupted[0]['label'],'test')
        self.assertEqual((self.sub/'results/processes/test.json').read_bytes(),original)

    def test_native_source_must_match_across_counts_not_only_within_each_count(self):
        import h200_analyze_priority_partial as analysis
        clocks={'passed':True,'clocks':{m:{'minimum':100.,'maximum':100.} for m in ('sm_clock_mhz','memory_clock_mhz')}}
        for variant in ('same','source_changed','freeze_changed'):
            for fraction in campaign.FRACTIONS:
                count=round(fraction*132)
                changed=fraction==.85
                for acq in range(3):
                    label=f'native_f{round(fraction*100)}_a{acq}'
                    data={'status':'complete','policy':{'warmups':3,'trials':10},'acquisition_id':acq,
                          'allocation':{'actual_sm_count':count,'green_context_id':1},
                          'source_sha256':{'native.cu':'modified' if changed and variant=='source_changed' else 'common'},
                          'cases':[{'id':'CPU-fake','correctness':'passed','metadata':{},'observed_smids':[0],
                                    'stream_proof':{'actual_sm_count':count,'green_context_id':1},
                                    'trials':[{'gpu_event_ms':1.,'repetitions':1} for _ in range(10)]}]}
                    process={'status':'complete','monitor':{'failure':None,'source':'CPU-fake'},
                             'source_freeze_identity':'other' if changed and variant=='freeze_changed' else 'common'}
                    campaign.write(self.sub/'runs'/label/'run.json',data)
                    campaign.write(self.sub/'results/processes'/(label+'.json'),process)
            with patch.object(analysis,'clock_qualification',return_value=clocks), patch.object(analysis,'hierarchical_interval',return_value={'value':1.,'ci95':[1.,1.]}):
                result=analysis.native(self.sub)
            self.assertEqual(result['qualified_curve_count'],1 if variant=='same' else 0)

    def test_closeout_does_not_promote_incomplete_curve(self):
        result=review.closeout(self.root,self.account)
        self.assertFalse(result['new_GPU_measurements'])
        self.assertEqual(result['allocation']['actual_allocated_gpu_hours'],2.)
        curve=result['devices'][0]['curves'][0]
        self.assertFalse(curve['complete_acquisition_set']);self.assertFalse(curve['complete_qualified_curve'])
        self.assertTrue((self.root/'analysis/allocation_closeout.md').is_file())
        self.assertEqual(campaign.load(self.sub/'results/test.json'),self.row)


if __name__=='__main__':unittest.main()
