"""CPU toy-engine failure contracts; no portable-input or GPU qualification."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from h200_portable_serving_capture import capture


class Clock:
    def __init__(self):self.now=0.
    def monotonic(self):return self.now
    def sleep(self,seconds):self.now+=seconds


class Engine:
    def __init__(self,clock,fail=False,unknown=False):
        self.clock=clock;self.active={};self.lengths=[];self.fail=fail;self.unknown=unknown
    def has_unfinished_requests(self):return bool(self.active)
    def add_request(self,rid,prompt,sampling):
        self.active[rid]={'tokens':[],'count':sampling};self.lengths.append(sampling)
        return rid
    def step(self):
        self.clock.now+=.01
        if self.fail:raise RuntimeError('toy engine failure')
        outputs=[]
        for rid,row in list(self.active.items()):
            row['tokens'].append(7);finished=len(row['tokens'])==row['count']
            outputs.append(SimpleNamespace(request_id='unknown' if self.unknown else rid,
                outputs=[SimpleNamespace(token_ids=list(row['tokens']),finish_reason='length' if finished else None)],finished=finished))
            if finished:del self.active[rid]
        return outputs


class Capture(unittest.TestCase):
    def setUp(self):
        self.trace={'sha256':'a'*64,'AMD_trace_sha256':'b'*64,'workload':'lclo',
            'arrival':'bursty','configured_rate_requests_s':1.,'seed':101,'split':'development',
            'model':{'vocab_size':152064},'warmup_requests':0,
            'requests':[{'id':'q'+str(i),'arrival_s':arrival,'prompt_ids':[1,2],'max_tokens':count}
                for i,(arrival,count) in enumerate(((0.,2),(0.,3),(.025,4)))]}
        self.expected={q['id']:[7]*q['max_tokens'] for q in self.trace['requests']}
        # Toy traces isolate client state transitions. Production validation is
        # unchanged and requires the real sealed AMD generator inputs.
        self.client_validator=patch('h200_portable_serving_capture.validate',return_value=True)
        self.score_validator=patch('h200_portable_serving_score.validate',return_value=True)
        self.client_validator.start();self.score_validator.start()
        self.addCleanup(self.client_validator.stop);self.addCleanup(self.score_validator.stop)

    def test_tied_arrivals_and_different_lengths_reach_exact_references(self):
        with tempfile.TemporaryDirectory() as folder:
            clock=Clock();engine=Engine(clock);path=Path(folder)/'capture.json'
            result=capture(engine,lambda q:q['max_tokens'],self.trace,self.expected,path,1.,clock=clock)
            self.assertEqual(engine.lengths,[2,3,4])
            self.assertEqual([len(r['output_ids']) for r in result['requests']],[2,3,4])
            self.assertEqual(result['requests'][0]['submitted_s'],result['requests'][1]['submitted_s'])
            self.assertEqual(result['accounting']['measured']['exact_reference_completed'],3)
            self.assertFalse(result['quantitative_performance_qualified'])
            self.assertIsNone(result['decoder_step_latency_ms'])
            self.assertTrue(path.with_suffix('.trace.json').exists())

    def test_failure_keeps_unreached_arrivals_and_partial_observations(self):
        self.trace['requests'][1]['arrival_s']=1.
        self.trace['requests'][2]['arrival_s']=2.
        self.trace['warmup_requests']=1
        with tempfile.TemporaryDirectory() as folder:
            clock=Clock();path=Path(folder)/'capture.json'
            with self.assertRaisesRegex(RuntimeError,'toy engine failure'):
                capture(Engine(clock,fail=True),lambda q:q['max_tokens'],self.trace,self.expected,path,3.,clock=clock)
            saved=json.loads(path.read_text())
            self.assertEqual(saved['accounting']['all_offered'],3)
            self.assertEqual(saved['accounting']['all_attempted'],1)
            self.assertEqual(saved['accounting']['measured']['incomplete'],2)
            self.assertFalse(saved['accounting']['measured_window_started'])
            self.assertIsNone(saved['accounting']['measured_window_including_drain_s'])
            self.assertEqual(saved['error_type'],'RuntimeError')

    def test_timeout_preserves_emitted_tokens_without_claiming_engine_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            clock=Clock();path=Path(folder)/'capture.json'
            with self.assertRaises(TimeoutError):
                capture(Engine(clock),lambda q:q['max_tokens'],self.trace,self.expected,path,.005,clock=clock)
            saved=json.loads(path.read_text())
            self.assertEqual([len(r['output_ids']) for r in saved['requests']],[1,1,0])
            self.assertFalse(saved['engine_abort'])
            self.assertEqual(saved['accounting']['all_offered'],3)
            self.assertEqual(saved['accounting']['measured']['incomplete'],3)

    def test_unoffered_output_is_retained_and_cannot_be_counted_successful(self):
        with tempfile.TemporaryDirectory() as folder:
            clock=Clock();path=Path(folder)/'capture.json'
            with self.assertRaisesRegex(AssertionError,'unoffered output'):
                capture(Engine(clock,unknown=True),lambda q:q['max_tokens'],self.trace,self.expected,path,1.,clock=clock)
            saved=json.loads(path.read_text())
            self.assertEqual(saved['unexpected_outputs'][0]['request_id'],'unknown')
            self.assertEqual(saved['accounting']['measured']['exact_reference_completed'],0)


if __name__=='__main__':unittest.main()
