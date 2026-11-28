"""CPU exact portable-input admission; no model/GPU execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h200_portable_prompt_contract import portable_prompts
from h200_portable_serving_trace import make_trace


class Prompts(unittest.TestCase):
    def case(self,folder):
        trace=make_trace('lclo','poisson',1.,30,101,'development')
        path=Path(folder)/'trace.json';path.write_text(json.dumps(trace))
        request=trace['requests'][0]
        config=dict(precision='BF16',seed=20261013,model_id=trace['model']['id'],
            model_revision=trace['model']['revision'],portable_trace_file=str(path),
            portable_trace_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            portable_request_id=request['id'])
        return trace,request,config,path

    def test_exact_full_ids_and_lengths_are_returned(self):
        with tempfile.TemporaryDirectory() as folder:
            trace,request,config,_=self.case(folder)
            ids,metadata=portable_prompts(config,1,len(request['prompt_ids']),request['max_tokens'])
            self.assertEqual(ids,[request['prompt_ids']])
            self.assertEqual(metadata['full_output_tokens'],request['max_tokens'])
            self.assertEqual(metadata['trace_sha256'],trace['sha256'])
            self.assertFalse(metadata['GPU_execution'])
            self.assertFalse(metadata['correctness_qualified'])

    def test_prompt_output_shortening_or_batch_replication_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _,request,config,_=self.case(folder);context=len(request['prompt_ids']);count=request['max_tokens']
            for batch,c,n in [(2,context,count),(1,context-1,count),(1,context,count-1)]:
                with self.subTest(shape=(batch,c,n)),self.assertRaises(AssertionError):portable_prompts(config,batch,c,n)

    def test_changed_file_pin_and_modified_sealed_content_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            trace,request,config,path=self.case(folder)
            trace['requests'][0]['prompt_ids'][0]+=1;path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(AssertionError,'config pin'):portable_prompts(config,1,len(request['prompt_ids']),request['max_tokens'])
            config['portable_trace_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(AssertionError):portable_prompts(config,1,len(request['prompt_ids']),request['max_tokens'])

    def test_wrong_model_precision_seed_request_or_second_input_route_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _,request,config,_=self.case(folder)
            for key,value in [('model_revision','other'),('precision','FP8'),('seed',42),('portable_request_id','missing'),('workload_file','text.json'),('decode_conditioning',{'protocol':'different_history'})]:
                changed=dict(config);changed[key]=value
                with self.subTest(field=key),self.assertRaises(AssertionError):portable_prompts(changed,1,len(request['prompt_ids']),request['max_tokens'])


if __name__=='__main__':unittest.main()
