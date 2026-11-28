#!/usr/bin/env python3
"""CPU checks of the reused-reference identity and exact-token gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from h200_traffic_resume import reference
from h200_priority_campaign import sha


class Contract(unittest.TestCase):
    def test_reference_reuse_requires_same_device_source_prompt_and_outputs(self):
        ids=[[9]*64];prompt=[[11]*4096]
        trial={'output_ids':ids,'prefill_barrier':{'passed':True},'decode_iterations':63}
        source={'status':'complete','hardware':{'uuid':'GPU-owned'},
                'config':{'model_revision':'immutable','config_sha256':'model_config'},
                'engine_settings':{'dtype':'bfloat16'},'versions':{'torch':'pinned','cuda_runtime':'13.0'},
                'cases':[{'batch_size':1,'context_length':4096,'generated_tokens':64,
                          'prompt_ids':prompt,'prompt_sha256':'prompt','output_digests_identical':True,
                          'warmup_output_ids':[ids]*3,'trials':[trial]*10}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.json';path.write_text(json.dumps(source))
            hf={'gpu_uuid':'GPU-owned','source_sha256':sha(path),'status':'passed',
                'reference_outputs_identical':True,'warmups':[{'output_ids':ids}]*3,
                'trials':[{'output_ids':ids}]*3,'prompt_sha256':'prompt','prompt_ids':prompt,
                'revision':'immutable','config_sha256':'model_config','precision':'BF16',
                'versions':{'torch':'pinned','cuda':'13.0'}}
            gold=Path(directory)/'hf.json';gold.write_text(json.dumps(hf))
            self.assertEqual(reference(path,gold,'GPU-owned')[1]['context_length'],4096)
            for key,value in [('gpu_uuid','GPU-foreign'),('source_sha256','tampered'),
                              ('prompt_ids',[[12]*4096]),('warmups',[{'output_ids':[[8]*64]}]*3)]:
                bad=copy.deepcopy(hf);bad[key]=value;gold.write_text(json.dumps(bad))
                with self.subTest(key=key),self.assertRaises(AssertionError):
                    reference(path,gold,'GPU-owned')


if __name__=='__main__':
    unittest.main()
