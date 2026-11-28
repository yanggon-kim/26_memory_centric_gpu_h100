import copy
from pathlib import Path
import tempfile
import unittest
from h200_prefill_replay import reserve
from h200_prefill_replay_review import qualify
from h200_priority_campaign import write


class Replay(unittest.TestCase):
    def fixture(self,base):
        output,publication,index=reserve(base,'whole');output.mkdir()
        prefix={'input_prompt_tokens':4,'first_selected_tokens':1,'actual_prefill_steps':1,'whole':True,'token_budget':4,
            'scheduler_rows':[{'composition':'prefill_only','any_preempted_request':False,'post_prompt_input_queries':0,
                'requests':[{'computed_tokens_before_step':0,'original_prompt_tokens':4,'prompt_input_queries':4}]}]}
        trial={'output_ids':[[7]*64],'decode_iterations':63,'prefill_barrier':{'passed':True},'prefill_and_first_token':prefix}
        case={'batch_size':1,'context_length':4,'generated_tokens':64,'prompt_ids':[[1,2,3,4]],'prompt_sha256':'CPU-fake',
            'warmup_output_ids':[[[7]*64] for _ in range(3)],'output_digests_identical':True,'trials':[copy.deepcopy(trial) for _ in range(3)]}
        native={'status':'complete','hardware':{'uuid':'GPU-CPU-fake'},'cases':[case],
            'config':{'model_id':'CPU-fake','model_revision':'CPU-fake','precision':'BF16','seed':20261013,'generated_tokens':64}}
        write(output/'whole/run.json',native)
        write(output/'whole/prefill_interval_output.json',{'prefix':'whole_measured0','prefill':prefix,'first_selected_ids':[[7]]})
        write(output/'prefill_resident.json',{'status':'complete','profile_condition':'whole','protocol':'resident_prefill_v2'})
        write(publication,{'status':'complete','profile_condition':'whole','output':str(output)})
        return native,output

    def test_complete_exact_replay_and_wrong_selected_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);native,output=self.fixture(base)
            self.assertTrue(qualify(base,native,'whole','GPU-CPU-fake')['passed'])
            self.assertFalse(qualify(base,native,'chunk256','GPU-CPU-fake')['passed'])
            self.assertFalse(qualify(base,native,'whole','GPU-other')['passed'])

    def test_unpublished_and_running_passes_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);native,output=self.fixture(base)
            new,publication,index=reserve(base,'whole');new.mkdir()
            self.assertEqual(index,1)
            self.assertFalse(qualify(base,native,'whole','GPU-CPU-fake')['passed'])
            publication.unlink()
            self.assertIn(new.name,qualify(base,native,'whole','GPU-CPU-fake')['unpublished_directories'])


if __name__=='__main__':unittest.main()
