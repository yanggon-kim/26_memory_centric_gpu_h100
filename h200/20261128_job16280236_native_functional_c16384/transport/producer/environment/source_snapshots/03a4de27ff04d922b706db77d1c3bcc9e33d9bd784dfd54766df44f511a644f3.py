import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from h200_priority_campaign import sha
from h200_producer_campaign import main


class ProducerChain(unittest.TestCase):
    def test_failed_functional_driver_never_starts_timing(self):
        with patch.dict('os.environ',{'SLURM_JOB_ID':'123'}), \
             patch('h200_producer_campaign.subprocess.run',side_effect=subprocess.CalledProcessError(1,'functional')) as run:
            with self.assertRaises(subprocess.CalledProcessError):main(SimpleNamespace(context=16384,preceding_job='122'))
            self.assertEqual(run.call_count,1)

    def test_completed_functional_certificate_precedes_distinct_pilot(self):
        with tempfile.TemporaryDirectory() as folder:
            package=Path(folder);root=package/'h200/20261128_job123_native_functional_c16384'
            (root/'results').mkdir(parents=True);cert=root/'results/cert.json';cert.write_text('{}')
            outcome=root/'results/worker_outcome.json'
            outcome.write_text(json.dumps({'status':'functional_prerequisites_passed','model_review':str(cert),'model_review_sha256':sha(cert)}))
            with patch.dict('os.environ',{'SLURM_JOB_ID':'123'}),patch('h200_producer_campaign.PACKAGE',package), \
                 patch('h200_producer_campaign.subprocess.run') as run:
                main(SimpleNamespace(context=16384,preceding_job='122'))
                self.assertEqual(run.call_count,2)
                second=list(map(str,run.call_args_list[1].args[0]))
                self.assertIn(str(root),second)
                self.assertIn(str(root)+'_producer_timing_pilot',second)
                outcome.write_text(json.dumps({'status':'running'}));run.reset_mock()
                with self.assertRaises(AssertionError):main(SimpleNamespace(context=16384,preceding_job='122'))
                self.assertEqual(run.call_count,1)


if __name__=='__main__':unittest.main()
