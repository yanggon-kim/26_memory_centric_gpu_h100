"""CPU safety contracts for fresh connector diagnostic admission."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import h200_connector_entrypoint as entry


class Entry(unittest.TestCase):
    def case(self, folder):
        package = Path(folder)
        core = package/'h200/20261128_job123_native_functional_c16384'
        (core/'results').mkdir(parents=True)
        certificate = core/'model/results/native_model_CPU_review.json'
        certificate.parent.mkdir(parents=True)
        certificate.write_text(json.dumps(dict(status='native_model_functional_review_passed',
            job_id='123', errors=[])))
        outcome = dict(status='functional_prerequisites_passed', model_review=str(certificate),
            model_review_sha256=entry.sha(certificate))
        (core/'results/worker_outcome.json').write_text(json.dumps(outcome))
        return package, core, certificate, SimpleNamespace(context=16384, preceding_job='122')

    def test_fresh_verified_certificate_required_before_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            package, core, _, args = self.case(folder)
            with patch.object(entry, 'PACKAGE', package), patch.dict('os.environ', SLURM_JOB_ID='123'), patch.object(entry.subprocess, 'run') as run:
                self.assertEqual(entry.main(args), core.with_name(core.name+'_connector_capture'))
                self.assertEqual(run.call_count, 2)
                self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))
                self.assertIn('--spare-native-counter', run.call_args_list[0].args[0])
                self.assertIn(str(package/'scripts/h200_connector_capture_campaign.py'), [str(x) for x in run.call_args_list[1].args[0]])

    def test_functional_failure_cannot_launch_profiler(self):
        with tempfile.TemporaryDirectory() as folder:
            package, _, _, args = self.case(folder)
            with patch.object(entry, 'PACKAGE', package), patch.dict('os.environ', SLURM_JOB_ID='123'), patch.object(entry.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'functional')) as run:
                with self.assertRaises(subprocess.CalledProcessError): entry.main(args)
                self.assertEqual(run.call_count, 1)

    def test_tampered_or_wrong_job_certificate_is_rejected(self):
        for failure in ('tampered', 'wrong_job'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                package, core, certificate, args = self.case(folder)
                record=json.loads(certificate.read_text());record['job_id']='999'
                certificate.write_text(json.dumps(record))
                if failure == 'wrong_job':
                    p=core/'results/worker_outcome.json';out=json.loads(p.read_text())
                    out['model_review_sha256']=entry.sha(certificate);p.write_text(json.dumps(out))
                with patch.object(entry, 'PACKAGE', package), patch.dict('os.environ', SLURM_JOB_ID='123'), patch.object(entry.subprocess, 'run') as run:
                    with self.assertRaises(AssertionError): entry.main(args)
                    self.assertEqual(run.call_count, 1)

    def test_existing_capture_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            package, core, _, args = self.case(folder)
            core.with_name(core.name+'_connector_capture').mkdir()
            with patch.object(entry, 'PACKAGE', package), patch.dict('os.environ', SLURM_JOB_ID='123'), patch.object(entry.subprocess, 'run') as run:
                with self.assertRaises(AssertionError): entry.main(args)
                self.assertEqual(run.call_count, 1)


if __name__ == '__main__': unittest.main()
