"""CPU ordering contracts: optional profiling cannot replace primary work."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from h200_priority_campaign import sha
from h200_producer_quant_entrypoint import main


class Entry(unittest.TestCase):
    def setup_case(self,folder):
        package=Path(folder);root=package/'h200/20261128_job123_native_functional_c16384'
        (root/'results').mkdir(parents=True);cert=root/'results/cert.json';cert.write_text('{}')
        (root/'results/worker_outcome.json').write_text(json.dumps({'status':'functional_prerequisites_passed',
            'model_review':str(cert),'model_review_sha256':sha(cert)}))
        helper=ModuleType('h200_planned_connector_followup');helper.run_if_planned=Mock(return_value={'status':'no_plan'})
        args=SimpleNamespace(context=16384,preceding_job='122',pilot_root=package/'h200/preserved-pilot')
        return package,root,helper,args

    def test_completed_primary_subprocess_precedes_optional_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            package,root,helper,args=self.setup_case(folder)
            with patch('h200_producer_quant_entrypoint.PACKAGE',package),patch.dict('os.environ',{'SLURM_JOB_ID':'123'}),patch.dict('sys.modules',{'h200_planned_connector_followup':helper}),patch('h200_producer_quant_entrypoint.subprocess.run') as run:
                self.assertEqual(main(args),{'status':'no_plan'})
                self.assertEqual(run.call_count,2)
                helper.run_if_planned.assert_called_once_with(root,root.with_name(root.name+'_producer_quant'))
                self.assertTrue(all(c.kwargs['check'] for c in run.call_args_list))

    def test_failed_fixed_cohort_never_invokes_optional_controller(self):
        with tempfile.TemporaryDirectory() as folder:
            package,_,helper,args=self.setup_case(folder)
            with patch('h200_producer_quant_entrypoint.PACKAGE',package),patch.dict('os.environ',{'SLURM_JOB_ID':'123'}),patch.dict('sys.modules',{'h200_planned_connector_followup':helper}),patch('h200_producer_quant_entrypoint.subprocess.run',side_effect=[None,subprocess.CalledProcessError(9,'quant')]):
                with self.assertRaises(subprocess.CalledProcessError):main(args)
                helper.run_if_planned.assert_not_called()

    def test_failed_functional_gate_never_invokes_cohort_or_optional_controller(self):
        with tempfile.TemporaryDirectory() as folder:
            package,_,helper,args=self.setup_case(folder)
            with patch('h200_producer_quant_entrypoint.PACKAGE',package),patch.dict('os.environ',{'SLURM_JOB_ID':'123'}),patch.dict('sys.modules',{'h200_planned_connector_followup':helper}),patch('h200_producer_quant_entrypoint.subprocess.run',side_effect=subprocess.CalledProcessError(8,'functional')) as run:
                with self.assertRaises(subprocess.CalledProcessError):main(args)
                self.assertEqual(run.call_count,1);helper.run_if_planned.assert_not_called()


if __name__=='__main__':unittest.main()
