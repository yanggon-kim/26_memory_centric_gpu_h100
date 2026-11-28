import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import controlled_vllm_experiments as baseline
from h200_serving_load import main


class CaptureLabels(unittest.TestCase):
    def test_instrumented_label_is_present_even_if_producer_stops_early(self):
        original_trial,original_dump=baseline.run_trial,baseline.dump
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            def stopped(args):
                baseline.dump(args.output/'run.json',{'status':'running','baseline_comparison_eligible':True})
                raise RuntimeError('CPU-only early-stop fixture')
            with patch.object(baseline,'main',stopped):
                with self.assertRaisesRegex(RuntimeError,'CPU-only early-stop'):
                    main(types.SimpleNamespace(output=output))
            record=json.loads((output/'run.json').read_text())
            self.assertFalse(record['baseline_comparison_eligible'])
            self.assertEqual(record['classification'],'measured-real-GPU-instrumented')
            self.assertIs(baseline.run_trial,original_trial)
            self.assertIs(baseline.dump,original_dump)


if __name__=='__main__':unittest.main()
