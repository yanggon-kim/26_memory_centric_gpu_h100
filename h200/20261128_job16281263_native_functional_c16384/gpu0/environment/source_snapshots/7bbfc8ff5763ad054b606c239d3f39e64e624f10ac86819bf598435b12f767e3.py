from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import Mock,patch
from h200_prefill_campaign import capture


class Budget(unittest.TestCase):
    def test_insufficient_comparison_budget_never_launches_a_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            execute=Mock(side_effect=AssertionError('GPU producer must not launch'))
            worker=types.SimpleNamespace(root=Path(directory),script=Path(__file__).resolve().parent,
                hs=types.SimpleNamespace(verify_allocation=lambda:{'effective_launch_cutoff_unix_s':time.time()+60}),execute=execute)
            with patch('h200_resident_campaign.build_environment'):
                result=capture(worker,1024)
            self.assertEqual(result['status'],'deferred_budget');execute.assert_not_called()


if __name__=='__main__':unittest.main()
