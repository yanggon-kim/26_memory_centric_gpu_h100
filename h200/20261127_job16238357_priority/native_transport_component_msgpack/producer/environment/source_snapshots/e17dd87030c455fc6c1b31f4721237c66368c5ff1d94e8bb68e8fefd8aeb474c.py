from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from h200_native_functional_campaign import run_pair,run_one


class PairedSupervisor(unittest.TestCase):
    def test_no_role_is_launched_without_complete_budget(self):
        with patch('h200_native_functional_campaign.subprocess.Popen') as spawn:
            with self.assertRaises(AssertionError):
                run_pair(Path('/unused'),{},'entry.py','label','output',540,time.time()+500)
            with self.assertRaises(AssertionError):
                run_one(Path('/unused'),'label','entry.py',[],300,time.time()+200)
            spawn.assert_not_called()

    def test_failed_role_stops_waiting_peer_without_waiting_native_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'logs').mkdir()
            for role in ('producer','consumer'):
                (root/role/'results').mkdir(parents=True)
            pair = {'roles':{'producer':{'gpu_uuid':'GPU-a'},'consumer':{'gpu_uuid':'GPU-b'}}}
            processes = []; real_spawn = subprocess.Popen
            def command(subroot,*args):
                return [sys.executable,'-c','raise SystemExit(9)' if subroot.name=='producer' else 'import time; time.sleep(60)'],{}
            def spawn(*args,**kwargs):
                proc = real_spawn(*args,**kwargs); processes.append(proc); return proc
            began = time.time()
            try:
                with patch('h200_native_functional_campaign.supervised_command',side_effect=command), \
                     patch('h200_native_functional_campaign.subprocess.Popen',side_effect=spawn):
                    with self.assertRaises(AssertionError):
                        run_pair(root,pair,'entry.py','label','output',540,time.time()+650)
                self.assertLess(time.time()-began,5)
                self.assertEqual(len(processes),2)
                self.assertTrue(all(p.poll() is not None for p in processes))
            finally:
                for proc in processes:
                    if proc.poll() is None: proc.kill(); proc.wait()


if __name__ == '__main__': unittest.main()
