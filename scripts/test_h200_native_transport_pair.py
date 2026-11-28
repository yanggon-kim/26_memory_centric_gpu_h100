import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from h200_native_transport_pair import validate_sources, choose_ports, stop_owned_role, validate_returned


class PairAdmission(unittest.TestCase):
    def test_idle_between_producers_does_not_mean_original_worker_returned(self):
        done = {'status': 'resident_curve_disposition', 'complete_acquisition_set': True}
        validate_returned([done, {'status': 'prefill_acquisition_disposition', 'complete_acquisition_set': True}])
        for other in ({'status': 'running'}, {'status': 'failed'},
                      {'status': 'prefill_acquisition_disposition', 'complete_acquisition_set': False}):
            with self.assertRaises(AssertionError): validate_returned([done, other])

    def test_terminal_reference_block_needs_separate_evidence_review(self):
        blocked={'status':'blocked_resident_pilot','complete_acquisition_set':False,'allow_context_expansion':True}
        done={'status':'prefill_acquisition_disposition','complete_acquisition_set':True}
        with self.assertRaises(AssertionError):validate_returned([blocked,done])
        with self.assertRaises(AssertionError):
            validate_returned([blocked,done],[{'status':'terminal_exact_reference_failure_verified','curve_qualified':True},None])
        validate_returned([blocked,done],[{'status':'terminal_exact_reference_failure_verified','curve_qualified':False},None])

    def test_fail_closed_for_wrong_allocation_busy_or_duplicate_device(self):
        sessions = [{'gpu_uuid': uuid, 'python': '/p', 'cuda_home': '/c',
                     'slurm_job_id': '123', 'expected_hostname': 'node.example'}
                    for uuid in ('GPU-a', 'GPU-b')]
        currents = [{'status': 'idle'}, {'status': 'idle'}]
        state = 'JobId=123 JobState=RUNNING UserId=user(55) NodeList=node AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h200=2'
        validate_sources(sessions, currents, '123', 'node.example', 55, state)
        for old, new in (('RUNNING', 'PENDING'), ('user(55)', 'user(56)'),
                         ('JobId=123', 'JobId=124'), ('h200=2', 'h200=4')):
            with self.assertRaises(AssertionError):
                validate_sources(sessions, currents, '123', 'node.example', 55, state.replace(old, new))
        bad = copy.deepcopy(sessions); bad[1]['gpu_uuid'] = 'GPU-a'
        with self.assertRaises(AssertionError):
            validate_sources(bad, currents, '123', 'node.example', 55, state)
        with self.assertRaises(AssertionError):
            validate_sources(sessions, [{'status': 'running'}, currents[1]], '123', 'node.example', 55, state)

    def test_distinct_loopback_ports(self):
        ports = choose_ports()
        self.assertEqual(len(set(ports)), 2)
        self.assertTrue(all(0 < p < 65536 for p in ports))

    def test_cleanup_owned_child_leaves_unrelated_process_running(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'results').mkdir()
            code = ('import subprocess,sys,json,pathlib; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True); '
                    'pathlib.Path(sys.argv[1]).write_text(json.dumps({"pid":p.pid})); p.wait()')
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
            supervisor = subprocess.Popen([sys.executable, '-c', code,
                str(root/'results/current_process.json')], start_new_session=True)
            try:
                deadline = time.time()+5
                while not (root/'results/current_process.json').exists():
                    self.assertLess(time.time(), deadline); time.sleep(.01)
                child = json.loads((root/'results/current_process.json').read_text())['pid']
                stop_owned_role(supervisor, root)
                self.assertIsNotNone(supervisor.poll())
                self.assertFalse(Path(f'/proc/{child}').exists())
                self.assertIsNone(unrelated.poll())
            finally:
                if supervisor.poll() is None: supervisor.kill(); supervisor.wait()
                unrelated.terminate(); unrelated.wait()


if __name__ == '__main__': unittest.main()
