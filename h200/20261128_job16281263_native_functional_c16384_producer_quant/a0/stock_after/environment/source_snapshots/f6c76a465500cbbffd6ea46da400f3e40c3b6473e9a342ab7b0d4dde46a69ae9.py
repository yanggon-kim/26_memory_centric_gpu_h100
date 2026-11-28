"""Real CPU-thread/file ordering and invalid-peer tests; no GPU calls."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest
from h200_connector_capture_barrier import PeerBarrier


def dump(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data));temporary.replace(path)


def wait(path,deadline):
    while not path.exists():
        if time.time()>=deadline:raise TimeoutError('peer never ready')
        time.sleep(.001)
    return json.loads(path.read_text())


class Barrier(unittest.TestCase):
    def pair(self,folder):
        return {'job_id':'123','shared_directory':str(folder),'deadline_unix_s':time.time()+5,
            'roles':{'producer':{'gpu_uuid':'GPU-a'},'consumer':{'gpu_uuid':'GPU-b'}}}

    def test_both_actor_starts_and_stops_are_required_before_advancing(self):
        with tempfile.TemporaryDirectory() as folder,ThreadPoolExecutor(2) as pool:
            root=Path(folder);pair=self.pair(root)
            producer=PeerBarrier(pair,'producer',100,dump,wait);consumer=PeerBarrier(pair,'consumer',200,dump,wait)
            for stage in ('started','stopped'):
                first=pool.submit(producer.meet,stage,'request')
                limit=time.time()+1
                while not (root/('profiler_'+stage+'_producer.json')).exists() and time.time()<limit:time.sleep(.001)
                self.assertFalse(first.done(),'producer advanced before peer profiler stage was ready')
                second=pool.submit(consumer.meet,stage,'request')
                a,b=first.result(timeout=2),second.result(timeout=2)
                self.assertEqual(a['peer_ready_unix_ns'],b['self_ready_unix_ns'])
                self.assertEqual(b['peer_ready_unix_ns'],a['self_ready_unix_ns'])
                self.assertEqual(a['status'],b['status'])

    def test_wrong_peer_gpu_or_request_is_preserved_as_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            for fault in ('gpu','request'):
                pair=self.pair(folder)
                peer={'job_id':'123','stage':'started','role':'consumer','actor_pid':200,'ready_unix_ns':1,
                    'gpu_uuid':'GPU-wrong' if fault=='gpu' else 'GPU-b','request_id':'wrong' if fault=='request' else 'request'}
                barrier=PeerBarrier(pair,'producer',100,dump,lambda *args:peer)
                with self.subTest(fault=fault),self.assertRaises(AssertionError):barrier.meet('started','request')
                self.assertEqual(barrier.history[0]['status'],'peer_barrier_failed')

    def test_timeout_is_bounded_and_does_not_claim_peer_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            pair=self.pair(folder);pair['deadline_unix_s']=time.time()+.01
            barrier=PeerBarrier(pair,'producer',100,dump,wait)
            with self.assertRaises(TimeoutError):barrier.meet('started','request')
            self.assertEqual(barrier.history[0]['status'],'peer_barrier_failed')
            self.assertIsNone(barrier.history[0]['peer_received_unix_ns'])


if __name__=='__main__':unittest.main()
