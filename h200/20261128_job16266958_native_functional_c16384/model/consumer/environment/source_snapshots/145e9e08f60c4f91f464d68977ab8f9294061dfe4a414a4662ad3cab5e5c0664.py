import threading
import unittest
from types import SimpleNamespace
from h200_connector_spans import ConnectorSpans,interval_union_ns


class Spans(unittest.TestCase):
    def fixture(self):
        pool=SimpleNamespace(store_tensor=lambda x:x,load_tensor=lambda x:x,free=lambda x:x)
        native=SimpleNamespace(pool=pool,send_tensor=lambda *a,**k:False,recv_tensor=lambda *a,**k:None,
            send_sync=lambda item: native.send(None,item.tensor,None),wait_for_sent=lambda:None,
            send=lambda *a,**k:'native-return',recv=lambda *a,**k:None)
        connector=SimpleNamespace(p2p_nccl_engine=native,save_kv_layer=lambda *a,**k:None,
            wait_for_save=lambda:None,start_load_kv=lambda *a,**k:None,wait_for_layer_load=lambda *a:None)
        return connector,native

    def test_union_preserves_overlap_and_rejects_bad_intervals(self):
        self.assertEqual(interval_union_ns([(10,30),(15,20),(25,40),(50,55)]),35)
        self.assertEqual(interval_union_ns([]),0)
        with self.assertRaises(AssertionError):interval_union_ns([(4,3)])

    def test_async_thread_request_ownership_and_nested_scopes(self):
        connector,native=self.fixture();marks=[]
        observer=ConnectorSpans(connector,lambda name:marks.append(name),lambda:marks.append('pop'))
        item=SimpleNamespace(tensor_id='request7#layer1',tensor=object())
        results=[]
        thread=threading.Thread(target=lambda:results.append(native.send_sync(item)))
        thread.start();thread.join()
        self.assertEqual(results,['native-return'])
        self.assertEqual([e['request_id'] for e in observer.events],['request7','request7'])
        self.assertEqual(observer.events[1]['parent'],0)
        self.assertEqual(observer.summary()['unassigned_request_spans'],0)
        self.assertFalse(observer.summary()['GPU_compute_attributed'])
        self.assertGreaterEqual(observer.summary()['sum_nested_and_concurrent_host_ns'],observer.summary()['union_observed_host_ns'])
        observer.close();self.assertFalse(observer.originals)

    def test_false_returns_exceptions_and_unassigned_work_are_retained(self):
        connector,native=self.fixture()
        def fail(*args):raise ValueError('original')
        native.recv=fail
        original=native.recv
        observer=ConnectorSpans(connector,lambda name:None,lambda:None)
        self.assertIs(native.send_tensor('r#layer',object(),'remote'),False)
        with self.assertRaisesRegex(ValueError,'original'):native.recv(None,None,None)
        self.assertEqual(observer.events[-1]['error_type'],'ValueError')
        self.assertIsNone(observer.events[-1]['request_id'])
        observer.close();self.assertIs(native.recv,original)


if __name__=='__main__':unittest.main()
