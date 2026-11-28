#!/usr/bin/env python3
"""Prospective connector host-span/NVTX observer; never clean timing.

No CUDA imports, tensor reads, added copies, synchronization or backend edits.
A caller must pass a pinned-source connector and actual NVTX push/pop callbacks.
These scopes describe host API intervals; NSYS is still required for GPU
attribution. In particular, save_kv_layer includes extraction and send enqueue,
not transformer-layer compute. Concurrent/nested spans must not be added.
"""
import threading
import time


def interval_union_ns(intervals):
    """Covered host time without adding overlapping or nested intervals."""
    pairs=sorted(intervals)
    total=0;end=None
    for start,stop in pairs:
        assert isinstance(start,int) and isinstance(stop,int) and stop>=start
        if end is None or start>=end:
            total+=stop-start
        elif stop>end:
            total+=stop-end
        end=stop if end is None else max(end,stop)
    return total


class ConnectorSpans:
    classification='instrumented host API spans; not clean timing or GPU kernel attribution'

    def __init__(self,connector,push,pop):
        self.events=[];self.originals=[];self.local=threading.local()
        self.lock=threading.Lock();self.push=push;self.pop=pop
        native=connector.p2p_nccl_engine
        methods=[(connector,'save_kv_layer','layer'),(connector,'wait_for_save',None),
            (connector,'start_load_kv',None),(connector,'wait_for_layer_load','layer'),
            (native,'send_tensor','tensor_id'),(native,'recv_tensor','tensor_id'),
            (native,'send_sync','queue_item'),(native,'wait_for_sent',None),
            (native,'send',None),(native,'recv',None),
            (native.pool,'store_tensor',None),(native.pool,'load_tensor',None),
            (native.pool,'free',None)]
        assert all(callable(getattr(obj,name,None)) for obj,name,_ in methods)
        for obj,name,kind in methods:
            self._wrap(obj,name,kind)

    def _wrap(self,obj,name,kind):
        original=getattr(obj,name)
        owner=type(obj).__name__
        def observed(*args,**kwargs):
            inherited=getattr(self.local,'scope',None)
            rid=inherited['request_id'] if inherited else None
            layer=args[0] if kind=='layer' and args else None
            identifier=None
            if kind=='tensor_id':identifier=args[0] if args else kwargs.get('tensor_id')
            elif kind=='queue_item':
                item=args[0] if args else kwargs.get('item')
                identifier=getattr(item,'tensor_id',None)
            if identifier is not None:
                assert isinstance(identifier,str)
                rid=identifier.split('#',1)[0]
            with self.lock:
                index=len(self.events)
                event={'index':index,'parent':inherited['index'] if inherited else None,
                    'owner':owner,'method':name,'thread_id':threading.get_ident(),
                    'request_id':rid,'tensor_id':identifier,'layer_name':layer,
                    'host_start_unix_ns':time.time_ns(),'host_start_monotonic_ns':time.perf_counter_ns(),
                    'host_end_unix_ns':None,'host_end_monotonic_ns':None,'status':'running','error_type':None,
                    'scope':self.classification}
                self.events.append(event)
            self.local.scope=event
            pushed=False
            try:
                self.push('h200_connector/'+owner+'.'+name+'/'+str(index));pushed=True
                result=original(*args,**kwargs)
                event['status']='returned'
                return result
            except BaseException as error:
                event['status']='raised';event['error_type']=type(error).__name__
                raise
            finally:
                event['host_end_monotonic_ns']=time.perf_counter_ns()
                event['host_end_unix_ns']=time.time_ns()
                self.local.scope=inherited
                if pushed:self.pop()
        setattr(obj,name,observed)
        self.originals.append((obj,name,original,observed))

    def close(self):
        # Close only after owned native sends/receives have drained. Never join
        # native listener threads here; the backend's close can wait forever.
        assert all(getattr(obj,name) is hook for obj,name,_,hook in self.originals)
        assert all(event['status']!='running' for event in self.events),'native spans still active'
        for obj,name,original,_ in reversed(self.originals):setattr(obj,name,original)
        self.originals.clear()

    def summary(self):
        assert all(event['status']!='running' for event in self.events)
        spans=[(e['host_start_monotonic_ns'],e['host_end_monotonic_ns']) for e in self.events]
        return {'classification':self.classification,'host_span_count':len(spans),
            'union_observed_host_ns':interval_union_ns(spans),
            'sum_nested_and_concurrent_host_ns':sum(b-a for a,b in spans),
            'unassigned_request_spans':sum(e['request_id'] is None for e in self.events),
            'GPU_compute_attributed':False,'physical_bytes_measured':False,
            'scope':'Union and sum describe observed host intervals only. Unassigned work and gaps remain unmatched. No fractions of GPU time or transformer-layer compute are inferred.'}
