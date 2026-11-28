"""Bounded host-only peer barrier for future instrumented profiler ranges.

Both profilers must start before either actor submits the selected request.
Both must stop before either actor advances. This adds no GPU sync or copies;
its file IPC/waiting is explicitly part of the instrumented functional scope.
"""
import time
from pathlib import Path


class PeerBarrier:
    def __init__(self,pair,role,actor_pid,dump,wait):
        assert role in ('producer','consumer') and type(actor_pid) is int and actor_pid>0
        self.pair=pair;self.role=role;self.pid=actor_pid;self.dump=dump;self.wait=wait
        self.peer='consumer' if role=='producer' else 'producer'
        self.history=[]

    def meet(self,stage,request_id):
        assert stage==('started' if not self.history else 'stopped') and len(self.history)<2
        if self.history:assert self.history[0]['status']=='peer_ready_verified'
        record={'stage':stage,'request_id':request_id,'job_id':self.pair['job_id'],
            'role':self.role,'gpu_uuid':self.pair['roles'][self.role]['gpu_uuid'],'actor_pid':self.pid,
            'self_ready_unix_ns':time.time_ns(),'peer_received_unix_ns':None,
            'status':'waiting_for_peer','error':None}
        self.history.append(record)
        shared=Path(self.pair['shared_directory'])
        payload={k:record[k] for k in ('stage','request_id','job_id','role','gpu_uuid','actor_pid')}
        payload['ready_unix_ns']=record['self_ready_unix_ns']
        try:
            self.dump(shared/('profiler_'+stage+'_'+self.role+'.json'),payload)
            peer=self.wait(shared/('profiler_'+stage+'_'+self.peer+'.json'),
                min(self.pair['deadline_unix_s'],time.time()+60))
            assert peer['job_id']==self.pair['job_id'] and peer['request_id']==request_id and peer['stage']==stage
            assert peer['role']==self.peer and peer['gpu_uuid']==self.pair['roles'][self.peer]['gpu_uuid']
            assert type(peer['actor_pid']) is int and peer['actor_pid']>0 and peer['actor_pid']!=self.pid
            assert type(peer['ready_unix_ns']) is int and peer['ready_unix_ns']>0
            record.update(peer_role=self.peer,peer_actor_pid=peer['actor_pid'],peer_gpu_uuid=peer['gpu_uuid'],
                peer_ready_unix_ns=peer['ready_unix_ns'],peer_received_unix_ns=time.time_ns(),status='peer_ready_verified')
        except BaseException as error:
            record.update(status='peer_barrier_failed',error=type(error).__name__+': '+str(error))
            raise
        return record
