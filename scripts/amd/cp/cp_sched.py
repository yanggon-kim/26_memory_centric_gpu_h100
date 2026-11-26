"""Chunked-prefill campaign (docs/amd/chunked_prefill_campaign_20261118.md): scheduler policy control and per-step log
for vLLM 0.23 V1 (scheduler_cls). Loaded inside the vLLM container, in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) or in
an `vllm serve` engine-core process (PYTHONPATH includes scripts/amd/cp).

Policy (environment CP_POLICY, read at construction):
  mixed     stock vLLM V1: running requests first (decodes, unfinished prompt chunks), then waiting requests, in one
            step under the token budget (max_num_batched_tokens). Sarathi-style decode-first mixing.
  separate  every step is prefill-only or decode-only. A prefill step runs when any request still needs prompt tokens
            (unfinished chunk, waiting request with a free sequence slot, or a preempted request to recompute): the
            decoding requests are hidden from that step. Otherwise a decode-only step runs with the waiting queue
            hidden. Prefill-prioritizing, as vLLM V0's default. With enable_chunked_prefill=False the engine admits only
            whole prompts that fit the budget (no chunks): 'whole prompts, served separately'.
If a prefill-only step schedules nothing (KV memory full), the next step is decode-only, so the engine cannot stall.

Per-step log (CP_STEPLOG = path of a JSON-lines file; one line per executed step, written at update_from_output):
  i, t0 (perf_counter at schedule start), sched_s, exec_s (schedule return -> update_from_output: model execution
  incl. sampling; synchronous in this engine mode), kind (decode|prefill|mixed|empty), n_decode, decode_ctx_sum,
  decode_ctx_max, prefill: [[req_id, start, tokens, prompt_len, cached_at_admit, recompute]], n_running, n_waiting,
  preempted, total_tokens. CP_STEPLOG_IDS=1 adds the decode request ids.
"""
import json
import os
import time

from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler


def _p2p_without_cumem():
    """Platform adaptation (dev run 2026-11-18): vLLM 0.23's P2pNcclConnector forces NCCL_CUMEM_ENABLE=1 while it creates
    its RCCL communicator, and RCCL's cuMem peer path needs Linux >= 6.8 (cluster nodes run 5.14): the KV channel failed
    with 'unhandled cuda error'. With CP_P2P_CUMEM=0 the same context manager sets NCCL_CUMEM_ENABLE=0 (RCCL's IPC peer
    path, which carried the cp_link.py RCCL transfers). Nothing else in the connector changes."""
    from contextlib import contextmanager
    import vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine as eng
    orig = eng.set_p2p_nccl_context

    @contextmanager
    def ctx(num_channels):
        with orig(num_channels):
            os.environ['NCCL_CUMEM_ENABLE'] = '0'
            yield
    eng.set_p2p_nccl_context = ctx


def _p2p_none_blocks():
    """Fix for vLLM 0.23's P2pNcclConnector producer (s3 job 530467, 2026-11-18): when a chunked prompt's continuation
    chunk fits in blocks already allocated, the scheduler passes new_block_ids = None for that cached request and
    build_connector_meta hits `assert new_block_ids is not None`, killing the prefill worker (balanced at >= 2 rps and
    pheavy at >= 1.5 rps). None means 'no new blocks', so for this call only each None entry is replaced by one empty
    block list per KV-cache group (Qwen2.5 has one) and restored afterwards; the scheduler output is otherwise
    untouched."""
    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import P2pNcclConnector
    orig = P2pNcclConnector.build_connector_meta

    def build_connector_meta(self, scheduler_output):
        c = scheduler_output.scheduled_cached_reqs
        saved = c.new_block_ids
        if any(b is None for b in saved):
            c.new_block_ids = [([],) if b is None else b for b in saved]
        try:
            return orig(self, scheduler_output)
        finally:
            c.new_block_ids = saved
    P2pNcclConnector.build_connector_meta = build_connector_meta


if os.environ.get('CP_P2P_CUMEM') == '0':
    _p2p_without_cumem()
if os.environ.get('CP_P2P_NONE_BLOCKS') == '1':
    _p2p_none_blocks()


class PolicyScheduler(Scheduler):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cp_policy = os.environ.get('CP_POLICY', 'mixed')
        assert self.cp_policy in ('mixed', 'separate'), self.cp_policy
        path = os.environ.get('CP_STEPLOG')
        self.cp_file = open(path, 'a', buffering=1) if path else None
        self.cp_ids = os.environ.get('CP_STEPLOG_IDS') == '1'
        self.cp_step = 0
        self.cp_stalled = False
        self.cp_pending = {}
        self.cp_listeners = []  # in-process harnesses may append callables(record)

    @staticmethod
    def _needs_prompt(r):
        return r.num_computed_tokens < r.num_tokens - 1

    def schedule(self):
        t0 = time.perf_counter()
        pre = {rid: (r.num_computed_tokens, r.num_tokens, r.num_prompt_tokens, r.num_output_tokens)
               for rid, r in self.requests.items()}
        hidden, hidden_q, step_mode = None, None, 'mixed'
        if self.cp_policy == 'separate':
            prefilling = [r for r in self.running if self._needs_prompt(r)]
            decoding = [r for r in self.running if not self._needs_prompt(r)]
            waiting = len(self.waiting) + len(self.skipped_waiting)
            want = bool(prefilling) or (waiting > 0 and len(self.running) < self.max_num_running_reqs)
            if want and not self.cp_stalled:
                step_mode = 'prefill'
                if decoding:
                    hidden = decoding
                    self.running = prefilling
                    self.max_num_running_reqs -= len(decoding)
            else:
                step_mode = 'decode'
                hidden_q = (self.waiting, self.skipped_waiting)
                self.waiting, self.skipped_waiting = create_request_queue(self.policy), create_request_queue(self.policy)
        try:
            out = super().schedule()
        finally:
            if hidden is not None:
                self.running = hidden + self.running
                self.max_num_running_reqs += len(hidden)
            if hidden_q is not None:
                for q, saved in zip((self.waiting, self.skipped_waiting), hidden_q):
                    # a decode step can preempt a running request under KV pressure; vLLM prepends it to waiting
                    # (Scheduler._preempt_request). Carry it to the front of the restored queue, as vLLM would.
                    # (lc amendment 4: before this fix the step aborted with 'hidden waiting queue received requests')
                    if q:
                        saved.prepend_requests(q)
                self.waiting, self.skipped_waiting = hidden_q
        self.cp_stalled = step_mode == 'prefill' and out.total_num_scheduled_tokens == 0
        t1 = time.perf_counter()
        new_cached = {nr.req_id: nr.num_computed_tokens for nr in out.scheduled_new_reqs}
        n_dec, ctx_sum, ctx_max, dec_ids, chunks = 0, 0, 0, [], []
        for rid, n in out.num_scheduled_tokens.items():
            c, nt, npr, nout = pre.get(rid, (0, 0, 0, 0))
            if rid in new_cached:
                c = max(c, new_cached[rid])
            if c >= nt - 1 and n == 1:
                n_dec += 1
                ctx_sum += c
                ctx_max = max(ctx_max, c)
                if self.cp_ids:
                    dec_ids.append(rid)
            else:
                chunks.append([rid, c, n, npr, new_cached.get(rid), nout > 0])
        kind = ('empty' if not out.num_scheduled_tokens else 'mixed' if n_dec and chunks else
                'decode' if n_dec else 'prefill')
        rec = {'i': self.cp_step, 't0': t0, 'sched_s': t1 - t0, 'kind': kind, 'mode': step_mode, 'n_decode': n_dec,
               'decode_ctx_sum': ctx_sum, 'decode_ctx_max': ctx_max, 'prefill': chunks,
               'n_running': len(self.running), 'n_waiting': len(self.waiting) + len(self.skipped_waiting),
               'preempted': len(out.preempted_req_ids or ()), 'total_tokens': out.total_num_scheduled_tokens}
        if self.cp_ids:
            rec['decode_ids'] = dec_ids
        self.cp_pending[id(out)] = (rec, t1)
        self.cp_step += 1
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        t = time.perf_counter()
        got = self.cp_pending.pop(id(scheduler_output), None)
        res = super().update_from_output(scheduler_output, model_runner_output)
        if got:
            rec, t1 = got
            rec['exec_s'] = t - t1
            rec['t_end'] = t
            if self.cp_file:
                self.cp_file.write(json.dumps(rec, separators=(',', ':')) + '\n')
            for f in self.cp_listeners:
                f(rec)
        return res
