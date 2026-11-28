"""Experimental synchronous serving policies; GPU pilot is required before use.

Masking a request from one schedule call does not evict its KV or mark it
finished. Held requests still consume the original seat and KV capacity.
No installed vLLM source is edited. Only this owned in-process scheduler is
wrapped. This is a scheduler intervention with explicit instrumentation cost.
"""
import hashlib
import inspect
from pathlib import Path
from h200_scheduler_observer import composition

PINNED_SCHEDULER_SHA256='41ff2e524c90d9aa72b72cd77492eb62ee2a729a773bd8233e970f39abbb5983'
POLICIES=('whole_separate','chunked_separate','chunked_mixed')


def is_prefill(request):
    return request.num_computed_tokens<request.num_prompt_tokens


def schedule_phase(scheduler,original,phase,empty_queue):
    """Select one role, preserving held seats and queue object identities."""
    assert phase in ('prefill','decode','empty')
    running=list(scheduler.running)
    held=[r for r in running if (is_prefill(r)!=(phase=='prefill'))]
    eligible=[r for r in running if r not in held]
    waiting,skipped=scheduler.waiting,scheduler.skipped_waiting
    seats=scheduler.max_num_running_reqs
    scheduler.running=eligible
    scheduler.max_num_running_reqs=seats-len(held)
    assert scheduler.max_num_running_reqs>=0
    if phase!='prefill':
        scheduler.waiting=empty_queue();scheduler.skipped_waiting=empty_queue()
    try:return original()
    finally:
        if phase!='prefill':
            # Native KV preemption may enqueue an eligible decoder even while
            # admission is masked. Preserve that mutation before restoring the
            # original queues. Reversed individual prepends preserve FCFS order;
            # priority queues retain their native comparator semantics.
            for destination,temporary in ((waiting,scheduler.waiting),(skipped,scheduler.skipped_waiting)):
                for request in reversed(list(temporary)):destination.prepend_request(request)
        # Preserve legitimate admission/preemption performed by the original
        # scheduler, then put held requests back in their original FCFS order.
        combined=list(scheduler.running)+held
        identities=[r.request_id for r in combined]
        scheduler.waiting=waiting;scheduler.skipped_waiting=skipped
        scheduler.max_num_running_reqs=seats
        if len(set(identities))!=len(identities):scheduler.running=running
        assert len(set(identities))==len(identities),'request duplicated by policy mask'
        by_id={r.request_id:r for r in combined}
        order=[r.request_id for r in running if r.request_id in by_id]
        order.extend(r.request_id for r in combined if r.request_id not in order)
        scheduler.running=[by_id[rid] for rid in order]
        assert len(scheduler.running)<=seats,'held requests must retain their seats'


class Policy:
    def __init__(self,engine,policy,token_budget):
        assert policy in POLICIES
        assert type(engine.engine_core).__name__=='InprocClient'
        self.scheduler=engine.engine_core.engine_core.scheduler
        source=Path(inspect.getsourcefile(type(self.scheduler)))
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        assert source_hash==PINNED_SCHEDULER_SHA256,'scheduler implementation requires a new versioned review'
        config=self.scheduler.scheduler_config
        assert not config.async_scheduling and self.scheduler.connector is None
        self.original=self.scheduler.schedule
        self.original_budget=self.scheduler.max_num_scheduled_tokens
        assert isinstance(token_budget,int) and 0<token_budget<=self.original_budget
        self.policy=policy;self.budget=token_budget;self.next_prefill=True
        self.rows=[];self.closed=False
        self.identity={'version':'separate_policy_v2','policy':policy,'token_budget':token_budget,
            'scheduler_source_sha256':source_hash,'adapter_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'clean_timing_eligible':False,
            'scope':'experimental policy with per-step checks; held requests keep seats and KV; internal vLLM running-count statistics during masking are not used as actual queue depth'}
        from vllm.v1.core.sched.request_queue import create_request_queue
        def schedule():
            scheduler=self.scheduler;cfg=scheduler.scheduler_config
            budget=scheduler.max_num_scheduled_tokens
            chunked=cfg.enable_chunked_prefill;threshold=cfg.long_prefill_token_threshold
            has_prefill=bool(scheduler.waiting or scheduler.skipped_waiting or any(is_prefill(r) for r in scheduler.running))
            has_decode=any(not is_prefill(r) for r in scheduler.running)
            if has_prefill and has_decode:
                phase='prefill' if self.next_prefill else 'decode'
                self.next_prefill=not self.next_prefill
            else:phase='prefill' if has_prefill else 'decode' if has_decode else 'empty'
            try:
                scheduler.max_num_scheduled_tokens=self.budget
                cfg.enable_chunked_prefill=policy!='whole_separate'
                cfg.long_prefill_token_threshold=0 if policy=='whole_separate' else self.budget
                if policy=='chunked_mixed':output=self.original()
                else:output=schedule_phase(scheduler,self.original,phase,lambda:create_request_queue(scheduler.policy))
            finally:
                scheduler.max_num_scheduled_tokens=budget
                cfg.enable_chunked_prefill=chunked;cfg.long_prefill_token_threshold=threshold
            observed=composition(output,scheduler.requests)
            observed.update(policy=policy,requested_phase=phase,token_budget=self.budget,
                            actual_running=len(scheduler.running),actual_waiting=len(scheduler.waiting)+len(scheduler.skipped_waiting))
            self.rows.append(observed)
            assert output.total_num_scheduled_tokens<=self.budget
            if policy!='chunked_mixed':assert observed['composition']!='mixed','separate policy emitted a mixed step'
            if policy=='whole_separate':
                for row in observed['requests']:
                    if row['prompt_input_queries']:
                        assert row['computed_tokens_before_step']==0 and row['prompt_input_queries']==row['original_prompt_tokens'], 'whole policy emitted partial prompt'
            return output
        self.hook=schedule;self.scheduler.schedule=schedule

    def close(self):
        if self.closed:return
        assert self.scheduler.schedule is self.hook,'close outer observer before policy adapter'
        self.scheduler.schedule=self.original;self.closed=True
