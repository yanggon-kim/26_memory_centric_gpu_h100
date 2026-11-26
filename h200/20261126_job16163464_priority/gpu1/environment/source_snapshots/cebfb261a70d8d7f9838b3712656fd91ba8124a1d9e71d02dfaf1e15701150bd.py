#!/usr/bin/env python3
"""Read-only vLLM V1 schedule observer for a separately identified serving pilot.

Input-query composition is not generated-token throughput. Observed step wall
times include this observer's overhead and need a matched intrusion control.
No scheduler policy or token budget is changed by this module.
"""
import hashlib
import inspect
import time
from pathlib import Path


def composition(output, requests):
    """Decode the explicit pre-step offsets retained in SchedulerOutput.

    Scheduler.schedule() has already advanced live request offsets when it
    returns. Never use those advanced offsets to classify the current step.
    """
    assert not output.scheduled_spec_decode_tokens, 'speculative decoding unsupported'
    assert not output.scheduled_encoder_inputs, 'encoder input unsupported'
    offsets = {}
    for row in output.scheduled_new_reqs:
        assert row.req_id not in offsets
        assert row.prompt_token_ids is not None, 'token-ID prompts required'
        offsets[row.req_id] = (row.num_computed_tokens, len(row.prompt_token_ids))
    cached = output.scheduled_cached_reqs
    assert len(cached.req_ids) == len(cached.num_computed_tokens)
    for rid, computed in zip(cached.req_ids, cached.num_computed_tokens):
        assert rid not in offsets, 'duplicate new/cached request'
        offsets[rid] = (computed, requests[rid].num_prompt_tokens)
    scheduled = output.num_scheduled_tokens
    assert set(offsets) == set(scheduled), 'scheduled input metadata incomplete'
    assert sum(scheduled.values()) == output.total_num_scheduled_tokens
    rows = []
    for rid, count in scheduled.items():
        computed, prompt = offsets[rid]
        assert count > 0 and computed >= 0 and prompt > 0
        prompt_queries = max(0, min(count, prompt - computed))
        rows.append({'request_id': rid, 'computed_tokens_before_step': computed,
                     'original_prompt_tokens': prompt, 'scheduled_input_queries': count,
                     'prompt_input_queries': prompt_queries,
                     'post_prompt_input_queries': count - prompt_queries,
                     'num_preemptions': requests[rid].num_preemptions})
    prefill = sum(r['prompt_input_queries'] for r in rows)
    decode = sum(r['post_prompt_input_queries'] for r in rows)
    return {'requests': rows, 'prompt_input_queries': prefill,
            'post_prompt_input_queries': decode,
            'composition': ('mixed' if prefill and decode else 'prefill_only' if prefill
                            else 'decode_only' if decode else 'empty'),
            'any_preempted_request': any(r['num_preemptions'] for r in rows),
            'interpretation': 'input queries relative to original prompt boundary; '
                              'post-prompt queries can include recomputation after preemption; '
                              'not a generated-output-token count'}


class Observer:
    """Attach only after asserting the pinned in-process synchronous engine."""
    def __init__(self, engine):
        assert type(engine.engine_core).__name__ == 'InprocClient'
        self.scheduler = engine.engine_core.engine_core.scheduler
        self.original = self.scheduler.schedule
        self.rows = []
        self.closed = False
        source = Path(inspect.getsourcefile(type(self.scheduler)))
        self.identity = {'scheduler_class': type(self.scheduler).__qualname__,
                         'source_path': str(source),
                         'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'observer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'clean_timing_eligible': False,
                         'scope': 'instrumented scheduling decisions; matched overhead pilot required'}

        def observed(*args, **kwargs):
            before = {'running': len(self.scheduler.running),
                      'waiting': len(self.scheduler.waiting)}
            start = time.time_ns()
            output = self.original(*args, **kwargs)
            result = composition(output, self.scheduler.requests)
            result.update(index=len(self.rows), schedule_start_unix_ns=start,
                          observation_end_unix_ns=time.time_ns(), queue_before=before,
                          queue_after={'running': len(self.scheduler.running),
                                       'waiting': len(self.scheduler.waiting)})
            self.rows.append(result)
            return output

        self.hook = observed
        self.scheduler.schedule = observed

    def close(self):
        if not self.closed:
            assert self.scheduler.schedule is self.hook, 'another observer replaced this hook'
            self.scheduler.schedule = self.original
            self.closed = True
