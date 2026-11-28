"""Prospective variable-length client capture, not a GPU campaign entrypoint.

Caller owns engine construction, model/source/HF gates, verified scheduling,
allocation, observers, isolation and cleanup. This adapter measures host-visible
outputs and engine calls; it never qualifies kernel latency or serving results.
All bookkeeping and periodic ledger writes are part of this instrumented path.
"""
import math
from pathlib import Path
import time
import traceback

from h200_portable_serving_trace import validate
from h200_portable_serving_score import score


def capture(engine, sampling_factory, trace, expected_outputs, output,
            timeout_s, observer=None, clock=time):
    """Capture one original offered trace using cumulative one-token outputs.

    sampling_factory(request) must create EOS-ignored greedy sampling with that
    request's original max_tokens. The caller must verify actual settings and
    source/reference provenance independently. No trace lengths or arrivals are
    shortened here. Exceptions preserve partial observations and are re-raised.
    Caller must clean up its owned engine even when capture fails.
    """
    from software_decode_experiments import dump
    validate(trace)
    assert math.isfinite(timeout_s) and timeout_s>0
    assert not engine.has_unfinished_requests()
    output=Path(output)
    assert not output.exists()
    trace_path=output.with_suffix('.trace.json')
    assert not trace_path.exists()
    dump(trace_path,trace)
    requests={q['id']:{'id':q['id'],'status':'not_submitted','submitted_s':None,
        'submission_return_s':None,'internal_request_id':None,
        'output_ids':[],'token_observed_s':[]} for q in trace['requests']}
    internal={}
    ledger={'status':'running','classification':'instrumented host client capture pending GPU/source/reference review',
        'trace_path':str(trace_path),'trace_sha256':trace['sha256'],
        'AMD_trace_sha256':trace['AMD_trace_sha256'],'requests':list(requests.values()),
        'steps':[],'unexpected_outputs':[],'timeout_s':timeout_s,
        'engine_abort':False,'quantitative_performance_qualified':False,
        'decoder_step_latency_ms':None,'primary_J_per_token':None,
        'scope':'Original variable lengths and nondecreasing arrivals retained. Host output times include queueing and instrumentation; step host spans are separate. No HTTP, GPU-event latency, steady-state capacity, clock/isolation/reference qualification or calibrated energy is established.'}
    epoch=clock.monotonic()
    next_flush=epoch+1.
    # Even a one-token prompt chunk has a finite useful schedule bound. Empty
    # scheduler calls and preemption/recomputation are retained, not successful
    # token progress; exceeding this bound is an explicit capture limitation.
    step_limit=3*sum(len(q['prompt_ids'])+q['max_tokens'] for q in trace['requests'])+1024
    ledger['scheduler_call_limit']=step_limit
    dump(output,ledger)
    try:
        while any(r['status']!='complete' for r in requests.values()):
            elapsed=clock.monotonic()-epoch
            if elapsed>=timeout_s:
                raise TimeoutError('original offered trace exceeded declared observation bound')
            for query in trace['requests']:
                record=requests[query['id']]
                if record['submitted_s'] is None and query['arrival_s']<=elapsed:
                    record['submitted_s']=clock.monotonic()-epoch
                    record['status']='submission_started'
                    rid='h200_portable_'+trace['sha256'][:16]+'_'+query['id']
                    admitted=engine.add_request(rid,{'prompt_token_ids':query['prompt_ids']},sampling_factory(query))
                    assert isinstance(admitted,str) and admitted not in internal
                    record.update(internal_request_id=admitted,submission_return_s=clock.monotonic()-epoch,status='submitted')
                    internal[admitted]=query['id']
            if not engine.has_unfinished_requests():
                assert not any(r['submitted_s'] is not None and r['status']!='complete' for r in requests.values()),'engine lost an admitted request'
                clock.sleep(.001)
                continue
            before_rows=len(observer.rows) if observer is not None else None
            started=clock.monotonic()-epoch
            batch=engine.step()
            visible=clock.monotonic()-epoch
            step={'start_s':started,'outputs_visible_s':visible,
                'engine_call_host_ms':1000*(visible-started),'output_request_ids':[],
                'scheduler_row_indices':None,
                'offered_not_submitted':sum(q['arrival_s']<=visible and requests[q['id']]['submitted_s'] is None for q in trace['requests']),
                'submitted_unfinished_before_outputs':sum(r['submitted_s'] is not None and r['status']!='complete' for r in requests.values())}
            ledger['steps'].append(step)
            if observer is not None:
                assert len(observer.rows)==before_rows+1,'one synchronous scheduler row per engine call required'
                step['scheduler_row_indices']=[before_rows,len(observer.rows)]
                step['composition']=observer.rows[-1]['composition']
            for result in batch:
                if result.request_id not in internal:
                    ledger['unexpected_outputs'].append({'request_id':result.request_id,'observed_s':visible})
                    raise AssertionError('unoffered output request')
                record=requests[internal[result.request_id]]
                assert record['status']=='submitted' and len(result.outputs)==1
                ids=list(result.outputs[0].token_ids)
                previous=record['output_ids']
                if ids[:len(previous)]!=previous or len(ids)!=len(previous)+1:
                    ledger['unexpected_outputs'].append({'request_id':result.request_id,'observed_s':visible,'output_ids':ids,'reason':'non-single-token cumulative progress'})
                    raise AssertionError('cumulative output changed or did not advance by one token')
                # Preserve wrong greedy outputs, too. Independent scoring uses
                # actual references; a caller correctness flag is insufficient.
                previous.append(ids[-1]);record['token_observed_s'].append(visible)
                step['output_request_ids'].append(record['id'])
                query=next(q for q in trace['requests'] if q['id']==record['id'])
                assert len(previous)<=query['max_tokens'],'output exceeds original request length'
                if result.finished:
                    record.update(status='complete',finish_reason=result.outputs[0].finish_reason,finished_s=visible)
                else:
                    assert len(previous)<query['max_tokens'],'request failed to finish at its token limit'
            assert len(ledger['steps'])<=step_limit,'bounded scheduler call count exceeded'
            if clock.monotonic()>=next_flush:
                dump(output,ledger)
                next_flush=clock.monotonic()+1.
        assert not engine.has_unfinished_requests()
        ledger['status']='portable_client_capture_complete_pending_review'
        return ledger
    except BaseException as error:
        ledger.update(status='partial_or_failed_portable_client_capture',
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        # Engine versus capture failures require caller/process evidence. Do
        # not relabel an observation timeout as a demonstrated engine crash.
        for record in requests.values():
            if record['status']!='complete':
                record['status']='observation_timeout' if isinstance(error,TimeoutError) else 'capture_aborted'
        raise
    finally:
        ledger['observation_end_s']=max(clock.monotonic()-epoch,1e-12)
        if observer is not None:ledger['scheduler_rows']=observer.rows
        try:ledger['accounting']=score(trace,list(requests.values()),expected_outputs,ledger['observation_end_s'])
        except BaseException as error:
            ledger['accounting_failure']={'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}
        dump(output,ledger)
