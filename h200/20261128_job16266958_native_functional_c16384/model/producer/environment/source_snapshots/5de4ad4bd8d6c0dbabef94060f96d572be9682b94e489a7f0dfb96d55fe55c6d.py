"""CPU request accounting for portable H200 inputs, never GPU qualification.

SLOs and interpolated per-request quantiles follow the pinned AMD amendment 7
and cp_analyze.disagg_trace. Exact output references and the original offered
denominator are additional H200 correctness gates. Host-visible token gaps do
not measure decoder kernel/step latency. Raw inputs must remain with the run.
"""
import math
from h200_portable_serving_trace import validate

STALL_MS = (250, 500, 1000, 2000)
SLO = {
    'lclo': {'ttft_s': 25., 'itl_ms': 110., 'itl_tight_ms': 50.},
    'lctx': {'ttft_s': 25., 'itl_ms': 110., 'itl_tight_ms': 50.},
    'lprompt': {'ttft_s': 41., 'itl_ms': 110., 'itl_tight_ms': 50.},
    'mixdoc': {'ttft_s': 2., 'long_prompt_min': 16384,
               'ttft_long_s': 41., 'itl_ms': 110., 'itl_tight_ms': 50.}}


def percentile(values, fraction):
    assert 0 <= fraction <= 1
    if not values:
        return None
    ordered = sorted(values)
    position = fraction*(len(ordered)-1)
    lo = int(position)
    return ordered[lo]+(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])*(position-lo)


def score(trace, records, expected_outputs, observation_end_s):
    """Records use offsets from one client monotonic epoch, including failures.

    Each record has id, status, submitted_s (or None), output_ids and
    token_observed_s. A completed record additionally has finish_reason=length.
    Missing records become explicit offered_no_record failures. Never infer
    successful output correctness from counts or a caller's correctness flag.
    """
    validate(trace)
    assert trace['workload'] in SLO, 'other workload SLOs require a prospectively frozen pilot rule'
    assert math.isfinite(observation_end_s) and observation_end_s > 0
    wanted = {q['id']:q for q in trace['requests']}
    observed = {}
    for record in records:
        rid = record['id']
        assert rid in wanted and rid not in observed, 'duplicate or unoffered request'
        observed[rid] = record
    assert not (set(expected_outputs)-set(wanted)), 'unoffered reference request'
    rule = dict(SLO[trace['workload']])
    per_request = []
    for index, request in enumerate(trace['requests']):
        rid = request['id']
        record = observed.get(rid, {'id':rid, 'status':'offered_no_record',
            'submitted_s':None,'output_ids':[],'token_observed_s':[]})
        ids = record['output_ids']; times = record['token_observed_s']; submitted = record['submitted_s']
        assert len(ids) == len(times) <= request['max_tokens']
        assert all(type(token) is int and 0 <= token < trace['model']['vocab_size'] for token in ids)
        assert all(math.isfinite(t) and request['arrival_s'] <= t <= observation_end_s for t in times)
        assert all(a <= b for a,b in zip(times,times[1:])), 'client token timestamps moved backward'
        if submitted is None:
            assert not ids, 'output without a submitted request'
        else:
            assert math.isfinite(submitted) and request['arrival_s'] <= submitted <= observation_end_s
            assert not times or times[0] >= submitted
        reference = expected_outputs.get(rid)
        if reference is not None:
            assert len(reference) == request['max_tokens'], 'incomplete expected-output reference'
            assert all(type(token) is int and 0 <= token < trace['model']['vocab_size'] for token in reference)
        complete = record['status']=='complete' and len(ids)==request['max_tokens'] and record.get('finish_reason')=='length'
        correct = reference is not None and ids == reference
        gaps_ms = [1000*(b-a) for a,b in zip(times,times[1:])]
        ttft = times[0]-request['arrival_s'] if times else None
        p95 = percentile(gaps_ms,.95)
        maximum = max(gaps_ms) if gaps_ms else None
        document = trace['workload']=='mixdoc' and len(request['prompt_ids']) >= rule['long_prompt_min']
        ttft_limit = rule['ttft_long_s'] if document else rule['ttft_s']
        eligible = complete and correct
        ttft_ok = eligible and ttft <= ttft_limit
        primary = ttft_ok and p95 is not None and p95 <= rule['itl_ms']
        per_request.append({'id':rid,'window':'warmup' if index < trace['warmup_requests'] else 'measured',
            'class':'document' if document else 'chat' if trace['workload']=='mixdoc' else 'all',
            'offered_s':request['arrival_s'],'submitted_s':submitted,'status':record['status'],
            'attempted':submitted is not None,'completed_by_count_and_length':complete,
            'exact_reference_available':reference is not None, 'exact_outputs_passed':eligible,
            'output_token_count':len(ids),'admission_delay_s':None if submitted is None else submitted-request['arrival_s'],
            'ttft_s':ttft,'ttft_limit_s':ttft_limit,'request_itl_p95_ms':p95,'request_max_gap_ms':maximum,
            'primary':primary,'itl50':ttft_ok and p95 is not None and p95 <= rule['itl_tight_ms'],
            **{'stall'+str(cap):primary and maximum is not None and maximum <= cap for cap in STALL_MS}})
    measured = [r for r in per_request if r['window']=='measured']
    assert len(measured)>=2
    offered_span = measured[-1]['offered_s']-measured[0]['offered_s']
    assert offered_span>=0
    including_drain = observation_end_s-measured[0]['offered_s']
    assert including_drain>0
    metrics = ('primary','stall250','stall500','stall1000','stall2000','itl50')

    def group(rows, measured_window=True):
        denom = len(rows)
        completed = [r for r in rows if r['completed_by_count_and_length']]
        return {'offered':denom,'attempted':sum(r['attempted'] for r in rows),
            'completed_by_count_and_length':len(completed),
            'exact_reference_completed':sum(r['exact_outputs_passed'] for r in rows),
            'incomplete':denom-len(completed),
            'completed_without_exact_reference':sum(not r['exact_reference_available'] for r in completed),
            'completed_with_wrong_outputs':sum(r['exact_reference_available'] and not r['exact_outputs_passed'] for r in completed),
            'conditional_completed_ttft_p50_s':percentile([r['ttft_s'] for r in completed],.5),
            'conditional_completed_ttft_p99_s':percentile([r['ttft_s'] for r in completed],.99),
            'conditional_completed_request_max_gap_p99_ms':percentile([r['request_max_gap_ms'] for r in completed],.99),
            'objectives':{name:{'compliant':sum(bool(r[name]) for r in rows),
                'compliant_output_tokens':sum(r['output_token_count'] for r in rows if r[name]),
                'attainment':sum(bool(r[name]) for r in rows)/denom if denom else None,
                'compliant_requests_per_s_over_measured_offered_span':sum(bool(r[name]) for r in rows)/offered_span if measured_window and offered_span>0 else None,
                'compliant_output_tokens_per_s_over_measured_offered_span':sum(r['output_token_count'] for r in rows if r[name])/offered_span if measured_window and offered_span>0 else None,
                'compliant_requests_per_s_including_drain':sum(bool(r[name]) for r in rows)/including_drain if measured_window else None,
                'compliant_output_tokens_per_s_including_drain':sum(r['output_token_count'] for r in rows if r[name])/including_drain if measured_window else None}
                for name in metrics}}
    return {'classification':'CPU accounting of supplied client request observations; no independent GPU qualification',
        'trace_sha256':trace['sha256'],'AMD_trace_sha256':trace['AMD_trace_sha256'],
        'workload':trace['workload'],'arrival':trace['arrival'],'rate':trace['configured_rate_requests_s'],
        'seed':trace['seed'],'split':trace['split'],'slo':rule,'stall_ms':list(STALL_MS),
        'all_offered':len(per_request),'all_attempted':sum(r['attempted'] for r in per_request),
        'all_host_observed_output_tokens':sum(r['output_token_count'] for r in per_request),
        'all_failed_or_incomplete_ids':[r['id'] for r in per_request if not r['exact_outputs_passed']],
        'missing_record_ids':[rid for rid in wanted if rid not in observed],
        'warmup':group([r for r in per_request if r['window']=='warmup'],False),
        'measured':group(measured),
        'measured_by_class':{name:group([r for r in measured if r['class']==name]) for name in sorted({r['class'] for r in measured})},
        'observation_end_s':observation_end_s,'measured_offered_span_s':offered_span,
        'measured_window_including_drain_s':including_drain,'per_request':per_request,
        'steady_state_demonstrated':False,'quantitative_performance_qualified':False,
        'decoder_step_latency_ms':None,'primary_J_per_token':None,
        'scope':'First10percent are retained warmups. Every remaining offered request remains in the SLO denominator, including failures, missing records and timeouts. Conditional latency summaries retain counts. Client token gaps include queue/transport effects and are separate from clean decoder-step latency. Rate selection, source/clock/isolation/reference provenance, independent acquisitions and calibrated all-board energy need separate qualification.'}


def sustainable(rows, rates, seeds, objective='primary'):
    """All declared seeds must pass at this and every lower declared rate.

    Rows are separately qualified summaries with source cohort already checked
    by the caller. Missing/duplicate cells cannot silently shorten the ladder.
    This helper never qualifies a supplied row or merges different cohorts.
    """
    assert rates == sorted(set(rates)) and rates and all(math.isfinite(r) and r>0 for r in rates)
    assert seeds and len(set(seeds))==len(seeds)
    indexed={}
    for row in rows:
        key=(row['rate'],row['seed'])
        assert key[0] in rates and key[1] in seeds and key not in indexed
        indexed[key]=row
    best=None; boundary=None; censored=False
    for rate in rates:
        failures=[]
        for seed in seeds:
            row=indexed.get((rate,seed))
            if row is None:
                failures.append({'seed':seed,'reason':'missing_cell'}); censored=True; continue
            assert type(row['scientific_qualified']) is bool
            assert type(row['offered']) is int and row['offered']>0
            assert type(row['incomplete']) is int and 0<=row['incomplete']<=row['offered']
            assert type(row['exact_reference_completed']) is int and 0<=row['exact_reference_completed']<=row['offered']-row['incomplete']
            attainment=row['objectives'][objective]['attainment']
            assert isinstance(attainment,(int,float)) and math.isfinite(attainment) and 0<=attainment<=1
            if row.get('engine_abort',False):
                failures.append({'seed':seed,'reason':'engine_abort'}); censored=True
            elif not row['scientific_qualified']:
                failures.append({'seed':seed,'reason':'unqualified_cell'}); censored=True
            elif row['incomplete'] != 0 or row['exact_reference_completed'] != row['offered']:
                failures.append({'seed':seed,'reason':'incomplete_or_incorrect_requests'})
            elif attainment < .9:
                failures.append({'seed':seed,'reason':'SLO_attainment_below_0.9'})
        if failures:
            boundary={'rate':rate,'failures':failures}; break
        best=rate
    return {'sustainable_rate_requests_s':best,'first_unpassed_rate':boundary,
        'top_of_ladder_passed':boundary is None,'crash_missing_or_unqualified_boundary':censored,
        'right_censored':boundary is None or censored,'required_seeds':list(seeds),'declared_rates':list(rates),
        'scope':'Only the complete lower-rate prefix qualifies. A censored frontier is a lower bound, never evidence of a higher passing capacity or a crash-caused architectural loss.'}
