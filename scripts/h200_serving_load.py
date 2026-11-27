#!/usr/bin/env python3
"""Versioned finite open-loop policy capture; GPU pilot/CPU qualification pending.

Per-step policy checks, event pairs and atomic ledger writes are included.
These are instrumented observations, never automatically ordinary serving SLOs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from h200_serving_trace import (make_trace, requests_for, record_output,
                               summarize, trial_seed)

POLICIES = ('whole_separate', 'chunked_separate', 'chunked_mixed')
SLO = {'ttft_s': 5., 'maximum_inter_token_s': .1}


def run_trace(engine, prompt, sampling, torch, output, policy_name, budget,
              trace, expected, timeout_s=180.):
    from h200_serving_policy import Policy
    from h200_scheduler_observer import Observer
    from software_decode_experiments import dump
    assert not engine.has_unfinished_requests() and len(expected) == 64
    requests = {f'{policy_name}_{trace["sha256"][:12]}_{rid}': record
                for rid, record in requests_for(trace).items()}
    policy = Policy(engine, policy_name, budget)
    try:
        observer = Observer(engine)
    except BaseException:
        policy.close()
        raise
    ledger = {'status': 'running', 'classification': 'measured-real-GPU-instrumented',
              'trace': trace, 'policy_identity': policy.identity,
              'observer_identity': observer.identity, 'requests': requests,
              'scheduler_rows': observer.rows, 'policy_rows': policy.rows,
              'steps': [], 'slo': SLO, 'primary_J_per_token': None,
              'partial_ledger_flush_interval_s': 1.,
              'clean_serving_qualified': False,
              'scope': 'finite pretokenized offered trace with exact outputs; policy, observer, event and ledger costs included; no automatic steady-state, clean SLO, tuning, energy or threshold claim'}
    def save():
        dump(output, ledger)
    save()
    torch.cuda.synchronize()
    start_unix_ns = time.time_ns()
    start = time.perf_counter()
    next_flush = start+1.
    ledger['start_unix_ns'] = start_unix_ns
    try:
        while any('finished_s' not in r for r in requests.values()):
            elapsed = time.perf_counter()-start
            assert elapsed < timeout_s, 'bounded serving trace deadline reached'
            for rid, record in requests.items():
                if record['admitted_s'] is None and record['offered_s'] <= elapsed:
                    admitted = engine.add_request(rid, {'prompt_token_ids': prompt}, sampling)
                    record['internal_request_id'] = admitted
                    record['admitted_s'] = time.perf_counter()-start
            if not engine.has_unfinished_requests():
                assert any(r['admitted_s'] is None for r in requests.values()), 'engine lost admitted requests'
                time.sleep(.001)
                continue
            before = len(observer.rows)
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            step_start = time.perf_counter()
            outputs = engine.step()
            # Timestamp host-visible outputs before the extra event synchronization.
            visible = time.perf_counter()-start
            end.record()
            torch.cuda.synchronize()
            step_end = time.perf_counter()-start
            assert len(observer.rows) == before+1
            for result in outputs:
                assert result.request_id in requests, 'unknown output request'
                record_output(requests[result.request_id], result, visible, expected)
            scheduled = observer.rows[-1]
            ledger['steps'].append({'start_s': step_start-start, 'outputs_visible_s': visible,
                'end_s': step_end, 'wall_ms': (step_end-(step_start-start))*1000,
                'gpu_event_ms': begin.elapsed_time(end), 'composition': scheduled['composition'],
                'outputs': [o.request_id for o in outputs],
                'offered_not_admitted': sum(r['admitted_s'] is None and r['offered_s'] <= step_end
                                          for r in requests.values()),
                'admitted_unfinished': sum(r['admitted_s'] is not None and 'finished_s' not in r
                                          for r in requests.values()),
                'actual_running': policy.rows[-1]['actual_running'],
                'actual_waiting': policy.rows[-1]['actual_waiting']})
            assert len(ledger['steps']) <= 65536, 'bounded scheduler progress exceeded'
            if time.perf_counter() >= next_flush:
                save()
                next_flush = time.perf_counter()+1.
        assert not engine.has_unfinished_requests()
        assert all(r['output_ids'] == expected for r in requests.values())
        # Current separate-policy recomputation classification is deliberately
        # not promoted to pressure robustness by this new capture adapter.
        assert not any(r['any_preempted_request'] for r in observer.rows), 'KV-pressure semantics require a separate GPU-validated protocol'
        for record in requests.values():
            assert len(record['token_observed_s']) == 64 and len(record['inter_token_s']) == 63
        expected_prompt = len(prompt)*len(requests)
        assert sum(r['prompt_input_queries'] for r in observer.rows) == expected_prompt
        assert sum(r['post_prompt_input_queries'] for r in observer.rows) == 63*len(requests)
        if policy_name != 'chunked_mixed':
            assert all(r['composition'] != 'mixed' for r in observer.rows)
        ledger['status'] = 'finite_trace_exact_outputs_and_composition_passed'
        return ledger
    except BaseException as error:
        ledger.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        ledger['end_unix_ns'] = time.time_ns()
        ledger['summary'] = summarize(requests, time.perf_counter()-start, SLO)
        observer.close()
        policy.close()
        save()


def main(args):
    import controlled_vllm_experiments as baseline
    from software_decode_experiments import dump
    original = baseline.run_trial
    captured = []
    def trial(engine, prompts, sampling, prefix, torch, output_dir, *extra, **kwargs):
        result = original(engine, prompts, sampling, prefix, torch, output_dir, *extra, **kwargs)
        if 'measured0' not in prefix:
            return result
        assert not captured and len(prompts) == 1 and sampling.max_tokens == 64
        expected = result['output_ids'][0]
        contexts = len(prompts[0])
        whole_budget = engine.engine_core.engine_core.scheduler.max_num_scheduled_tokens
        assert whole_budget >= contexts
        phases = [('warmup', 3), ('functional' if args.pilot else 'judging', 3 if args.pilot else 10)]
        manifest = {'status': 'running', 'protocol': 'finite_open_loop_policy_v1',
                    'pilot': args.pilot, 'acquisition': args.acquisition, 'rate': args.rate,
                    'context': contexts, 'tokens': 64, 'request_count': args.requests,
                    'slo': SLO, 'budget_settings': {'whole': whole_budget, 'chunked': args.chunk_budget},
                    'captures': captured, 'quantitative_qualified': False,
                    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        for phase, count in phases:
            for number in range(count):
                seed = trial_seed(args.acquisition, number,
                                  'development' if phase == 'functional' else phase)
                trace = make_trace(seed, args.rate, args.requests,
                                   'functional' if phase == 'functional' else phase)
                rotation = (args.acquisition+number) % 3
                order = POLICIES[rotation:]+POLICIES[:rotation]
                for name in order:
                    path = output_dir/f'{phase}{number}_{name}.json'
                    row = {'phase': phase, 'trial': number, 'policy': name,
                           'trace_sha256': trace['sha256'], 'trace_seed': seed, 'path': str(path),
                           'status': 'running'}
                    captured.append(row)
                    dump(output_dir/'serving_load.json', manifest)
                    try:
                        data = run_trace(engine, prompts[0], sampling, torch, path, name,
                                         whole_budget if name == 'whole_separate' else args.chunk_budget,
                                         trace, expected, args.trace_timeout)
                        row.update(status=data['status'], sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    except BaseException as error:
                        row.update(status='failed', error=str(error))
                        manifest['status'] = 'partial_failed'
                        raise
                    finally:
                        dump(output_dir/'serving_load.json', manifest)
        manifest['status'] = 'captured_pending_CPU_and_intrusion_review'
        dump(output_dir/'serving_load.json', manifest)
        return result
    baseline.run_trial = trial
    try:
        baseline.main(args)
    finally:
        baseline.run_trial = original
    assert len(captured) == (18 if args.pilot else 39)
    path = args.output/'run.json'
    data = json.loads(path.read_text())
    data.update(baseline_comparison_eligible=False,
                experimental_scope='finite open-loop instrumented policy captures; serial baseline retained only as exact-output reference')
    dump(path, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--requests', type=int, default=24)
    parser.add_argument('--acquisition', type=int, choices=range(3), required=True)
    parser.add_argument('--chunk-budget', type=int, default=256)
    parser.add_argument('--trace-timeout', type=float, default=180.)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--mode', choices=['timing'], default='timing')
    parser.add_argument('--generated-tokens', type=int, choices=[64], default=64)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--context', type=int)
    parser.set_defaults(external_profiler=False, collect_worker_metadata=False)
    args = parser.parse_args()
    assert math.isfinite(args.rate) and args.rate > 0 and args.requests >= 12
    assert math.isfinite(args.trace_timeout) and args.trace_timeout > 0
    assert args.chunk_budget >= 1
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except BaseException as error:
        from software_decode_experiments import dump
        dump(args.output/'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
