"""Prospective exact-input adapter for a future serial-reference capture.

CPU only; not integrated into the active/queued numerical program. Returning
prompt IDs does not establish model correctness or qualify timing/reference
reuse. Caller must version its integration and acquire the unchanged full HF
dynamic/StaticCache gates before serving measurements.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]/'scripts'))
from h200_portable_serving_trace import validate


def portable_prompts(config, batch, context, count):
    assert config['precision']=='BF16' and config['seed']==20261013
    assert not config.get('workload_file'), 'no simultaneous text/prompt routes'
    assert not config.get('decode_conditioning'), 'unconditioned serial reference required'
    path=Path(config['portable_trace_file']).resolve()
    raw=path.read_bytes()
    file_sha=hashlib.sha256(raw).hexdigest()
    assert file_sha==config['portable_trace_file_sha256'], 'trace file differs from config pin'
    trace=json.loads(raw)
    validate(trace)
    assert (config['model_id'],config['model_revision'])==(trace['model']['id'],trace['model']['revision'])
    selected=[r for r in trace['requests'] if r['id']==config['portable_request_id']]
    assert len(selected)==1, 'exactly one offered request required'
    request=selected[0]
    assert batch==1 and context==len(request['prompt_ids']) and count==request['max_tokens'], 'full original request shape required'
    assert context+count<=trace['model']['max_positions']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==file_sha, 'trace changed during read'
    return [list(request['prompt_ids'])], {
        'version':'portable_AMD_full_request_ids_v1',
        'file':str(path), 'file_sha256':file_sha,
        'trace_sha256':trace['sha256'], 'AMD_trace_sha256':trace['AMD_trace_sha256'],
        'AMD_generator_sha256':trace['AMD_generator_sha256'],
        'request_id':request['id'], 'input_seed':trace['seed'], 'split':trace['split'],
        'full_prompt_tokens':context, 'full_output_tokens':count,
        'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selection':'One exact original full request; no tokenizer substitution, prompt prefix, output shortening or arrival modification. Serial reference capture has no offered-load timing claim.',
        'GPU_execution':False, 'correctness_qualified':False}
