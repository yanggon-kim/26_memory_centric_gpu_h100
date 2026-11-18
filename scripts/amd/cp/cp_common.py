"""Chunked-prefill campaign: shared engine construction and records (in-process vLLM 0.23 engine in the container).

Engine settings follow scripts/amd/goal/serve_mix.py (pinned model directory = revision, BF16, TP1, greedy, EOS ignored,
uni executor, in-process engine core) with the campaign's scheduler (scripts/amd/cp/cp_sched.py) and explicit
chunked-prefill / prefix-caching switches. Everything a run used is written to run.json before the engine starts.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # scripts/
from software_decode_experiments import Sensor, digest, dump  # noqa: E402,F401

# CUDA/HIP graph policy common to every engine of the campaign: vLLM's own capture-size pattern (1, 2, 4, then every 8
# up to 256, then every 16) up to 512 tokens, the limit stock `vllm serve` uses at its default max_num_seqs. vLLM would
# otherwise derive the limit from max_num_seqs (2 x max_num_seqs), so it would differ between harness settings. Mixed
# iterations above 512 tokens run eagerly in stock serving and here (recorded per step as graph path). Dev run 2026-11-18:
# a custom list up to 2048 padded a 1032-token step to 1280 and made it slower than eager (43.8 vs 35.7 ms, Qwen2.5-7B).
COMPILATION = {'max_cudagraph_capture_size': 512}
ENGINE_KEYS = ('max_model_len', 'max_num_seqs', 'max_num_batched_tokens', 'gpu_memory_utilization',
               'enable_chunked_prefill', 'enable_prefix_caching', 'long_prefill_token_threshold', 'enforce_eager')


def setup_env():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
                      VLLM_ENABLE_V1_MULTIPROCESSING='0')


def open_gpu(uuid):
    from gpu_target import load_nvml, select, verify_cuda
    nvml = load_nvml()
    nvml.nvmlInit()
    handle, physical = select(nvml, uuid)
    import torch
    verify_cuda(torch, uuid)
    return nvml, handle, physical, torch


def hw(nvml, handle, torch):
    from gpu_target import describe
    return describe(nvml, handle, torch)


def engine_settings(cfg, over):
    path = Path(cfg['model_path'])
    assert path.is_dir() and path.name == cfg['model_revision'], path
    s = {'model': str(path), 'dtype': 'bfloat16', 'seed': cfg['seed'], 'generation_config': 'vllm',
         'enable_prefix_caching': False, 'enable_chunked_prefill': True, 'disable_log_stats': True,
         'async_scheduling': False, 'distributed_executor_backend': 'uni', 'tensor_parallel_size': 1,
         'max_model_len': 16384, 'max_num_seqs': 64, 'max_num_batched_tokens': 2048,
         'gpu_memory_utilization': 0.9, 'enforce_eager': False}
    s.update({k: v for k, v in over.items() if k in ENGINE_KEYS and v is not None})
    return s


def build(cfg, over, policy, steplog):
    """Returns (llm, engine, scheduler, settings). CP_POLICY / CP_STEPLOG are read by the scheduler at construction."""
    os.environ['CP_POLICY'] = policy
    if steplog:
        os.environ['CP_STEPLOG'] = str(steplog)
    import vllm
    from cp_sched import PolicyScheduler
    s = engine_settings(cfg, over)
    s['compilation_config'] = dict(COMPILATION)
    llm = vllm.LLM(scheduler_cls=PolicyScheduler, **s)
    engine = llm.llm_engine
    sched = engine.engine_core.engine_core.scheduler
    assert isinstance(sched, PolicyScheduler), type(sched)
    return llm, engine, sched, s


def model_layout(cfg):
    c = json.loads((Path(cfg['model_path']) / 'config.json').read_text())
    hd = c.get('head_dim') or c['hidden_size'] // c['num_attention_heads']
    kv = 2 * c['num_hidden_layers'] * c['num_key_value_heads'] * hd * 2  # K and V, BF16
    return {'layers': c['num_hidden_layers'], 'kv_heads': c['num_key_value_heads'], 'head_dim': hd,
            'attention_heads': c['num_attention_heads'], 'hidden': c['hidden_size'], 'vocab': c['vocab_size'],
            'sliding_window': c.get('sliding_window') if c.get('use_sliding_window') else None,
            'precision': 'BF16 weights and KV', 'kv_bytes_per_token': kv}


class Tokens:
    """Seeded synthetic prompts (uniform ids over the vocabulary, as serve_mix.py). Every call returns new ids."""

    def __init__(self, vocab, seed):
        self.rng = random.Random(seed)
        self.vocab = vocab

    def __call__(self, n):
        r = self.rng
        return [r.randrange(self.vocab) for _ in range(n)]


def sampling(vllm, max_tokens, seed):
    from vllm.sampling_params import RequestOutputKind
    return vllm.SamplingParams(temperature=0, repetition_penalty=1.0, ignore_eos=True, detokenize=False,
                               max_tokens=max_tokens, seed=seed, output_kind=RequestOutputKind.CUMULATIVE)


def versions(torch):
    import vllm
    return {'vllm': vllm.__version__, 'torch': torch.__version__, 'hip': torch.version.hip,
            'image': os.environ.get('AMD_VLLM_IMAGE'), 'image_sha256': os.environ.get('AMD_VLLM_IMAGE_SHA256')}


def env_record():
    return {k: os.environ.get(k, 'unset') for k in ('HIP_VISIBLE_DEVICES', 'HSA_CU_MASK', 'CUBW_CU_COUNT_OVERRIDE',
                                                     'CP_POLICY', 'VLLM_ENABLE_V1_MULTIPROCESSING', 'CUBW_STEP_TAG')}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def engine_info(engine, llm):
    """Effective attention backend / compilation settings, best effort (recorded, never asserted)."""
    out = {}
    try:
        vc = engine.vllm_config
        cc = vc.compilation_config
        out['cudagraph_mode'] = str(getattr(cc, 'cudagraph_mode', None))
        out['cudagraph_capture_sizes'] = list(getattr(cc, 'cudagraph_capture_sizes', None) or [])
        out['max_cudagraph_capture_size'] = getattr(cc, 'max_cudagraph_capture_size', None)
        out['block_size'] = vc.cache_config.block_size
        out['num_gpu_blocks'] = vc.cache_config.num_gpu_blocks
        out['attention_backend'] = os.environ.get('VLLM_ATTENTION_BACKEND') or str(getattr(vc.attention_config, 'backend', None)
                                                                                 if hasattr(vc, 'attention_config') else None)
        out['max_num_batched_tokens'] = vc.scheduler_config.max_num_batched_tokens
        out['enable_chunked_prefill'] = vc.scheduler_config.enable_chunked_prefill
    except Exception as e:  # noqa: BLE001
        out['error'] = repr(e)[:300]
    return out


def now():
    return time.perf_counter()
